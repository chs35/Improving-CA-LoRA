"""Train LoRA on a previously computed projection selection.

    python scripts/train_lora.py --config configs/tiny_cpu.yaml \
        --selection selection_multi_t.json --run-name multi_t

Every ablation arm is this same command with a different ``--selection``, so any
difference in the results is attributable to the projection choice alone.

With ``train.stability_every > 0`` the run also produces
``stability.json`` — the CA-drift curve. It changes no training number, but each
snapshot is a full CA measurement, so it does cost wall clock; the log and
``train_summary.json`` report that time separately from the optimisation steps.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calora import cli  # noqa: E402
from calora.analysis.drift_data import drift_samples  # noqa: E402
from calora.analysis.stability import (  # noqa: E402
    MultiAxisStabilityTracker,
    build_multi_axis_stability_tracker,
    make_stability_callback,
)
from calora.ca.select import Selection  # noqa: E402
from calora.data import apply_training_subset, build_dataset  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.models.projections import ProjectionTable  # noqa: E402
from calora.train.latent_store import LatentStore  # noqa: E402
from calora.train.lora import resolve_target_modules, train_lora  # noqa: E402
from calora.utils.io import dump_json, load_json  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402
from calora.utils.plotting import plot_loss_curve  # noqa: E402

logger = get_logger("train_lora")


def main() -> None:
    parser = cli.base_parser(__doc__ or "")
    parser.add_argument(
        "--selection", required=True, help="Selection JSON (path, or filename inside the run dir)"
    )
    parser.add_argument(
        "--run-name", default=None, help="Subdirectory under the run dir for this arm"
    )
    parser.add_argument(
        "--baseline-timestep",
        type=int,
        default=None,
        help="Timestep for CA drift measurement (default: the selection's first timestep)",
    )
    args = parser.parse_args()
    cfg = cli.resolve(args)

    selection_path = Path(args.selection)
    if not selection_path.exists():
        selection_path = cfg.run_dir / args.selection
    selection_data = load_json(selection_path)

    output_dir = cfg.run_dir / (args.run_name or selection_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(cfg.model)
    # The selection file decides the granularity, not the config: an arm has to be
    # trained on the units it was selected as, and a config edit must not silently
    # reinterpret a saved selection.
    granularity = Selection.from_dict(selection_data).granularity
    if granularity != cfg.select.granularity:
        logger.warning(
            "Selection %s is %s-granularity but select.granularity=%s; following the selection.",
            selection_path, granularity, cfg.select.granularity,
        )
    table = ProjectionTable.from_model(adapter.unet, granularity=granularity)
    selection = Selection.from_dict(selection_data, table)
    logger.info(
        "Arm '%s' [%s]: %d units over %d matrices, %s params (%.3f%% of attention)",
        selection.strategy,
        selection.granularity,
        len(selection),
        len(selection.module_names()),
        f"{selection.params_selected:,}",
        100 * selection.fraction_achieved,
    )

    # Resolved now rather than inside train_lora: a selection the model cannot
    # express has to fail before the drift baseline below, which is a full CA
    # measurement (minutes on an A100) that would be thrown away.
    resolve_target_modules(adapter.unet, selection)

    # The axis is a property of the selection, not of the config: one config
    # trains both the style and the viewpoint adapter of an arm, and the two
    # checkpoints are otherwise indistinguishable.
    axis = str(selection.meta.get("axis", "") or "")
    # What drift is tracked against, read before the label fallback below: a
    # control selected on no axis has no axis of its own to be judged on.
    drift_primary_axis = axis or None
    if not axis:
        axis = cfg.select.axis
        logger.warning(
            "Selection %s records no axis; labelling this arm with select.axis=%r. "
            "Re-run select_projections.py if that is not what it was selected on.",
            selection_path, axis,
        )
    elif axis != cfg.select.axis:
        logger.warning(
            "Selection %s was made on axis %r but select.axis=%r; following the selection.",
            selection_path, axis, cfg.select.axis,
        )

    dataset = build_dataset(cfg.data, resolution=cfg.model.resolution, seed=cfg.train.seed)
    training_subset = _restrict_to_training_subset(cfg, dataset)

    callbacks = []
    tracker: MultiAxisStabilityTracker | None = None
    baseline_seconds = 0.0
    if cfg.train.stability_every > 0:
        baseline_started = time.perf_counter()
        # A second, centre-cropped and un-mirrored view of the same data. Drift is
        # a comparison of CA measured at step 0 against CA measured at step N, so
        # those two measurements have to see the same pixels even when training is
        # reading random crops and random flips out of `dataset`.
        # x0 per measure.drift_source: the base model's self-generated set (the
        # selection's own protocol) -- fixed, or regenerated from the current
        # weights before each snapshot -- or real images (crop="center",
        # hflip=False inside drift_samples, as at every measurement call site).
        samples, sample_provider, drift_x0 = drift_samples(cfg, adapter)
        logger.info(
            "Drift samples: %d, x0 %s (%s)",
            len(samples), cfg.measure.drift_source, drift_x0,
        )
        tracker = build_multi_axis_stability_tracker(
            adapter,
            samples,
            axes=cfg.data.concept_axes,
            timestep=_baseline_timestep(cfg, selection, args.baseline_timestep),
            primary_axis=drift_primary_axis,
            fraction=cfg.select.fraction,
            budget_basis=cfg.select.budget_basis,
            seed=cfg.measure.seed,
            trained_selection=selection,
            drift_autocast=cfg.measure.drift_autocast,
            granularity=granularity,
        )
        baseline_seconds = time.perf_counter() - baseline_started
        logger.info(
            "Drift baseline at t=%d (autocast=%s, primary axis=%s): Jaccard vs trained set = %s "
            "(%.1fs)",
            tracker.timestep,
            cfg.measure.drift_autocast,
            drift_primary_axis,
            {
                name: round(single.baseline_selection.jaccard(selection), 3)
                for name, single in tracker.trackers.items()
            },
            baseline_seconds,
        )
        callbacks.append(
            make_stability_callback(
                adapter,
                samples,
                tracker,
                axes=cfg.data.concept_axes,
                every=cfg.train.stability_every,
                max_steps=cfg.train.max_steps,
                fraction=cfg.select.fraction,
                budget_basis=cfg.select.budget_basis,
                seed=cfg.measure.seed,
                drift_autocast=cfg.measure.drift_autocast,
                sample_provider=sample_provider,
                on_snapshot=lambda s: logger.info(
                    "drift @ step %d [%s]: jaccard=%.3f spearman=%.3f (%.1fs)",
                    s.step, s.axis, s.jaccard_vs_baseline, s.spearman_vs_baseline, s.seconds,
                ),
            )
        )

    # Every arm encodes the same images through the same frozen VAE; on disk, the
    # first arm pays for the latents and the rest read them back.
    latent_store = (
        LatentStore.for_run(cfg.train.latent_cache_dir, cfg.model, dataset)
        if cfg.train.cache_latents and cfg.train.latent_cache_dir
        else None
    )

    state = train_lora(
        adapter,
        dataset,
        selection,
        cfg.train,
        output_dir=output_dir,
        callbacks=callbacks,
        extra_metadata={"axis": axis},
        latent_store=latent_store,
    )

    dump_json(
        {
            "run": cfg.name,
            "arm": args.run_name or selection_path.stem,
            "selection": selection.to_dict(),
            "training_subset": training_subset,
            "final": state.to_dict(),
            # Optimisation vs CA-drift wall clock, so arms with and without drift
            # tracking can be compared on the part they share.
            "timing": {
                "train_seconds": state.train_elapsed,
                "train_seconds_per_step": state.train_elapsed / max(state.step, 1),
                "callback_seconds": state.callback_elapsed,
                "drift_baseline_seconds": baseline_seconds,
                "loop_seconds": state.elapsed,
                "mixed_precision": cfg.train.mixed_precision,
                "allow_tf32": cfg.model.allow_tf32,
                "gradient_checkpointing": cfg.model.enable_gradient_checkpointing,
                "stability_every": cfg.train.stability_every,
                "drift_source": cfg.measure.drift_source,
                "crop": cfg.data.crop,
                "cache_latents": cfg.train.cache_latents,
            },
            "history": state.history,
        },
        output_dir / "train_summary.json",
    )
    curve = plot_loss_curve(state.history, output_dir / "loss_curve.png")
    if curve is not None:
        logger.info("Loss curve: %s", curve)
    if tracker is not None:
        dump_json({**tracker.to_dict(), "x0": drift_x0}, output_dir / "stability.json")
        logger.info("CA drift summary: %s", tracker.summary())
    logger.info("Done: %s", output_dir)


def _restrict_to_training_subset(cfg, dataset) -> dict | None:
    """Apply ``train.subset_size`` and pin the draw to the run directory.

    Arms are compared on the premise that they saw the same images, so the first
    arm records the draw and every later one has to reproduce it.
    """
    if cfg.train.subset_size is None:
        return None
    record = apply_training_subset(dataset, cfg.train.subset_size, seed=cfg.train.subset_seed)
    path = cfg.run_dir / "training_subset.json"
    if path.exists():
        recorded = load_json(path)
        if recorded != record:
            raise RuntimeError(
                f"Training subset differs from the one recorded in {path}: "
                f"{recorded.get('keys')} vs {record['keys']}. Arms of one run must share it."
            )
    else:
        dump_json(record, path)
    logger.info(
        "Training on %d of %d images (subset_seed=%d): %s",
        record["size"], record["source_size"], record["seed"], record["keys"],
    )
    return record


def _baseline_timestep(cfg, selection: Selection, override: int | None) -> int:
    """Which timestep to track drift at: the one the selection was made from."""
    if override is not None:
        return override
    timesteps = selection.meta.get("timesteps")
    if timesteps:
        return int(timesteps[0])
    return cfg.measure.timesteps[0]


if __name__ == "__main__":
    main()
