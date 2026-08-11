"""Train LoRA on a previously computed projection selection.

    python scripts/train_lora.py --config configs/tiny_cpu.yaml \
        --selection selection_multi_t.json --run-name multi_t

Every ablation arm is this same command with a different ``--selection``, so any
difference in the results is attributable to the projection choice alone.

With ``train.stability_every > 0`` the run also produces
``stability.json`` — the CA-drift curve — at no extra training cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calora import cli  # noqa: E402
from calora.analysis.stability import (  # noqa: E402
    StabilityTracker,
    build_stability_tracker,
    make_stability_callback,
)
from calora.ca.measure import prepare_samples  # noqa: E402
from calora.ca.select import Selection  # noqa: E402
from calora.data import build_dataset  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.models.projections import ProjectionTable  # noqa: E402
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

    callbacks = []
    tracker: StabilityTracker | None = None
    if cfg.train.stability_every > 0:
        # A second, centre-cropped and un-mirrored view of the same data. Drift is
        # a comparison of CA measured at step 0 against CA measured at step N, so
        # those two measurements have to see the same pixels even when training is
        # reading random crops and random flips out of `dataset`.
        measure_dataset = build_dataset(
            cfg.data,
            resolution=cfg.model.resolution,
            seed=cfg.measure.seed,
            crop="center",
            hflip=False,
            augs_per_sample=cfg.measure.augs_per_sample,
        )
        samples = prepare_samples(
            adapter,
            measure_dataset,
            axes=cfg.data.concept_axes,
            num_samples=cfg.train.stability_num_samples,
        )
        tracker = build_stability_tracker(
            adapter,
            samples,
            axes=cfg.data.concept_axes,
            timestep=_baseline_timestep(cfg, selection, args.baseline_timestep),
            selection_axis=cfg.select.axis,
            fraction=cfg.select.fraction,
            budget_basis=cfg.select.budget_basis,
            seed=cfg.measure.seed,
            trained_selection=selection,
            drift_autocast=cfg.measure.drift_autocast,
            granularity=granularity,
        )
        logger.info(
            "Drift baseline at t=%d (autocast=%s): %d projections, Jaccard vs trained set = %.3f",
            tracker.timestep,
            cfg.measure.drift_autocast,
            len(tracker.baseline_selection),
            tracker.baseline_selection.jaccard(selection),
        )
        callbacks.append(
            make_stability_callback(
                adapter,
                samples,
                tracker,
                axes=cfg.data.concept_axes,
                every=cfg.train.stability_every,
                max_steps=cfg.train.max_steps,
                selection_axis=cfg.select.axis,
                fraction=cfg.select.fraction,
                budget_basis=cfg.select.budget_basis,
                seed=cfg.measure.seed,
                drift_autocast=cfg.measure.drift_autocast,
                on_snapshot=lambda s: logger.info(
                    "drift @ step %d: jaccard=%.3f spearman=%.3f",
                    s.step, s.jaccard_vs_baseline, s.spearman_vs_baseline,
                ),
            )
        )

    state = train_lora(
        adapter,
        dataset,
        selection,
        cfg.train,
        output_dir=output_dir,
        callbacks=callbacks,
        extra_metadata={"axis": axis},
    )

    dump_json(
        {
            "run": cfg.name,
            "arm": args.run_name or selection_path.stem,
            "selection": selection.to_dict(),
            "final": state.to_dict(),
            "history": state.history,
        },
        output_dir / "train_summary.json",
    )
    curve = plot_loss_curve(state.history, output_dir / "loss_curve.png")
    if curve is not None:
        logger.info("Loss curve: %s", curve)
    if tracker is not None:
        dump_json(tracker.to_dict(), output_dir / "stability.json")
        logger.info("CA drift summary: %s", tracker.summary())
    logger.info("Done: %s", output_dir)


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
