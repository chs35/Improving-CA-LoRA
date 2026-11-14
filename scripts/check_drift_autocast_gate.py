"""fp32-vs-bf16 equivalence gate for the CA *drift* measurement.

    python scripts/check_drift_autocast_gate.py --config configs/sdxl_cityscapes.yaml \
        --run-name t81 --lora runs/sdxl_cityscapes/t81/lora-final.safetensors

``--lora`` is optional. Without it the gate runs on the *base* UNet weights — no
checkpoint is loaded and no merge round-trip happens — which is how this study
ran it once up front, before any arm existed. The output then records
``"lora": null`` and lands in ``<run_dir>/base/`` unless ``--run-name`` says
otherwise.

The drift snapshot is ~50% of a training arm's wall clock, and running its
forward/backward under bf16 autocast roughly halves it. That is only sound if the
*measured* CA is unchanged for the purpose it is put to — ranking projections —
so a gate was fixed in advance: measure CA twice at the run's drift timestep
with the identical samples and seed, once under
``drift_autocast="no"`` and once under ``"bf16"``, and require

* Spearman rho over **all** projections >= :data:`SPEARMAN_THRESHOLD`, and
* Jaccard of the two top-``select.fraction`` selections >= :data:`JACCARD_THRESHOLD`.

Both must hold; ``measure.drift_autocast: "bf16"`` may then be enabled for the
training runs (and only if applied to every arm, so drift stays comparable).

This script is a read-only consumer of the training path's primitives: the same
``prepare_samples`` over the same centre-cropped dataset, the same
``measure_concept_awareness`` call as
:func:`calora.analysis.stability.make_stability_callback`, inside the same
:func:`calora.analysis.stability.merged_lora` context when a checkpoint is given.
Nothing is trained and no config is modified; the only output is
``drift_autocast_gate.json`` in the arm's directory.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from calora import cli  # noqa: E402
from calora.analysis.stability import (  # noqa: E402
    DRIFT_AUTOCAST_MODES,
    drift_autocast_context,
    jaccard,
    merged_lora,
    spearman,
)
from calora.ca.measure import measure_concept_awareness  # noqa: E402
from calora.ca.select import Selection, axis_scores, select_topk  # noqa: E402
from calora.analysis.drift_data import drift_samples  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.models.projections import ProjectionTable  # noqa: E402
from calora.train.lora import load_lora  # noqa: E402
from calora.utils.device import set_matmul_tf32  # noqa: E402
from calora.utils.io import dump_json, load_json  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402

logger = get_logger("check_drift_autocast_gate")

#: The gate's thresholds, fixed in advance. Both are required; neither is tunable
#: after the fact, which is the whole point of writing them down here.
SPEARMAN_THRESHOLD = 0.99
JACCARD_THRESHOLD = 0.8

#: How many projections of each mode's ranking land in the JSON. The full [P]
#: score vector is not serialised — the gate is a statement about the ranking,
#: and the two summary statistics already carry it.
TOP_N_REPORTED = 20

GATE_FILENAME = "drift_autocast_gate.json"


def resolve_timestep(arm_dir: Path, cfg, override: int | None) -> tuple[int, str]:
    """The timestep the arm's drift was tracked at, and where that came from.

    Read off the arm's own artefacts rather than recomputed, because the gate has
    to bound the error at the ``t`` the snapshots actually used:
    ``stability.json`` records it directly, and ``train_summary.json``'s selection
    carries the ``t`` that ``train_lora.py`` would have derived it from. The
    config default is the last resort, and is only right for an arm that took it.
    """
    if override is not None:
        return int(override), "cli"

    stability = arm_dir / "stability.json"
    if stability.exists():
        data = load_json(stability)
        if data.get("timestep") is not None:
            return int(data["timestep"]), str(stability)

    summary = arm_dir / "train_summary.json"
    if summary.exists():
        timesteps = load_json(summary).get("selection", {}).get("meta", {}).get("timesteps")
        if timesteps:
            return int(timesteps[0]), str(summary)

    return int(cfg.measure.timesteps[0]), "config:measure.timesteps[0]"


def measure_mode(
    adapter,
    table: ProjectionTable,
    samples,
    mode: str,
    *,
    timestep: int,
    axes,
    selection_axis: str,
    fraction: float,
    budget_basis: str,
    seed: int,
    log_every: int = 0,
) -> tuple[torch.Tensor, Selection, dict]:
    """One drift measurement under ``mode``, plus the ranking read off it.

    The ``measure_concept_awareness`` keywords are exactly the drift callback's
    (``accumulate_direction=False``, ``compute_paired_cosine=False``, batch_size
    and accumulator defaults), so the two modes differ in nothing but the
    autocast context. ``log_every`` only prints.

    Returns the ``[P]`` fp64 CPU score vector, its top-k selection, and a JSON
    summary. The ``MeasurementResult`` is dropped before returning: on CUDA it
    holds the per-projection accumulators, and both modes' copies must never be
    alive at once (the tsweep OOM lesson).
    """
    result = None  # bound before the measurement so the del below always has a name
    logger.info("Measuring CA at t=%d under drift_autocast=%r", timestep, mode)
    started = time.perf_counter()
    with drift_autocast_context(adapter.device, mode):
        result = measure_concept_awareness(
            adapter,
            table,
            samples,
            timestep=timestep,
            axes=axes,
            accumulate_direction=False,
            compute_paired_cosine=False,
            seed=seed,
            log_every=log_every,
        )
    elapsed = time.perf_counter() - started

    scores = axis_scores(result, selection_axis).detach().to(torch.float64).cpu()
    measurement = result.summary()
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    selection = select_topk(
        scores,
        table,
        fraction=fraction,
        budget_basis=budget_basis,
        strategy=f"drift-gate:{mode}@t{timestep}",
    )
    order = torch.argsort(scores, descending=True).tolist()[:TOP_N_REPORTED]
    summary = {
        "drift_autocast": mode,
        "seconds": elapsed,
        "num_selected": len(selection),
        "params_selected": selection.params_selected,
        "fraction_achieved": selection.fraction_achieved,
        # Per-axis CA summaries (mean/std/min/max/sem per axis) as the measurement
        # itself reports them, so the two modes are comparable line by line.
        "measurement": measurement,
        "top_projections": [
            {"rank": rank, "name": table[index].name, "score": float(scores[index])}
            for rank, index in enumerate(order)
        ],
    }
    logger.info(
        "  drift_autocast=%r: %.1fs, mean CA(%s)=%.6g, %d projections selected",
        mode,
        elapsed,
        selection_axis,
        float(scores.mean()),
        len(selection),
    )
    return scores, selection, summary


def run_gate(
    adapter,
    table: ProjectionTable,
    samples,
    *,
    timestep: int,
    axes,
    selection_axis: str,
    fraction: float,
    budget_basis: str,
    seed: int,
    log_every: int = 0,
) -> dict:
    """Measure both autocast modes on one fixed set of weights and compare them.

    On an arm, both measurements run inside a single :func:`merged_lora` context
    opened by the caller: CA is a statement about the effective weight ``W0 + BA``,
    and one context guarantees the two modes saw bit-identical weights rather than
    two merge/unmerge round-trips of them — so ``table`` must then be the *merged*
    projection table. On the base model there is nothing to merge and the caller
    passes the base table; either way the two modes differ only in autocast.
    """
    scores: dict[str, torch.Tensor] = {}
    selections: dict[str, Selection] = {}
    modes: dict[str, dict] = {}
    for mode in DRIFT_AUTOCAST_MODES:
        scores[mode], selections[mode], modes[mode] = measure_mode(
            adapter,
            table,
            samples,
            mode,
            timestep=timestep,
            axes=axes,
            selection_axis=selection_axis,
            fraction=fraction,
            budget_basis=budget_basis,
            seed=seed,
            log_every=log_every,
        )

    rho = spearman(scores["no"], scores["bf16"])
    overlap = jaccard(selections["no"].names, selections["bf16"].names)
    passed = bool(rho >= SPEARMAN_THRESHOLD and overlap >= JACCARD_THRESHOLD)
    return {
        "timestep": timestep,
        "num_samples": len(samples),
        "num_contributions": modes["no"]["measurement"]["num_samples"],
        "seed": seed,
        "axes": list(axes),
        "selection_axis": selection_axis,
        "fraction": fraction,
        "budget_basis": budget_basis,
        "granularity": table.granularity,
        "num_projections": len(table),
        "modes": modes,
        "spearman": rho,
        "jaccard": overlap,
        "thresholds": {"spearman": SPEARMAN_THRESHOLD, "jaccard": JACCARD_THRESHOLD},
        "pass": passed,
        "speedup": _speedup(modes),
    }


def _speedup(modes: dict[str, dict]) -> float | None:
    """fp32 seconds / bf16 seconds — the reason the gate exists, so it is recorded
    next to the verdict. ``None`` if the bf16 pass somehow took no measurable time."""
    bf16 = modes["bf16"]["seconds"]
    return modes["no"]["seconds"] / bf16 if bf16 > 0 else None


def main() -> None:
    parser = cli.base_parser(__doc__ or "")
    parser.add_argument(
        "--lora",
        default=None,
        help="lora-final.safetensors of the arm to gate on (default: gate the base model)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Arm subdirectory under the run dir "
        "(default: the --lora file's parent name, or 'base' without --lora)",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help="Drift timestep (default: the arm's stability.json / selection timestep)",
    )
    args = parser.parse_args()
    cfg = cli.resolve(args)

    lora_path = Path(args.lora) if args.lora is not None else None
    if lora_path is not None and not lora_path.exists():
        raise SystemExit(f"No LoRA checkpoint at {lora_path}")
    # Without a checkpoint the gate is about the base weights, and 'base' is where
    # it lands: an arm name would claim a verdict about an arm that does not exist.
    arm = args.run_name or (lora_path.parent.name if lora_path is not None else "base")
    arm_dir = cfg.run_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    timestep, timestep_source = resolve_timestep(arm_dir, cfg, args.timestep)
    logger.info(
        "Gating arm '%s' at t=%d (from %s) on %s",
        arm,
        timestep,
        timestep_source,
        lora_path if lora_path is not None else "the base model (no --lora)",
    )

    adapter = load_adapter(cfg.model)
    # The gate's reference is fixed as an fp32 measurement; with
    # model.allow_tf32 on (a training setting) the "no" side would silently be TF32.
    set_matmul_tf32(False)
    # Pre-injection, so the identity check below compares like with like. The gate
    # bounds the error of the drift measurement, so it has to run at the same unit
    # granularity the drift callback will.
    table = ProjectionTable.from_model(adapter.unet, granularity=cfg.select.granularity)

    # The drift callback's baseline samples, built by the very function
    # scripts/train_lora.py uses (calora.analysis.drift_data.drift_samples):
    # measure.drift_source decides the x0 (the self-generated base set, or real
    # images with crop="center" and hflip=False), measure.seed, and
    # train.stability_num_samples of them. The gate bounds the precision error of
    # one measurement, so it is run on the base set even for selfgen_current.
    samples, _provider, _x0 = drift_samples(cfg, adapter)
    logger.info(
        "Prepared %d drift samples (%d aug(s) each)", len(samples), cfg.measure.augs_per_sample
    )

    # Nothing below needs the VAE or the text encoders; on CUDA that is several GB
    # of the budget the two measurements draw from.
    offload = getattr(adapter, "offload_encoders", None)
    if callable(offload):
        offload()

    gate_kwargs = dict(
        timestep=timestep,
        axes=cfg.data.concept_axes,
        selection_axis=cfg.select.axis,
        fraction=cfg.select.fraction,
        budget_basis=cfg.select.budget_basis,
        seed=cfg.measure.seed,
        log_every=cfg.measure.log_every,
    )
    if lora_path is None:
        # Nothing injected, so nothing to merge and no identity to re-check: the
        # base ``table`` already names the weights both modes will measure.
        payload = run_gate(adapter, table, samples, **gate_kwargs)
    else:
        # Same granularity as the table above, checked rather than assumed: the gate
        # is a statement about the drift measurement of *this* arm.
        load_lora(
            adapter.unet, lora_path, cfg.train.lora, expect_granularity=cfg.select.granularity
        )

        with merged_lora(adapter.unet):
            merged_table = ProjectionTable.from_model(
                adapter.unet, granularity=cfg.select.granularity
            )
            if merged_table.names != table.names:
                raise RuntimeError(
                    "Projection identities changed after LoRA injection; the two modes "
                    "would be compared over different layers."
                )
            payload = run_gate(adapter, merged_table, samples, **gate_kwargs)

    payload = {
        "run": cfg.name,
        "arm": arm,
        "lora": str(lora_path) if lora_path is not None else None,
        "timestep_source": timestep_source,
        **payload,
    }
    path = arm_dir / GATE_FILENAME
    dump_json(payload, path)

    logger.info(
        "drift_autocast gate %s: spearman=%.6f (>= %.2f) jaccard=%.4f (>= %.2f)  "
        "[t=%d, %d samples, seed %d, %s]",
        "PASS" if payload["pass"] else "FAIL",
        payload["spearman"],
        SPEARMAN_THRESHOLD,
        payload["jaccard"],
        JACCARD_THRESHOLD,
        timestep,
        payload["num_samples"],
        payload["seed"],
        "bf16 may be enabled for arms C onward"
        if payload["pass"]
        else "keep measure.drift_autocast='no'",
    )
    if payload["speedup"] is not None:
        logger.info(
            "Snapshot cost: %.1fs fp32 vs %.1fs bf16 (%.2fx)",
            payload["modes"]["no"]["seconds"],
            payload["modes"]["bf16"]["seconds"],
            payload["speedup"],
        )
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
