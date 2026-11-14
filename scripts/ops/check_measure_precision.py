"""Pre-fixed gate: may the CA selection measurement leave full fp32?

    python scripts/ops/check_measure_precision.py --config configs/ca_selfgen.yaml

Every earlier selection was measured in fp32 with TF32 off, which leaves the A100
tensor cores idle (~76 min per timestep at 256 samples). This script measures
the same self-generated samples at one timestep under each candidate regime and
compares it with the fp32 reference. A candidate is adopted only if it passes
EVERY criterion below; the fastest passing one wins; if none passes, fp32 stays.
The verdict is written to ``<run_dir>/precision_gate.json`` and read by
scripts/ops/run_ca_selfgen.sh, which applies it to all timesteps alike.

Criteria (fixed here before any GPU number exists, not tunable afterwards):

* CA ranking, both axes: Spearman >= 0.99 and top-``select.fraction`` Jaccard
  >= 0.8 against fp32 -- the thresholds of the drift autocast gate.
* Cross-axis direction (added because ews is a CA-weighted cos_of_means):
  per-unit ``cos_of_means`` Pearson >= 0.99 against fp32, and |ews(t) - ews_fp32(t)| <= 0.01 for the expanded weighted similarity
  on the cos_of_means source (0.01 is meant to sit well under the ews gaps the
  timestep ranking is decided by; a pair of timesteps closer than it is reported
  as a tie).
* Logged, not gated: per-axis ``mean_of_cos`` Pearson and the change of the
  per-axis direction concentration.

Gradient checkpointing is a memory decision, not a numeric one: after the
precision verdict the adopted regime is re-run once with checkpointing off, and
off is adopted when it neither runs out of memory nor is slower, with its
numbers checked against the same criteria.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from calora.analysis.stability import spearman  # noqa: E402
from calora.ca.measure import measure_concept_awareness, prepare_samples  # noqa: E402
from calora.ca.precision import measurement_precision  # noqa: E402
from calora.ca.select import axis_scores, rank_timesteps, select_topk  # noqa: E402
from calora.config import load_config, parse_cli_overrides  # noqa: E402
from calora.data import build_measurement_dataset  # noqa: E402
from calora.data.selfgen import manifest_sha256, selfgen_latents  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.models.projections import ProjectionTable  # noqa: E402
from calora.utils.io import dump_json  # noqa: E402
from calora.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("check_measure_precision")

SPEARMAN_THRESHOLD = 0.99
JACCARD_THRESHOLD = 0.8
COS_OF_MEANS_PEARSON_THRESHOLD = 0.99
EWS_ABS_TOLERANCE = 0.01
#: Ordered fastest-expected first; the reference is always measured first.
CANDIDATES = ("bf16", "tf32")
GATE_FILENAME = "precision_gate.json"


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    mask = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float("nan")


def reading(result, table, cfg) -> dict:
    """Everything the criteria need from one measurement, on the CPU in fp64."""
    axes = list(cfg.data.concept_axes)
    pair = f"{axes[0]}|{axes[1]}"
    scores = {axis: axis_scores(result, axis).to(torch.float64).cpu() for axis in axes}
    tops = {
        axis: set(
            select_topk(
                scores[axis], table, fraction=cfg.select.fraction,
                budget_basis=cfg.select.budget_basis,
            ).names
        )
        for axis in axes
    }
    ews = rank_timesteps(
        {result.timestep: result},
        table,
        axis_pair=(axes[0], axes[1]),
        similarity_source="cos_of_means",
        conflict_quantiles=cfg.select.conflict_sim_quantiles,
        primary_quantile=cfg.select.conflict_primary_quantile,
        ca_weighting=cfg.select.tscore_ca_weighting,
        ca_normalization=cfg.select.tscore_ca_normalization,
        fraction=cfg.select.fraction,
        selection_multiplier=cfg.select.tscore_selection_multiplier,
        budget_basis=cfg.select.budget_basis,
        selection_axis=cfg.select.axis,
    )[0].expanded_weighted_similarity
    summary = result.summary()
    return {
        "scores": scores,
        "tops": tops,
        "cos_of_means": result.cos_of_means[pair].to(torch.float64).cpu(),
        "mean_of_cos": {k: v.to(torch.float64).cpu() for k, v in result.mean_of_cos.items()},
        "ews": float(ews),
        "concentration": dict(summary.get("direction_concentration_mean") or {}),
    }


def compare(ref: dict, other: dict) -> dict:
    axes = sorted(ref["scores"])
    per_axis = {}
    for axis in axes:
        union = ref["tops"][axis] | other["tops"][axis]
        per_axis[axis] = {
            "spearman": spearman(ref["scores"][axis], other["scores"][axis]),
            "jaccard": len(ref["tops"][axis] & other["tops"][axis]) / len(union) if union else 1.0,
            "concentration_delta": other["concentration"].get(axis, float("nan"))
            - ref["concentration"].get(axis, float("nan")),
        }
    cos_pearson = pearson(ref["cos_of_means"], other["cos_of_means"])
    ews_delta = other["ews"] - ref["ews"]
    checks = {
        **{f"spearman/{a}": per_axis[a]["spearman"] >= SPEARMAN_THRESHOLD for a in axes},
        **{f"jaccard/{a}": per_axis[a]["jaccard"] >= JACCARD_THRESHOLD for a in axes},
        "cos_of_means_pearson": cos_pearson >= COS_OF_MEANS_PEARSON_THRESHOLD,
        "ews_abs_delta": abs(ews_delta) <= EWS_ABS_TOLERANCE,
    }
    return {
        "per_axis": per_axis,
        "cos_of_means_pearson": cos_pearson,
        "ews": other["ews"],
        "ews_delta": ews_delta,
        "mean_of_cos_pearson": {
            key: pearson(ref["mean_of_cos"][key], other["mean_of_cos"][key])
            for key in ref["mean_of_cos"]
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def set_checkpointing(adapter, enabled: bool) -> None:
    unet = adapter.unet
    name = "enable_gradient_checkpointing" if enabled else "disable_gradient_checkpointing"
    fn = getattr(unet, name, None)
    if callable(fn):
        fn()


def run_one(adapter, table, samples, cfg, *, precision: str, checkpointing: bool,
            timestep: int) -> dict:
    """One full-protocol measurement (directions on, as measure_ca.py runs it)."""
    set_checkpointing(adapter, checkpointing)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    logger.info("t=%d precision=%s checkpointing=%s ...", timestep, precision, checkpointing)
    result = None
    started = time.perf_counter()
    try:
        with measurement_precision(adapter.device, precision):
            result = measure_concept_awareness(
                adapter, table, samples,
                timestep=timestep,
                axes=cfg.data.concept_axes,
                batch_size=cfg.measure.batch_size,
                granularity=cfg.measure.granularity,
                accumulate_direction=cfg.measure.accumulate_direction,
                compute_paired_cosine=cfg.measure.compute_paired_cosine,
                accum_device=(
                    adapter.device if cfg.measure.accum_device == "auto"
                    else cfg.measure.accum_device
                ),
                eps=cfg.measure.eps,
                seed=cfg.measure.seed,
                paired_across_timesteps=cfg.measure.paired_across_timesteps,
                split_half=False,
                log_every=cfg.measure.log_every,
            )
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        out = {
            "precision": precision,
            "gradient_checkpointing": checkpointing,
            "seconds": seconds,
            "seconds_per_sample": seconds / len(samples),
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
            "oom": False,
            "reading": reading(result, table, cfg),
        }
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        logger.warning("OOM under precision=%s checkpointing=%s: %s", precision, checkpointing, exc)
        out = {"precision": precision, "gradient_checkpointing": checkpointing, "oom": True}
    finally:
        del result
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    logger.info("  -> %s", {k: v for k, v in out.items() if k != "reading"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--timestep", type=int, default=81)
    parser.add_argument("--timesteps-to-estimate", type=int, default=None,
                        help="For the ETA (default: len(measure.timesteps))")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_cli_overrides(args.set))
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", log_file=run_dir / "precision_gate.log")
    if cfg.measure.source != "selfgen":
        raise SystemExit("The gate is defined on the self-generated set (measure.source=selfgen)")

    adapter = load_adapter(cfg.model)
    table = ProjectionTable.from_model(adapter.unet, granularity=cfg.select.granularity)
    dataset = build_measurement_dataset(
        cfg, num_items=args.num_samples, crop="center", hflip=False,
        augs_per_sample=cfg.measure.augs_per_sample,
    )
    latents = selfgen_latents(adapter, dataset, cfg.model)
    samples = prepare_samples(
        adapter, dataset, axes=cfg.data.concept_axes, num_samples=args.num_samples,
        latents=latents,
    )
    offload = getattr(adapter, "offload_encoders", None)
    if callable(offload):
        offload()

    t = args.timestep
    reference = run_one(adapter, table, samples, cfg, precision="fp32", checkpointing=True,
                        timestep=t)
    if reference["oom"]:
        raise SystemExit("The fp32 reference itself ran out of memory")
    runs = {"fp32/gc": reference}
    verdicts = {}
    for precision in CANDIDATES:
        run = run_one(adapter, table, samples, cfg, precision=precision, checkpointing=True,
                      timestep=t)
        runs[f"{precision}/gc"] = run
        verdicts[precision] = (
            {"pass": False, "reason": "oom"} if run["oom"]
            else compare(reference["reading"], run["reading"])
        )

    passing = [p for p in CANDIDATES if verdicts[p]["pass"]]
    adopted = min(passing, key=lambda p: runs[f"{p}/gc"]["seconds"]) if passing else "fp32"
    if adopted != "fp32" and runs[f"{adopted}/gc"]["seconds"] >= reference["seconds"]:
        logger.info("%s passes but is not faster than fp32; keeping fp32", adopted)
        adopted = "fp32"

    # Checkpointing off for the adopted regime: memory and speed, same criteria.
    no_gc = run_one(adapter, table, samples, cfg, precision=adopted, checkpointing=False,
                    timestep=t)
    runs[f"{adopted}/no_gc"] = no_gc
    gc_verdict = (
        {"pass": False, "reason": "oom"} if no_gc["oom"]
        else compare(reference["reading"], no_gc["reading"])
    )
    keep_gc = (
        not gc_verdict["pass"]
        or no_gc["seconds"] >= runs[f"{adopted}/gc"]["seconds"]
        # leave >= 8 GiB headroom: the gate's 32 samples do not exercise the
        # allocator fragmentation of a 256-sample, 20-timestep run
        or (no_gc.get("peak_gib") or 0) > 72
    )
    chosen = runs[f"{adopted}/gc"] if keep_gc else no_gc
    timesteps = args.timesteps_to_estimate or len(cfg.measure.timesteps)
    eta_hours = chosen["seconds_per_sample"] * cfg.measure.num_samples * timesteps / 3600

    payload = {
        "config": args.config,
        "timestep": t,
        "num_samples": args.num_samples,
        "selfgen_manifest_sha256": manifest_sha256(dataset.directory),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "thresholds": {
            "spearman": SPEARMAN_THRESHOLD,
            "jaccard": JACCARD_THRESHOLD,
            "cos_of_means_pearson": COS_OF_MEANS_PEARSON_THRESHOLD,
            "ews_abs_delta": EWS_ABS_TOLERANCE,
        },
        "reference": {k: v for k, v in reference.items() if k != "reading"}
        | {"ews": reference["reading"]["ews"]},
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "reading"} for k, v in runs.items()},
        "verdicts": verdicts,
        "no_gc_verdict": gc_verdict,
        "adopted": {"precision": adopted, "gradient_checkpointing": keep_gc},
        "eta": {
            "seconds_per_sample": chosen["seconds_per_sample"],
            "num_samples": cfg.measure.num_samples,
            "timesteps": timesteps,
            "hours": eta_hours,
        },
    }
    dump_json(payload, run_dir / GATE_FILENAME)
    logger.info("ADOPTED precision=%s gradient_checkpointing=%s; ETA %.1f h for %d timesteps",
                adopted, keep_gc, eta_hours, timesteps)


if __name__ == "__main__":
    main()
