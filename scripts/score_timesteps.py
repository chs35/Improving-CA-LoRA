"""Rank measured timesteps by concept-discriminativeness.

    python scripts/score_timesteps.py --config configs/tiny_cpu.yaml

Reads every ``<run_dir>/measurements/t*.safetensors`` and writes
``<run_dir>/timestep_scores.json``. One run reports, per timestep:

* ``expanded_weighted_similarity`` - CA-weighted mean similarity over the top-n·k
  (primary; lower is better)
* ``conflict_count``               - |top-k CA and sim < tau| with tau pooled across t,
  swept over ``select.conflict_sim_quantiles`` (auxiliary; higher is better)
* ``selected_mean_similarity``     - the unweighted top-k reading (control)
* ``score``                        - S(t) = sum_theta CA_norm * (1 - sim) (diagnostic)

Both similarity sources are computed every time: ``cos_of_means`` is the primary
statistic and ``mean_of_cos`` is the control that shows whether a timestep effect
is direction or the 1/||g_d|| weighting. If ``measure.split_half`` wrote half
files, each half is ranked independently and the agreement between the two
rankings is reported — a timestep order that does not survive halving the samples
is not a result.

The top ``select.top_timesteps`` entries are what the aggregated selection
uses (the trained multi-t run used the top 3). Whether the paper's t=81 lands
near the top is itself a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calora import cli  # noqa: E402
from calora.ca.select import (  # noqa: E402
    RANK_KEYS,
    TimestepScore,
    rank_timesteps,
    sort_timestep_scores,
    timestep_rank_agreement,
)
from calora.ca.stats import load_measurement  # noqa: E402
from calora.utils.io import dump_json  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402

logger = get_logger("score_timesteps")

#: Reported for every similarity source, best-first under each.
RANK_MODES = ("expanded_weighted_similarity", "conflict_count", "score")


def load_all(cfg, half: int | None = None) -> dict:
    """Every full measurement, or every measurement of one split half."""
    directory = cli.measurement_dir(cfg)
    if half is None:
        paths = [
            p for p in sorted(directory.glob("t*.safetensors")) if not cli.is_half_measurement(p)
        ]
    else:
        paths = sorted(directory.glob(f"t*_half{half}.safetensors"))
    if not paths and half is None:
        raise FileNotFoundError(
            f"No measurements in {directory}. Run scripts/measure_ca.py first."
        )
    results = {}
    for path in paths:
        # Rejects a stray measurement from the other granularity: the two are
        # indistinguishable by shape and would pool into a meaningless ranking.
        result = load_measurement(path, expect_granularity=cfg.select.granularity)
        results[result.timestep] = result
    if half is None:
        logger.info(
            "Loaded %d %s-granularity measurements: %s",
            len(results), cfg.select.granularity, sorted(results),
        )
    return results


def score_all(cfg, results: dict, source: str, axis_pair: tuple[str, str]) -> list[TimestepScore]:
    """Score every timestep once under ``source``; the rankings are re-sorts of it."""
    return rank_timesteps(
        results,
        next(iter(results.values())).table(),
        axis_pair=axis_pair,
        similarity_source=source,
        conflict_quantiles=cfg.select.conflict_sim_quantiles,
        primary_quantile=cfg.select.conflict_primary_quantile,
        ca_weighting=cfg.select.tscore_ca_weighting,
        ca_normalization=cfg.select.tscore_ca_normalization,
        fraction=cfg.select.fraction,
        selection_multiplier=cfg.select.tscore_selection_multiplier,
        budget_basis=cfg.select.budget_basis,
        selection_axis=cfg.select.axis,
    )


def main() -> None:
    parser = cli.base_parser(__doc__ or "")
    parser.add_argument(
        "--rank-by",
        default="expanded_weighted_similarity",
        choices=tuple(RANK_KEYS),
        help="Which metric picks the headline top_timesteps",
    )
    args = parser.parse_args()
    cfg = cli.resolve(args)

    results = load_all(cfg)
    axes = cfg.data.concept_axes
    if len(axes) < 2:
        raise ValueError("Timestep scoring needs at least two concept axes")
    axis_pair = (axes[0], axes[1])

    if cfg.select.budget_basis == "unet_params":
        # Scoring runs offline from saved statistics, with no model to count against.
        raise ValueError(
            "select.budget_basis='unet_params' is unavailable here; "
            "use 'attn_params' or 'projection_count' for timestep scoring."
        )

    sources = {
        source: score_all(cfg, results, source, axis_pair)
        for source in ("cos_of_means", "mean_of_cos")
    }
    primary = sources[cfg.select.similarity_source]
    top = [s.timestep for s in sort_timestep_scores(primary, args.rank_by)][
        : cfg.select.top_timesteps
    ]

    halves = [load_all(cfg, half=h) for h in (0, 1)]
    agreement: dict[str, dict[str, dict]] | None = None
    if all(halves):
        agreement = {}
        for source in sources:
            half_scores = [score_all(cfg, half, source, axis_pair) for half in halves]
            agreement[source] = {
                mode: timestep_rank_agreement(
                    half_scores[0], half_scores[1], rank_by=mode, top_n=cfg.select.top_timesteps
                )
                for mode in RANK_MODES
            }

    payload = {
        "run": cfg.name,
        "granularity": cfg.select.granularity,
        "budget_basis": cfg.select.budget_basis,
        "axis_pair": list(axis_pair),
        "primary_similarity_source": cfg.select.similarity_source,
        "ranked_by": args.rank_by,
        "conflict_sim_quantiles": cfg.select.conflict_sim_quantiles,
        "conflict_primary_quantile": cfg.select.conflict_primary_quantile,
        "tscore_selection_multiplier": cfg.select.tscore_selection_multiplier,
        "top_timesteps": top,
        "sources": {
            source: {
                "scores": [s.to_dict() for s in scores],
                "rankings": {
                    mode: [s.timestep for s in sort_timestep_scores(scores, mode)]
                    for mode in RANK_MODES
                },
            }
            for source, scores in sources.items()
        },
        "split_half_agreement": agreement,
    }
    dump_json(payload, cfg.run_dir / "timestep_scores.json")

    _log_scores(primary, cfg.select.similarity_source, args.rank_by)
    _log_conflict_sweep(primary)
    if agreement is not None:
        _log_agreement(agreement)
    else:
        logger.info(
            "No split-half measurements found; set measure.split_half=true and re-measure "
            "to get a stability check on this ranking."
        )
    logger.info("Top %d timesteps: %s", cfg.select.top_timesteps, top)
    logger.info("Wrote %s", cfg.run_dir / "timestep_scores.json")


def _log_scores(scores: list[TimestepScore], source: str, rank_by: str) -> None:
    logger.info("Similarity source %s, ranked by %s (best first)", source, rank_by)
    logger.info(
        "%6s %12s %12s %10s %9s %10s",
        "t", "exp_w_sim", "sel_sim", "S(t)", "conflict", "mean_sim",
    )
    for s in sort_timestep_scores(scores, rank_by):
        logger.info(
            "%6d %12.4f %12.4f %10.4f %9d %10.4f",
            s.timestep,
            s.expanded_weighted_similarity,
            s.selected_mean_similarity,
            s.score,
            s.conflict_count,
            s.mean_similarity,
        )


def _log_conflict_sweep(scores: list[TimestepScore]) -> None:
    """conflict_count per quantile. A ranking that flips across the row is fragile."""
    quantiles = sorted(scores[0].conflict_counts)
    logger.info("conflict_count by pooled similarity quantile (tau in brackets)")
    logger.info(
        "%6s %s",
        "t",
        " ".join(f"{f'q={q:g}[{scores[0].sim_thresholds[q]:+.3f}]':>16}" for q in quantiles),
    )
    for s in sorted(scores, key=lambda x: x.timestep):
        counts = " ".join(f"{s.conflict_counts[q]:>16d}" for q in quantiles)
        logger.info("%6d %s", s.timestep, counts)


def _log_agreement(agreement: dict[str, dict[str, dict]]) -> None:
    logger.info("Split-half rank agreement (even vs odd samples)")
    for source, per_mode in agreement.items():
        for mode, stats in per_mode.items():
            logger.info(
                "  %-13s %-30s spearman=%+.3f  top%d overlap=%.3f  %s vs %s",
                source,
                mode,
                stats["spearman"],
                stats["top_n"],
                stats["top_overlap"],
                stats["top_a"],
                stats["top_b"],
            )


if __name__ == "__main__":
    main()
