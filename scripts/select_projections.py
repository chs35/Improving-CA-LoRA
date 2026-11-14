"""Turn CA measurements into a concrete set of LoRA target modules.

    # paper reproduction: single timestep
    python scripts/select_projections.py --config configs/tiny_cpu.yaml \
        --strategy single --timesteps 81 --out selection_t81.json

    # multi-t: aggregate the top-ranked timesteps from score_timesteps.py
    # (select.top_timesteps defaults to 5; the trained multi-t run used 3)
    python scripts/select_projections.py --config configs/tiny_cpu.yaml \
        --strategy aggregate --from-scores --set select.top_timesteps=3 \
        --out selection_multi_t.json

    # random-selection control
    python scripts/select_projections.py --config configs/tiny_cpu.yaml \
        --strategy random --out selection_random.json

Every arm of the ablation matrix is one invocation of this script; the training
script then differs only in which selection JSON it is given.

The unit is ``select.granularity``: attention heads by default (names like
``...attn1.to_q@h3``), or whole fused projections under ``module``. The
measurements loaded here must have been produced at the same granularity, which
is checked rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calora import cli  # noqa: E402
from calora.ca.select import (  # noqa: E402
    aggregate_across_timesteps,
    axis_scores,
    select_all,
    select_random,
    select_topk,
)
from calora.utils.io import dump_json, load_json  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402

logger = get_logger("select_projections")


def load_all(cfg) -> dict:
    directory = cli.measurement_dir(cfg)
    # Split-half files sit next to the full ones and carry the same timestep;
    # including them here would silently overwrite the full measurement with
    # whichever half sorted last.
    paths = [p for p in sorted(directory.glob("t*.safetensors")) if not cli.is_half_measurement(p)]
    if not paths:
        raise FileNotFoundError(f"No measurements in {directory}. Run measure_ca.py first.")
    from calora.ca.stats import common_provenance, load_measurement

    results = {}
    for path in paths:
        # Head- and module-unit measurements have the same filename and shape and
        # differ only in what an entry means; loading is where that gets caught.
        result = load_measurement(path, expect_granularity=cfg.select.granularity)
        results[result.timestep] = result
    # Same rule as granularity, for the x0 source and precision: refuse to pool.
    common_provenance(results)
    return results


def main() -> None:
    parser = cli.base_parser(__doc__ or "")
    parser.add_argument(
        "--strategy",
        default="aggregate",
        choices=("single", "aggregate", "random", "all"),
        help="single: one timestep; aggregate: several; random/all: baselines",
    )
    parser.add_argument(
        "--timesteps", type=int, nargs="*", default=None, help="Timesteps to use"
    )
    parser.add_argument(
        "--from-scores",
        action="store_true",
        help="Take timesteps from <run_dir>/timestep_scores.json (top_timesteps of them)",
    )
    parser.add_argument("--fraction", type=float, default=None, help="Override select.fraction")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --strategy random")
    parser.add_argument("--out", default="selection.json", help="Filename inside the run dir")
    args = parser.parse_args()

    cfg = cli.resolve(args)
    fraction = args.fraction if args.fraction is not None else cfg.select.fraction
    results = load_all(cfg)
    table = next(iter(results.values())).table()

    if args.strategy == "all":
        selection = select_all(table)
    elif args.strategy == "random":
        selection = select_random(
            table, fraction=fraction, budget_basis=cfg.select.budget_basis, seed=args.seed
        )
    else:
        timesteps = _resolve_timesteps(args, cfg, results)
        method = "single" if args.strategy == "single" else cfg.select.aggregation
        if method == "single" and len(timesteps) != 1:
            raise ValueError(f"--strategy single needs exactly one timestep, got {timesteps}")
        scores = (
            axis_scores(results[timesteps[0]], cfg.select.axis)
            if method == "single"
            else aggregate_across_timesteps(
                results,
                axis=cfg.select.axis,
                method=method,
                normalization=cfg.select.normalization,
                timesteps=timesteps,
            )
        )
        selection = select_topk(
            scores,
            table,
            fraction=fraction,
            budget_basis=cfg.select.budget_basis,
            strategy=f"{args.strategy}:{method}",
            meta={
                "timesteps": timesteps,
                "axis": cfg.select.axis,
                "normalization": cfg.select.normalization,
                **_provenance_meta({t: results[t] for t in timesteps}),
            },
        )

    path = cfg.run_dir / args.out
    dump_json(selection.to_dict(), path)
    logger.info(
        "%s [%s]: %d units over %d matrices (%.3f%% of %d units), %s params "
        "(%.3f%% of attention), breakdown=%s",
        selection.strategy,
        selection.granularity,
        len(selection),
        len(selection.module_names()),
        100 * selection.unit_fraction_achieved,
        selection.units_total,
        f"{selection.params_selected:,}",
        100 * selection.fraction_achieved,
        selection.breakdown["kind"],
    )
    logger.info("Wrote %s", path)


def _provenance_meta(results) -> dict:
    """What the scores were measured on, carried into the selection JSON.

    Downstream consumers (CA-LoRA-TSM, Hierarchical-CFG) refuse a selection whose
    ``data_source`` is not ``selfgen``, and read ``selfgen_manifest_sha256`` to
    tell which generated set it came from.
    """
    from calora.ca.stats import common_provenance

    provenance = common_provenance(results)
    return {
        "data_source": provenance.get("data_source"),
        "selfgen_manifest_sha256": provenance.get("selfgen_manifest_sha256"),
        "precision": provenance.get("precision"),
        "provenance": provenance,
    }


def _resolve_timesteps(args, cfg, results) -> list[int]:
    if args.from_scores:
        scores_path = cfg.run_dir / "timestep_scores.json"
        if not scores_path.exists():
            raise FileNotFoundError(
                f"{scores_path} not found. Run scripts/score_timesteps.py first."
            )
        payload = load_json(scores_path)
        timesteps = payload["top_timesteps"][: cfg.select.top_timesteps]
    elif args.timesteps:
        timesteps = list(args.timesteps)
    elif cfg.select.single_timestep is not None:
        timesteps = [cfg.select.single_timestep]
    else:
        timesteps = sorted(results)

    missing = [t for t in timesteps if t not in results]
    if missing:
        raise KeyError(f"No measurement for timestep(s) {missing}; measured: {sorted(results)}")
    return timesteps


if __name__ == "__main__":
    main()
