"""Regenerate every figure in the report from the committed evidence.

    python scripts/make_figures.py \
        --measurements-dir runs/tsweep_sdxl_head/measurements \
        --generated-root runs/sdxl_cityscapes

Four figures, one script:

``heatmap``      ``figures/ca_heatmap_qkvout_t_panelnorm.png``
    Per-head CA over the UNet's 140 attention modules, Q/K/V/OUT bands, style
    (red) over viewpoint (blue), at t = 81 / 41 / 1. Needs ``--measurements-dir``.
``tradeoff``     ``figures/tradeoff_scatter.png``
    CMMD against weather CLIP-Score, one point per arm. Needs only ``results/``.
``drift``        ``figures/drift_curves.png``
    Jaccard and Spearman against step 0 for all ten runs. Needs only ``results/``.
``qualitative``  ``figures/qualitative_weather_grid.png``
    The same two snowy prompts and seeds through five models. Needs
    ``--generated-root``.

Two of the four read artefacts that are far too large to commit — 4.9 MB of
measurement safetensors per timestep, and ~10k generated PNGs — so they are
opt-in behind a path flag and skipped with a pointer when it is absent. The
other two read only ``results/``, which is in the repository, so a checkout can
always rebuild them.

This takes no ``--config``: it renders a finished results tree, and the run
configuration that produced it has no say in how the figures look.

The originals were drawn by scratch scripts that were never committed (see
PROJECT_LOG.md); this file is their permanent replacement, reverse-engineered
from the PNG/PDF output so that the regenerated figures carry the same data and
essentially the same layout.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: no display on the pod, none in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from calora.utils.io import load_json  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402

logger = get_logger("make_figures")

FIGURES = ("heatmap", "tradeoff", "drift", "qualitative")

#: Where the measurement safetensors and the generated PNGs live when they are
#: not on the machine drawing the figures.
ARTIFACTS_HINT = "too large to commit; fetch the run artifacts and point at them"

# -- shared styling -----------------------------------------------------------------------
#
# The reference categorical palette and neutral ink of the project's plotting
# convention (see calora.utils.plotting, which uses slots 1 and 2 of the same
# set). Named here rather than inlined because three of the four figures share
# them and the fourth shares the ink.

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e5e4de"
SPINE = "#c3c2b7"

STYLE_COLOR = "#2a78d6"  # slot 1, blue: style-CA adapters
VIEWPOINT_COLOR = "#eb6834"  # slot 2, orange: viewpoint-CA adapters
CONTROL_COLOR = "#1baf7a"  # slot 3, aqua: the no-CA controls

#: Style-axis panels are red, viewpoint-axis panels blue, matching the two
#: adapter families everywhere else in the report.
AXIS_COLORMAPS = {"style": "Reds", "viewpoint": "Blues"}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "grid.color": GRID,
            "font.size": 10,
        }
    )


def _save(figure, path: Path, *, facecolor: str = SURFACE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=facecolor)
    plt.close(figure)
    logger.info("Wrote %s (%.1f MB)", path, path.stat().st_size / 2**20)
    return path


# ==========================================================================================
# Figure 1 - per-head CA heatmap
# ==========================================================================================

#: Timesteps of the low-t cluster, left to right, as in the report's Figure 1.
HEATMAP_TIMESTEPS = (81, 41, 1)

#: Sub-bands within a panel, top to bottom, and the projection kind each reads.
HEATMAP_BANDS = (("Q", "q"), ("K", "k"), ("V", "v"), ("OUT", "out"))

#: Rows per band. SDXL's widest attention modules have 20 heads; the 640-channel
#: blocks have 10, and their rows are repeated x2 so every column fills the band
#: (the first version NaN-padded them, which left the shallow and deep columns
#: half white — the paper's Figures 10/11 are solid).
HEATMAP_BAND_ROWS = 20

#: Blank rows drawn between bands. They are part of the same image, not a
#: subplot gap, so one imshow carries a whole panel.
HEATMAP_BAND_GAP = 2

#: Per-panel colour limits: the panel's own minimum, and its 99th percentile
#: rather than its maximum. A handful of heads sit 3x above the rest of the
#: distribution and stretching the ramp to them flattens every panel to pale.
HEATMAP_VMAX_PERCENTILE = 99.0


def head_matrix(result, axis: str) -> np.ndarray:
    """``[bands*rows, modules]`` of per-head CA for one axis, ready for imshow.

    Columns are attention modules in :class:`~calora.models.projections.ProjectionTable`
    order, which is ``named_modules`` order and therefore runs shallow to deep
    (down blocks, up blocks, mid block). The table is rebuilt from the
    measurement's own records, so the ordering here cannot drift from the
    ordering the measurement was written in.
    """
    table = result.table()
    if table.granularity != "head":
        raise ValueError(
            f"The heatmap is per attention head, but t={result.timestep} was measured at "
            f"{table.granularity} granularity."
        )
    scores = result.ca(axis).to("cpu").numpy()

    columns: OrderedDict[str, dict[str, list[tuple[int, int]]]] = OrderedDict()
    for index, spec in enumerate(table.specs):
        attention = spec.module_name.rsplit(".to_", 1)[0]
        columns.setdefault(attention, {}).setdefault(spec.kind, []).append(
            (spec.head_index, index)
        )

    height = len(HEATMAP_BANDS) * HEATMAP_BAND_ROWS + (len(HEATMAP_BANDS) - 1) * HEATMAP_BAND_GAP
    matrix = np.full((height, len(columns)), np.nan)
    for column, (attention, kinds) in enumerate(columns.items()):
        for band, (_label, kind) in enumerate(HEATMAP_BANDS):
            entries = sorted(kinds[kind])
            values = np.array([scores[index] for _head, index in entries])
            if HEATMAP_BAND_ROWS % len(values):
                raise ValueError(
                    f"{attention} has {len(values)} heads, which does not divide the "
                    f"{HEATMAP_BAND_ROWS}-row band"
                )
            top = band * (HEATMAP_BAND_ROWS + HEATMAP_BAND_GAP)
            matrix[top : top + HEATMAP_BAND_ROWS, column] = np.repeat(
                values, HEATMAP_BAND_ROWS // len(values)
            )
    return matrix


def plot_heatmap(measurements: Path, path: Path) -> Path:
    """Style over viewpoint, three timesteps across, per-panel normalisation."""
    from calora.ca.stats import load_measurement

    results = {}
    for timestep in HEATMAP_TIMESTEPS:
        source = measurements / f"t{timestep:04d}.safetensors"
        if not source.exists():
            raise FileNotFoundError(f"{source} ({ARTIFACTS_HINT})")
        # Full measurements only: the _half files are the split-half control and
        # carry half the samples.
        results[timestep] = load_measurement(source, expect_granularity="head")

    figure, axes = plt.subplots(
        2,
        len(HEATMAP_TIMESTEPS),
        figsize=(12.0, 6.7),
        gridspec_kw={"wspace": 0.054, "hspace": 0.061},
    )
    for row, axis in enumerate(AXIS_COLORMAPS):
        colormap = plt.get_cmap(AXIS_COLORMAPS[axis]).copy()
        colormap.set_bad(SURFACE)  # the inter-band gaps
        for column, timestep in enumerate(HEATMAP_TIMESTEPS):
            panel = axes[row][column]
            matrix = head_matrix(results[timestep], axis)
            finite = matrix[~np.isnan(matrix)]
            panel.imshow(
                matrix,
                cmap=colormap,
                vmin=float(finite.min()),
                vmax=float(np.percentile(finite, HEATMAP_VMAX_PERCENTILE)),
                aspect="auto",
                interpolation="nearest",
            )
            panel.set_xticks([])
            panel.set_yticks([])
            for side in panel.spines.values():
                side.set_color(SPINE)
                side.set_linewidth(0.8)

            if row == 0:
                panel.set_title(f"Timestep = {timestep}", fontsize=11, color=INK)
            if row == len(AXIS_COLORMAPS) - 1:
                panel.set_xlabel("Attention Layers", fontsize=9, color=INK_SECONDARY)
                panel.text(
                    0.0, -0.045, "Shallower", transform=panel.transAxes,
                    fontsize=7.5, color=INK_MUTED, ha="left", va="top",
                )
                panel.text(
                    1.0, -0.045, "Deeper", transform=panel.transAxes,
                    fontsize=7.5, color=INK_MUTED, ha="right", va="top",
                )
            if column == 0:
                for band, (label, _kind) in enumerate(HEATMAP_BANDS):
                    centre = band * (HEATMAP_BAND_ROWS + HEATMAP_BAND_GAP) + HEATMAP_BAND_ROWS / 2
                    panel.text(
                        -0.02, centre, label, transform=panel.get_yaxis_transform(),
                        fontsize=9, ha="right", va="center",
                    )
                panel.text(
                    -0.32, 0.5, f"{axis.capitalize()}\nAwareness",
                    transform=panel.transAxes, fontsize=11, ha="center", va="center",
                )

    figure.suptitle(
        "per-panel normalization (paper-style display)",
        fontsize=10, color=INK_SECONDARY, y=0.965,
    )
    return _save(figure, path)


# ==========================================================================================
# Figure 2 - domain fit vs. weather controllability
# ==========================================================================================

#: ``arm -> (family, label)``. The 0% control is deliberately absent: at CMMD
#: 4.08 it is five times the worst adapter and would compress the whole frontier
#: into the left edge, so it is annotated off-scale instead.
TRADEOFF_ARMS: dict[str, tuple[str, str]] = {
    "A_t81_style": ("style", "t=81"),
    "B_top3_style": ("style", "multi-t"),
    "C_t41_style": ("style", "t=41"),
    "D_t1_style": ("style", "t=1"),
    "A_t81_viewpoint": ("viewpoint", "t=81"),
    "B_top3_viewpoint": ("viewpoint", "multi-t"),
    "C_t41_viewpoint": ("viewpoint", "t=41"),
    "D_t1_viewpoint": ("viewpoint", "t=1"),
    "E_random": ("control", "random 2%"),
    "E_all": ("control", "full attention"),
}

TRADEOFF_BASELINE = "control_base"

TRADEOFF_FAMILIES = {
    "style": (STYLE_COLOR, "o", "Style adapters"),
    "viewpoint": (VIEWPOINT_COLOR, "o", "Viewpoint adapters"),
    "control": (CONTROL_COLOR, "s", "Controls (no CA)"),
}

#: Marker for the random control, so the two controls stay distinguishable while
#: sharing one legend entry (they are one category: no concept signal).
TRADEOFF_RANDOM_MARKER = "D"

#: Label placement, in points from the marker. Hand-set: the points are close
#: enough that an automatic placement collides, and the report's caption refers
#: to specific labels.
TRADEOFF_LABEL_OFFSETS: dict[str, tuple[float, float, str, str]] = {
    "B_top3_style": (8.0, 6.0, "left", "center"),
    "D_t1_viewpoint": (8.0, -7.5, "left", "center"),
    "E_all": (-14.0, -13.5, "left", "top"),
}
TRADEOFF_DEFAULT_OFFSET = (8.0, 0.0, "left", "center")


def _arm_metrics(results: Path, arm: str) -> tuple[float, float]:
    payload = load_json(results / arm / "eval.json")
    return float(payload["cmmd"]["cmmd"]), float(payload["clip_score"]["weather_mean"])


def plot_tradeoff(results: Path, path: Path) -> Path:
    """One point per trained adapter; no arm wins both metrics."""
    points = {arm: _arm_metrics(results, arm) for arm in TRADEOFF_ARMS}
    figure, panel = plt.subplots(figsize=(7.8, 5.2))

    for family, (color, marker, label) in TRADEOFF_FAMILIES.items():
        arms = [arm for arm, (kind, _) in TRADEOFF_ARMS.items() if kind == family]
        panel.scatter([], [], s=70, color=color, marker=marker, label=label)
        for arm in arms:
            cmmd, weather = points[arm]
            panel.scatter(
                cmmd, weather, s=70, color=color, zorder=3,
                marker=TRADEOFF_RANDOM_MARKER if arm == "E_random" else marker,
            )
            dx, dy, ha, va = TRADEOFF_LABEL_OFFSETS.get(arm, TRADEOFF_DEFAULT_OFFSET)
            panel.annotate(
                TRADEOFF_ARMS[arm][1],
                (cmmd, weather),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=9,
                color=INK_SECONDARY,
                ha=ha,
                va=va,
            )

    baseline_cmmd, baseline_weather = _arm_metrics(results, TRADEOFF_BASELINE)
    panel.annotate(
        f"0% control off scale: CMMD {baseline_cmmd:.2f}, weather {baseline_weather:.1f} →",
        (0.99, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color=INK_MUTED,
    )

    panel.set_xlabel("CMMD  (lower = better domain fit)", fontsize=11)
    panel.set_ylabel("Weather CLIP-Score  (higher = better prompt control)", fontsize=11)
    panel.set_title(
        "Domain fit vs. weather controllability (one point per trained adapter)",
        fontsize=12, color=INK,
    )
    panel.margins(x=0.11, y=0.16)
    panel.grid(color=GRID, linewidth=0.8)
    panel.set_axisbelow(True)
    for side in ("top", "right"):
        panel.spines[side].set_visible(False)
    legend = panel.legend(loc="upper left", frameon=False, fontsize=11, labelspacing=0.6)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return _save(figure, path)


# ==========================================================================================
# Figure 3 - CA drift during training
# ==========================================================================================

#: ``arm -> family``. The four style arms and four viewpoint arms are drawn as
#: families (one legend entry each); the two controls get their own.
DRIFT_FAMILIES: dict[str, str] = {
    "A_t81_style": "style",
    "B_top3_style": "style",
    "C_t41_style": "style",
    "D_t1_style": "style",
    "A_t81_viewpoint": "viewpoint",
    "B_top3_viewpoint": "viewpoint",
    "C_t41_viewpoint": "viewpoint",
    "D_t1_viewpoint": "viewpoint",
    "E_all": "full",
    "E_random": "random",
}

#: colour, linestyle, alpha, linewidth, legend label, end-of-curve direct label.
DRIFT_STYLES: dict[str, tuple] = {
    "style": (STYLE_COLOR, "-", 0.75, 1.6, "Style adapters (4 runs)", None),
    "viewpoint": (VIEWPOINT_COLOR, "-", 0.75, 1.6, "Viewpoint adapters (4 runs)", None),
    "full": (CONTROL_COLOR, "-", 1.0, 2.0, "Full attention (control)", "full attn"),
    "random": (INK_MUTED, "--", 1.0, 2.0, "Random 2% (control)", "random"),
}


def drift_curves(results: Path, arm: str) -> tuple[list[int], list[float], list[float], dict]:
    """Steps, Jaccard and Spearman against the step-0 baseline, for one arm.

    Step 0 is prepended explicitly: it is the comparison's own baseline, so both
    statistics are 1 there by construction and ``stability.json`` does not store
    a snapshot for it. Leaving it out would hide the single most visible feature
    of the figure, the collapse over the first 500 steps.
    """
    payload = load_json(results / arm / "stability.json")
    snapshots = payload["snapshots"]
    steps = [0] + [int(entry["step"]) for entry in snapshots]
    jaccard = [1.0] + [float(entry["jaccard_vs_baseline"]) for entry in snapshots]
    spearman = [1.0] + [float(entry["spearman_vs_baseline"]) for entry in snapshots]
    return steps, jaccard, spearman, payload


def plot_drift(results: Path, path: Path) -> Path:
    """Membership collapses in every run; the global ranking mostly does not."""
    curves = {arm: drift_curves(results, arm) for arm in DRIFT_FAMILIES}
    any_payload = next(iter(curves.values()))[3]
    num_units = int(any_payload["num_units"])
    top_k = int(any_payload["baseline_selection"]["num_projections"])

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    seen: set[str] = set()
    for arm, family in DRIFT_FAMILIES.items():
        color, linestyle, alpha, width, label, endpoint = DRIFT_STYLES[family]
        steps, jaccard, spearman, _ = curves[arm]
        for panel, values in ((left, jaccard), (right, spearman)):
            panel.plot(
                steps, values,
                color=color, linestyle=linestyle, alpha=alpha, linewidth=width,
                label=label if family not in seen else None,
                zorder=3 if family in ("full", "random") else 2,
            )
        seen.add(family)
        if endpoint is not None:
            right.annotate(
                endpoint,
                (steps[-1], spearman[-1]),
                textcoords="offset points",
                xytext=(6, 0),
                fontsize=9,
                color=INK_SECONDARY,
                va="center",
            )

    left.set_title(
        f"Selected-set membership\nJaccard(top-{top_k} now, top-{top_k} at step 0)",
        fontsize=11, color=INK,
    )
    right.set_title(
        f"Global CA ranking\nSpearman(all {num_units:,} units vs step 0)",
        fontsize=11, color=INK,
    )
    left.set_ylim(0.0, 1.03)
    right.set_ylim(0.5, 1.03)
    for panel in (left, right):
        panel.set_xlabel("training step", fontsize=11)
        panel.set_xlim(0, 10600)
        panel.grid(color=GRID, linewidth=0.8)
        panel.set_axisbelow(True)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)

    legend = left.legend(loc="upper right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    figure.tight_layout()
    return _save(figure, path)


# ==========================================================================================
# Figure 4 - qualitative weather grid
# ==========================================================================================

#: Columns, left to right: the unadapted model, the two style adapters the report
#: contrasts (t=81 vs t=41), the best viewpoint adapter, and full fine-tuning.
QUALITATIVE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("control_base", "Base SDXL (0%)"),
    ("A_t81_style", "Style, t=81"),
    ("C_t41_style", "Style, t=41"),
    ("B_top3_viewpoint", "Viewpoint, multi-t"),
    ("E_all", "Full attention"),
)

#: Snowy is the largest style/viewpoint gap in Table 4. The two indices are the
#: first of that condition, chosen mechanically rather than picked.
QUALITATIVE_CONDITION = "snowy"
QUALITATIVE_INDICES = (800, 801)

#: The grid is 5x2 at ~390 px per cell, so anything past ~512 px of source is
#: thrown away by the resampler and only costs PNG bytes. At 512 the figure
#: lands under 2 MB, which is what the committed original weighs.
QUALITATIVE_CELL_PIXELS = 512

#: Side of one image cell, in inches at the script's 200 dpi. The figure size is
#: *derived* from it (see :func:`plot_qualitative`) rather than guessed, because
#: the cells are square and an axes box even slightly off square opens a visible
#: letterbox gap between the two rows.
QUALITATIVE_CELL_INCHES = 1.79

#: Where the grid sits in the figure, and the gaps between cells. The top margin
#: is the column titles plus the prompt line; ``bbox_inches="tight"`` trims
#: whatever is left over.
QUALITATIVE_REGION = {"left": 0.035, "right": 0.998, "bottom": 0.004, "top": 0.87}
QUALITATIVE_WSPACE = 0.055
QUALITATIVE_HSPACE = 0.006


def plot_qualitative(generated_root: Path, results: Path, path: Path) -> Path:
    """One prompt and one seed per row, five models across."""
    from PIL import Image

    prompt = load_json(results / QUALITATIVE_COLUMNS[0][0] / "eval.json")["protocol"]["prompt"]

    rows, columns = len(QUALITATIVE_INDICES), len(QUALITATIVE_COLUMNS)
    span = QUALITATIVE_REGION
    width = (
        QUALITATIVE_CELL_INCHES
        * (columns + QUALITATIVE_WSPACE * (columns - 1))
        / (span["right"] - span["left"])
    )
    height = (
        QUALITATIVE_CELL_INCHES
        * (rows + QUALITATIVE_HSPACE * (rows - 1))
        / (span["top"] - span["bottom"])
    )
    figure, axes = plt.subplots(rows, columns, figsize=(width, height))
    figure.subplots_adjust(**span, wspace=QUALITATIVE_WSPACE, hspace=QUALITATIVE_HSPACE)
    for column, (arm, title) in enumerate(QUALITATIVE_COLUMNS):
        for row, index in enumerate(QUALITATIVE_INDICES):
            source = (
                generated_root / arm / "generated" / QUALITATIVE_CONDITION / f"{index:05d}.png"
            )
            if not source.exists():
                raise FileNotFoundError(f"{source} ({ARTIFACTS_HINT})")
            with Image.open(source) as handle:
                image = handle.convert("RGB").resize(
                    (QUALITATIVE_CELL_PIXELS, QUALITATIVE_CELL_PIXELS), Image.LANCZOS
                )
            panel = axes[row][column]
            panel.imshow(np.asarray(image))
            panel.set_xticks([])
            panel.set_yticks([])
            for side in panel.spines.values():
                side.set_visible(False)
            if row == 0:
                panel.set_title(title, fontsize=11, color=INK, pad=8)
            if column == 0:
                panel.set_ylabel(
                    f"{QUALITATIVE_CONDITION} #{index}", fontsize=9, color=INK_SECONDARY
                )

    figure.suptitle(
        f'Prompt: "{prompt} in {QUALITATIVE_CONDITION} with [class names]" '
        "(same per-image seed in every column)",
        fontsize=10, color=INK_SECONDARY, y=0.98,
    )
    return _save(figure, path)


# ==========================================================================================
# Entry point
# ==========================================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__ or "", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Results tree; sdxl_cityscapes/<arm>/{eval,stability}.json are read from it",
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--only",
        choices=FIGURES,
        default=None,
        help="Draw a single figure instead of all four",
    )
    parser.add_argument(
        "--measurements-dir",
        type=Path,
        default=None,
        help=f"tsweep_sdxl_head/measurements; needed for the heatmap ({ARTIFACTS_HINT})",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=None,
        help=f"sdxl_cityscapes run dir with <arm>/generated/**; needed for the qualitative "
        f"grid ({ARTIFACTS_HINT})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _apply_style()

    arms = args.results / "sdxl_cityscapes"
    if not arms.is_dir():
        raise SystemExit(f"No results tree at {arms}")
    wanted = FIGURES if args.only is None else (args.only,)
    drawn: list[Path] = []

    if "heatmap" in wanted:
        if args.measurements_dir is None:
            logger.warning(
                "Skipping the heatmap: pass --measurements-dir "
                "<run>/tsweep_sdxl_head/measurements (%s)", ARTIFACTS_HINT,
            )
        else:
            drawn.append(
                plot_heatmap(
                    args.measurements_dir,
                    args.figures_dir / "ca_heatmap_qkvout_t_panelnorm.png",
                )
            )

    if "tradeoff" in wanted:
        drawn.append(plot_tradeoff(arms, args.figures_dir / "tradeoff_scatter.png"))

    if "drift" in wanted:
        drawn.append(plot_drift(arms, args.figures_dir / "drift_curves.png"))

    if "qualitative" in wanted:
        if args.generated_root is None:
            logger.warning(
                "Skipping the qualitative grid: pass --generated-root "
                "<run>/sdxl_cityscapes (%s)", ARTIFACTS_HINT,
            )
        else:
            drawn.append(
                plot_qualitative(
                    args.generated_root, arms, args.figures_dir / "qualitative_weather_grid.png"
                )
            )

    logger.info("Drew %d figure(s): %s", len(drawn), ", ".join(p.name for p in drawn))


if __name__ == "__main__":
    main()
