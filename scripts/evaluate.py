"""Score a generated arm: CMMD against real data, CLIP-Score against the prompts.

    python scripts/evaluate.py --config configs/sdxl_cityscapes_selfgen_fewshot10.yaml \
        --run-name C_t201_style

Reads ``generated/manifest.json`` from :mod:`scripts.generate_images`, exports the
CMMD reference set once, and writes ``eval.json``.

The arms are only ever compared with each other (CMMD for domain fit, CLIP-Score
for weather-prompt controllability), so the numbers only mean anything if
every arm was scored identically. ``eval.json`` therefore carries the whole
protocol next to the values, and the reference set is exported once per run
directory and reused rather than rebuilt per arm.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calora import cli  # noqa: E402
from calora.config import Config  # noqa: E402
from calora.data import build_dataset, to_pil  # noqa: E402
from calora.eval.embeddings import PixelEmbedder  # noqa: E402
from calora.eval.metrics import CMMD_SCALE, CMMD_SIGMA, clip_score, cmmd  # noqa: E402
from calora.utils.io import dump_json, load_json  # noqa: E402
from calora.utils.io_images import save_image_verified, verify_image  # noqa: E402
from calora.utils.logging import get_logger  # noqa: E402

logger = get_logger("evaluate")


def export_reference(cfg: Config) -> Path:
    """Write the CMMD reference images once, at the model's resolution.

    Cropped exactly the way the training data was read (centre, un-mirrored) so
    that CMMD measures the gap between generated and real *content*, not the gap
    between two framings of the same photographs.

    Resumable per image. The old version only counted ``*.png`` and re-exported
    the whole set otherwise, which is the wrong trade against the RunPod network
    volume in both directions: an ``EIO`` part way through (which is exactly how
    this died on 2026-07-30) meant redoing 500 exports on the retry, while a
    truncated file left by such a write still counted towards the total and was
    fed to CMMD as if it were a photograph. Now each expected file is decoded,
    intact ones are kept, and anything that does not decode is deleted and drawn
    again.
    """
    directory = cfg.run_dir / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    expected = [
        directory / f"{index:05d}.png" for index in range(cfg.eval.reference_num_images)
    ]
    missing = [index for index, path in enumerate(expected) if not _is_intact(path)]
    if not missing:
        logger.info("Reusing %d verified reference images in %s", len(expected), directory)
        return directory

    # Built only once something actually has to be drawn: on the common resumed
    # path this walks the dataset root for nothing.
    dataset = build_dataset(
        cfg.data,
        resolution=cfg.model.resolution,
        seed=cfg.eval.seed,
        crop="center",
        hflip=False,
        split=cfg.eval.reference_split,
    )
    count = min(cfg.eval.reference_num_images, len(dataset))
    todo = [index for index in missing if index < count]
    for index in todo:
        save_image_verified(to_pil(dataset[index]["image"]), expected[index])
    logger.info(
        "Exported %d reference images (%d reused) from split %r to %s",
        len(todo), count - len(todo), cfg.eval.reference_split, directory,
    )
    return directory


def _is_intact(path: Path) -> bool:
    """True if ``path`` exists and decodes; otherwise remove it and say no.

    Deliberately unretried: a re-export costs one image, whereas backing off
    1s + 2s on each of 500 corrupt files would cost the eval half an hour. A
    transient read error therefore just redraws an image that was probably fine,
    which is the cheap direction to be wrong in.
    """
    if not path.exists():
        return False
    try:
        verify_image(path)
    except OSError as exc:
        logger.warning("Reference image %s does not decode (%s); re-exporting", path, exc)
        with contextlib.suppress(OSError):
            path.unlink()
        return False
    return True


def main() -> None:
    parser = cli.base_parser(__doc__ or "")
    parser.add_argument("--run-name", required=True, help="Arm subdirectory under the run dir")
    parser.add_argument(
        "--embedder",
        choices=("clip", "pixel"),
        default="clip",
        help="'pixel' is the offline stand-in; its numbers are not reportable",
    )
    args = parser.parse_args()
    cfg = cli.resolve(args)

    arm_dir = cfg.run_dir / args.run_name
    manifest = load_json(arm_dir / "generated" / "manifest.json")
    images = [record for record in manifest["images"] if "path" in record]
    if not images:
        raise FileNotFoundError(
            f"{arm_dir / 'generated' / 'manifest.json'} lists no rendered images; "
            "run scripts/generate_images.py first"
        )

    if args.embedder == "pixel":
        logger.warning(
            "Using PixelEmbedder: this checks the plumbing only. The resulting CMMD and "
            "CLIP-Score are NOT reportable numbers."
        )
        embedder = PixelEmbedder()
    else:
        from calora.eval.embeddings import CLIPEmbedder

        embedder = CLIPEmbedder(cfg.eval.clip_model)

    results: dict[str, object] = {}

    # -- CMMD: real vs in-domain generations --------------------------------------------
    in_domain = [r for r in images if r["condition"] == "in_domain"]
    if in_domain:
        reference_dir = export_reference(cfg)
        generated_dir = (cfg.run_dir / in_domain[0]["path"]).parent
        results["cmmd"] = cmmd(reference_dir, generated_dir, embedder=embedder)
        logger.info("CMMD = %.4f (lower is better)", results["cmmd"]["cmmd"])
    else:
        logger.warning("No in-domain images in the manifest; skipping CMMD")

    # -- CLIP-Score: per condition and overall ------------------------------------------
    # Grouped per condition because that is how the paper reports it: an average
    # over all conditions can hide one condition the model ignores entirely.
    per_condition: dict[str, dict] = {}
    for condition in sorted({r["condition"] for r in images}):
        group = [r for r in images if r["condition"] == condition]
        per_condition[condition] = clip_score(
            [cfg.run_dir / r["path"] for r in group],
            [r["prompt"] for r in group],
            embedder=embedder,
        )
        logger.info(
            "CLIP-Score[%s] = %.3f (n=%d)",
            condition, per_condition[condition]["clip_score"], len(group),
        )
    overall = clip_score(
        [cfg.run_dir / r["path"] for r in images],
        [r["prompt"] for r in images],
        embedder=embedder,
    )
    weather = [c for c in per_condition if c != "in_domain"]
    results["clip_score"] = {
        "overall": overall,
        "per_condition": per_condition,
        # The paper's headline CLIP-Score is over the weather conditions only;
        # the in-domain prompts are the CMMD set, not the CLIP-Score set.
        "weather_mean": (
            sum(per_condition[c]["clip_score"] for c in weather) / len(weather)
            if weather else None
        ),
    }

    payload = {
        "run": cfg.name,
        "arm": args.run_name,
        "embedder": args.embedder,
        "reportable": args.embedder == "clip",
        "protocol": {
            **manifest.get("protocol", {}),
            "clip_model": cfg.eval.clip_model if args.embedder == "clip" else "pixel-stub",
            "reference_split": cfg.eval.reference_split,
            "reference_num_images": cfg.eval.reference_num_images,
            "cmmd_sigma": CMMD_SIGMA,
            "cmmd_scale": CMMD_SCALE,
        },
        "num_images": len(images),
        **results,
    }
    dump_json(payload, arm_dir / "eval.json")
    logger.info("Wrote %s", arm_dir / "eval.json")


if __name__ == "__main__":
    main()
