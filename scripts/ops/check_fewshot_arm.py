"""Does one arm's stage have the artifacts it should? Exit 0 if so, 1 if not.

    python scripts/ops/check_fewshot_arm.py \
        --run-dir runs/sdxl_cityscapes_selfgen_fewshot10 --arm A_t81_style --stage train

scripts/ops/run_fewshot10.sh skips a stage that passes and stops if a finished
stage does not. The network volume corrupts and loses writes, so this reads the
files back rather than trusting that they exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_IMAGES = 900
EXPECTED_SNAPSHOTS = 20
#: Below this per-image pixel std (0-255) the image is flat: what a NaN latent decodes to.
MIN_PIXEL_STD = 1.0


def _finite_problems(path: Path) -> list[str]:
    import torch
    from safetensors import safe_open

    problems = []
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            if not torch.isfinite(handle.get_tensor(key).to(torch.float32)).all():
                problems.append(f"{path.name}: non-finite tensor {key}")
    return problems


def check_train(arm_dir: Path) -> list[str]:
    final = arm_dir / "lora-final.safetensors"
    if not final.exists():
        return [f"missing {final}"]
    problems = _finite_problems(final)
    summary_path, stability_path = arm_dir / "train_summary.json", arm_dir / "stability.json"
    if not summary_path.exists():
        return [*problems, "missing train_summary.json"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    subset = summary.get("training_subset") or {}
    if subset.get("size") != 10:
        problems.append(f"training subset size is {subset.get('size')}, expected 10")
    if not summary.get("timing", {}).get("stability_every"):
        # Drift off (decided 2026-11-09 for the self-generated-CA rerun): no
        # stability.json, but every intermediate checkpoint must be there and intact
        # so drift can still be measured on them afterwards.
        steps = summary.get("final", {}).get("step") or 0
        every = _checkpoint_every(arm_dir)
        for step in range(every, steps, every) if every else ():
            path = arm_dir / f"lora-step{step:06d}.safetensors"
            if not path.exists():
                problems.append(f"missing {path.name} (needed for post-hoc drift)")
            else:
                problems += _finite_problems(path)
        if not every:
            problems.append("train.checkpoint_every is 0: no checkpoints for post-hoc drift")
        return problems
    if not stability_path.exists():
        return [*problems, "missing stability.json"]
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    for axis, block in stability.get("axes", {}).items():
        if len(block["snapshots"]) != EXPECTED_SNAPSHOTS:
            problems.append(f"{axis}: {len(block['snapshots'])} drift snapshots")
    if set(stability.get("axes", {})) != {"style", "viewpoint"}:
        problems.append("stability.json does not track both axes")
    return problems


def _checkpoint_every(arm_dir: Path) -> int:
    """train.checkpoint_every of the run this arm belongs to (its config.yaml)."""
    import yaml

    config = arm_dir.parent / "config.yaml"
    if not config.exists():
        return 0
    return int((yaml.safe_load(config.read_text(encoding="utf-8")) or {})
               .get("train", {}).get("checkpoint_every") or 0)


def check_generate(arm_dir: Path, run_dir: Path) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from PIL import Image

    manifest_path = arm_dir / "generated" / "manifest.json"
    if not manifest_path.exists():
        return ["missing generated/manifest.json"]
    records = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]
    if len(records) != EXPECTED_IMAGES:
        return [f"{len(records)} images in the manifest"]

    def problem(record: dict) -> str | None:
        path = run_dir / record["path"]
        try:
            with Image.open(path) as image:
                std = float(np.asarray(image.convert("RGB")).std())
        except OSError as exc:
            return f"{path}: {exc}"
        return None if std >= MIN_PIXEL_STD else f"{path}: flat image (std={std:.2f})"

    with ThreadPoolExecutor(8) as pool:
        return [found for found in pool.map(problem, records) if found]


def check_evaluate(arm_dir: Path) -> list[str]:
    path = arm_dir / "eval.json"
    if not path.exists():
        return ["missing eval.json"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [] if "cmmd" in json.dumps(payload) else ["eval.json has no CMMD"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--stage", choices=("train", "generate", "evaluate"), required=True)
    args = parser.parse_args()
    arm_dir = args.run_dir / args.arm
    if args.stage == "train":
        problems = check_train(arm_dir)
    elif args.stage == "generate":
        problems = check_generate(arm_dir, args.run_dir)
    else:
        problems = check_evaluate(arm_dir)
    for line in problems:
        print(f"{args.arm} {args.stage}: {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
