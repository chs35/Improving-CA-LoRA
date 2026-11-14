"""Are the A100's tensor cores in use, and what does each stage cost on them?

    python scripts/ops/check_tensor_cores.py --config configs/sdxl_cityscapes_fewshot10.yaml

Writes runs/<name>_gate/tensor_core_gate.json and prints a verdict. Four parts:

1. The flags the training process runs with, and a TF32 on/off matmul timing.
2. Sampling on base SDXL: {fp32 + TF32, bf16 UNet} x batch sizes. bf16 is adopted
   only if it clears the thresholds below, which were fixed before running.
3. 50 real training steps plus one 64-sample drift snapshot with no gradient
   checkpointing, and the same briefly on E_all, the arm with the most LoRA
   parameters: s/step, snapshot seconds and peak VRAM, in fresh processes.
4. The fp32-vs-bf16 drift gate on the viewpoint axis, which the original gate
   did not cover, at its fixed thresholds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from generate_images import build_prompts  # noqa: E402

from calora.config import load_config  # noqa: E402
from calora.eval.embeddings import CLIPEmbedder  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.utils.device import module_forward_autocast, set_matmul_tf32  # noqa: E402
from calora.utils.seeding import derive_generator  # noqa: E402

#: bf16 sampling is adopted only if all three hold (fixed before the first run).
MIN_SPEEDUP = 1.3
MIN_MEAN_COSINE = 0.99
MIN_PIXEL_STD = 1.0

COMPARE_IMAGES = 32  # fp32 vs bf16 at batch 4, for the CLIP-embedding comparison
TIMING_IMAGES = 16  # the other batch sizes only need a rate
BATCH_SIZES = (4, 8, 16)
PY = sys.executable


class VramPeak:
    """Polls nvidia-smi, so it also sees the memory of child processes."""

    def __enter__(self):
        self.peak, self._stop = 0, threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stop.wait(2.0):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip()
            if out.isdigit():
                self.peak = max(self.peak, int(out))

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def matmul_rates() -> dict:
    a, b = (torch.randn(4096, 4096, device="cuda") for _ in range(2))

    def rate() -> float:
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            (a @ b).sum()
        torch.cuda.synchronize()
        return 20 / (time.perf_counter() - started)

    set_matmul_tf32(False)
    off = rate()
    set_matmul_tf32(True)
    on = rate()
    del a, b
    torch.cuda.empty_cache()
    return {"fp32_per_s": off, "tf32_per_s": on, "tf32_speedup": on / off}


def sample(adapter, cfg, prompts, batch: int, mode: str) -> tuple[list, float]:
    """Images and seconds per image, on the generation script's generator keying."""
    images, started = [], time.perf_counter()
    for start in range(0, len(prompts), batch):
        chunk = prompts[start : start + batch]
        with module_forward_autocast(adapter.unet, adapter.device, mode):
            images += adapter.generate(
                chunk,
                num_inference_steps=cfg.eval.num_inference_steps,
                guidance_scale=cfg.eval.guidance_scale,
                generator=derive_generator(cfg.eval.seed, "gen", start),
            )
    return images, (time.perf_counter() - started) / len(prompts)


def bad_images(images) -> int:
    return sum(float(np.asarray(image.convert("RGB")).std()) < MIN_PIXEL_STD for image in images)


def sampling_gate(cfg) -> dict:
    cfg.eval.num_images = COMPARE_IMAGES // 2
    cfg.eval.images_per_condition = COMPARE_IMAGES // 2 // len(cfg.eval.weather_conditions)
    prompts = [record["prompt"] for record in build_prompts(cfg, "both")]

    adapter = load_adapter(cfg.model)
    adapter.generate(prompts[:1], num_inference_steps=2, guidance_scale=cfg.eval.guidance_scale,
                     generator=derive_generator(0, "warmup"))
    rates: dict[str, dict] = {}
    compared = {}
    for mode in ("no", "bf16"):
        for batch in BATCH_SIZES:
            count = COMPARE_IMAGES if batch == 4 else TIMING_IMAGES
            torch.cuda.reset_peak_memory_stats()
            try:
                images, seconds = sample(adapter, cfg, prompts[:count], batch, mode)
            except Exception as exc:  # noqa: BLE001 - recorded; the verdict below refuses it
                torch.cuda.empty_cache()
                rates[f"{mode}/b{batch}"] = {"failed": f"{type(exc).__name__}: {exc}"[:300]}
                print(f"sampling {mode} batch {batch}: {rates[f'{mode}/b{batch}']}", flush=True)
                continue
            rates[f"{mode}/b{batch}"] = {
                "seconds_per_image": seconds,
                "bad_images": bad_images(images),
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            print(f"sampling {mode} batch {batch}: {rates[f'{mode}/b{batch}']}", flush=True)
            if batch == 4:
                compared[mode] = images
    del adapter
    torch.cuda.empty_cache()

    speedup = cosine_mean = cosine_min = None
    adopt = False
    if {"no", "bf16"} <= set(compared):
        embedder = CLIPEmbedder(cfg.eval.clip_model, device="cuda")
        cosines = (
            embedder.embed_images(compared["no"]) * embedder.embed_images(compared["bf16"])
        ).sum(1)
        del embedder
        torch.cuda.empty_cache()
        cosine_mean, cosine_min = float(cosines.mean()), float(cosines.min())
        speedup = rates["no/b4"]["seconds_per_image"] / rates["bf16/b4"]["seconds_per_image"]
        adopt = (
            speedup >= MIN_SPEEDUP
            and rates["bf16/b4"]["bad_images"] == 0
            and cosine_mean >= MIN_MEAN_COSINE
        )
    mode = "bf16" if adopt else "no"
    usable = {
        batch: rates[f"{mode}/b{batch}"]["seconds_per_image"]
        for batch in BATCH_SIZES
        if "seconds_per_image" in rates[f"{mode}/b{batch}"]
        and not rates[f"{mode}/b{batch}"]["bad_images"]
    }
    best = min(usable, key=usable.get)
    return {
        "rates": rates,
        "bf16_speedup_at_b4": speedup,
        "clip_cosine_mean": cosine_mean,
        "clip_cosine_min": cosine_min,
        "adopt_bf16": adopt,
        "generation_autocast": mode,
        "generation_batch_size": best,
        "seconds_per_image": usable[best],
    }


def child(command: list[str], log: Path) -> dict:
    with VramPeak() as vram, log.open("w", encoding="utf-8") as handle:
        started = time.perf_counter()
        code = subprocess.run(
            command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT
        ).returncode
    text = log.read_text(encoding="utf-8", errors="replace")
    return {
        "returncode": code,
        "seconds": time.perf_counter() - started,
        "peak_vram_mib": vram.peak,
        "oom": "out of memory" in text or "OutOfMemoryError" in text,
    }


def train_probe(config: str, name: str, run_dir: Path, source: Path, arm: str, steps: int,
                num_samples: int) -> dict:
    result = child(
        [
            PY, "scripts/train_lora.py", "--config", config, "--name", name,
            "--selection", str(REPO / source / f"selection_{arm}.json"),
            "--run-name", f"probe_{arm}",
            "--set", f"train.max_steps={steps}", f"train.stability_every={steps}",
            f"train.stability_num_samples={num_samples}", "train.checkpoint_every=0",
        ],
        run_dir / f"probe_{arm}.log",
    )
    if result["returncode"] != 0:
        return result
    arm_dir = run_dir / f"probe_{arm}"
    timing = json.loads((arm_dir / "train_summary.json").read_text(encoding="utf-8"))["timing"]
    stability = json.loads((arm_dir / "stability.json").read_text(encoding="utf-8"))
    axes = stability["axes"]
    first = next(iter(axes.values()))
    return {
        **result,
        "seconds_per_step": timing["train_seconds_per_step"],
        "snapshot_seconds": first["snapshots"][-1]["seconds"],
        "baseline_seconds": timing["drift_baseline_seconds"],
        "primary_axis": stability["primary_axis"],
        "baseline_axes_jaccard": _jaccard(
            axes["style"]["baseline_selection"]["names"],
            axes["viewpoint"]["baseline_selection"]["names"],
        ),
        "baseline_vs_trained": {
            axis: block["baseline_vs_trained_jaccard"] for axis, block in axes.items()
        },
    }


def _jaccard(a, b) -> float:
    return len(set(a) & set(b)) / len(set(a) | set(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", type=Path, default=Path("runs/sdxl_cityscapes"),
                        help="Run directory holding the selection JSONs")
    parser.add_argument("--skip-sampling", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = f"{cfg.name}_gate"
    run_dir = REPO / cfg.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "tensor_core_gate.json"
    report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    def save() -> None:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["device"] = torch.cuda.get_device_name(0)
    report["matmul"] = matmul_rates()
    report["flags"] = {
        "model.allow_tf32": cfg.model.allow_tf32,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "train.mixed_precision": cfg.train.mixed_precision,
        "measure.drift_autocast": cfg.measure.drift_autocast,
        "gradient_checkpointing": cfg.model.enable_gradient_checkpointing,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    print(json.dumps({k: report[k] for k in ("device", "matmul", "flags")}, indent=2), flush=True)
    save()

    if not args.skip_sampling:
        report["sampling"] = sampling_gate(load_config(args.config))
        print(json.dumps(report["sampling"], indent=2), flush=True)
        save()

    report["train_viewpoint"] = train_probe(
        args.config, name, run_dir, args.source, "A_t81_viewpoint", 50, 64
    )
    print("train_viewpoint", report["train_viewpoint"], flush=True)
    save()
    report["train_E_all"] = train_probe(args.config, name, run_dir, args.source, "E_all", 10, 4)
    print("train_E_all", report["train_E_all"], flush=True)
    save()

    report["viewpoint_drift_gate"] = child(
        [
            PY, "scripts/check_drift_autocast_gate.py", "--config", args.config, "--name", name,
            "--run-name", "drift_gate_viewpoint", "--timestep", "81",
            "--set", "select.axis=viewpoint",
        ],
        run_dir / "viewpoint_drift_gate.log",
    )
    gate_path = run_dir / "drift_gate_viewpoint" / "drift_autocast_gate.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        report["viewpoint_drift_gate"].update(
            {key: gate[key] for key in ("pass", "spearman", "jaccard", "speedup")}
        )
    print("viewpoint_drift_gate", report["viewpoint_drift_gate"], flush=True)

    train = report["train_viewpoint"]
    sampling = report.get("sampling", {})
    if train.get("returncode") == 0:
        snapshots = 21 * train["snapshot_seconds"]
        report["estimates"] = {
            "train": cfg.train.max_steps * train["seconds_per_step"] + snapshots + 120,
            "generate": 900 * sampling.get("seconds_per_image", 12.0) + 120,
            "evaluate": 600.0,
        }
    save()

    failed = [
        key for key in ("train_viewpoint", "train_E_all", "viewpoint_drift_gate")
        if report[key].get("returncode") != 0 or report[key].get("oom")
    ]
    if not report["viewpoint_drift_gate"].get("pass"):
        failed.append("viewpoint equivalence")
    print("VERDICT:", "FAIL " + str(failed) if failed else "PASS", flush=True)
    print("estimates:", report.get("estimates"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
