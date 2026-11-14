"""Sample images from a trained arm, for CMMD and CLIP-Score.

    python scripts/generate_images.py --config configs/sdxl_cityscapes_selfgen_fewshot10.yaml \
        --lora runs/sdxl_cityscapes_selfgen_fewshot10/C_t201_style/lora-final.safetensors \
        --run-name C_t201_style --mode both

Two prompt families, matching the paper's two measurements:

* ``in_domain`` — ``eval.num_images`` samples of the plain training prompt. This
  is the set CMMD compares against real Cityscapes.
* ``weather``   — ``eval.images_per_condition`` samples for each of
  ``eval.weather_conditions``. This is the set CLIP-Score is read on, since the
  question there is whether the requested condition actually appears.

Scope note: the paper conditions generation on a real ``gtFine`` label map via a
label generator, and reuses that map as the segmentation ground truth. We have no
label generator (see :func:`calora.eval.metrics.segmentation_miou`), so only the
*prompt* is reproduced — a real label map is drawn and read for the class names
that go into "... with [Class names]". The generated images therefore have no
paired ground truth and cannot feed a downstream mIoU.
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calora import cli  # noqa: E402
from calora.config import Config  # noqa: E402
from calora.data import ClassNameSampler  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.train.lora import load_lora  # noqa: E402
from calora.utils.device import module_forward_autocast  # noqa: E402
from calora.utils.io import dump_json  # noqa: E402

# Saving lives in the package, not here: scripts/evaluate.py exports the CMMD
# reference set against the same flaky volume and scripts/ is not importable.
from calora.utils.io_images import (  # noqa: E402
    SAVE_ATTEMPTS,
    SAVE_BACKOFF_SECONDS,
    save_image_verified,
)
from calora.utils.logging import get_logger  # noqa: E402
from calora.utils.seeding import derive_generator  # noqa: E402

logger = get_logger("generate_images")

MODES = ("in_domain", "weather", "both")

__all__ = ["SAVE_ATTEMPTS", "SAVE_BACKOFF_SECONDS", "save_image_verified"]


def build_prompts(cfg: Config, mode: str) -> list[dict]:
    """One record per image to generate: prompt, condition and per-image seed key.

    Built before any sampling happens so the manifest is a plan, not a log: if a
    run dies half way the resumed run asks for the identical images.
    """
    sampler = ClassNameSampler(cfg.data, seed=cfg.eval.seed) if cfg.eval.include_class_names else None
    records: list[dict] = []

    def add(condition: str, count: int) -> None:
        for _ in range(count):
            index = len(records)
            text = cfg.eval.prompt
            if condition != "in_domain":
                text = f"{text} in {condition}"
            classes = sampler(index) if sampler is not None else []
            if classes:
                text = f"{text} with {', '.join(classes)}"
            records.append(
                {
                    "index": index,
                    "prompt": text,
                    "condition": condition,
                    "class_names": classes,
                    "seed": cfg.eval.seed,
                }
            )

    if mode in ("in_domain", "both"):
        add("in_domain", cfg.eval.num_images)
    if mode in ("weather", "both"):
        for condition in cfg.eval.weather_conditions:
            add(condition, cfg.eval.images_per_condition)
    return records


class _ImageWriter:
    """Encodes and saves finished batches on a single background thread.

    PNG-encoding a 1024x1024 frame and pushing it to a network volume is seconds
    of CPU and I/O that the GPU would otherwise sit through. One thread, not a
    pool: saves stay ordered and the retry backoff of a struggling volume stays
    sequential instead of turning into a thundering herd. The queue is bounded so
    a slow volume throttles generation rather than piling up decoded images.

    The first failure is latched and re-raised in the *main* thread — at the next
    :meth:`submit` or at :meth:`close` — so a dead write can never be mistaken
    for a finished run. That is also why the manifest is written only after
    :meth:`close` returns: the manifest claims every listed image exists.
    """

    def __init__(
        self, *, depth: int = 4, saver: Callable[..., Path] = save_image_verified
    ) -> None:
        self._saver = saver
        self._queue: queue.Queue = queue.Queue(maxsize=max(depth, 1))
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="calora-image-writer", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            if self._error is not None:
                continue  # already failed; drain so nobody blocks on a full queue
            try:
                image, path = job
                self._saver(image, path)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
                self._error = exc

    def submit(self, image, path: Path) -> None:
        self.raise_if_failed()
        self._queue.put((image, path))

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        """Drain the queue and join the thread. Does not raise; see
        :meth:`raise_if_failed`, which the caller runs once the loop is done."""
        self._queue.put(None)
        self._thread.join()

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def main() -> None:
    parser = cli.base_parser(__doc__ or "")
    parser.add_argument("--lora", default=None, help="lora-final.safetensors from train_lora.py")
    parser.add_argument("--run-name", default=None, help="Subdirectory under the run dir")
    parser.add_argument("--mode", choices=MODES, default="both")
    parser.add_argument(
        "--expect-axis",
        default=None,
        help="Refuse to load a checkpoint whose recorded axis is not this "
        "(style/viewpoint). Off by default: one config samples both axes of an arm.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Leave already-rendered images alone"
    )
    args = parser.parse_args()
    cfg = cli.resolve(args)

    arm = args.run_name or (Path(args.lora).parent.name if args.lora else "base")
    output_dir = cfg.run_dir / arm / "generated"

    adapter = load_adapter(cfg.model)
    if args.lora:
        # The granularity is checked, not inferred: a module-granularity arm
        # sampled under a head-granularity config (or the reverse) is a different
        # experiment, and both produce perfectly plausible images.
        loaded = load_lora(
            adapter.unet,
            args.lora,
            cfg.train.lora,
            expect_granularity=cfg.select.granularity,
            expect_axis=args.expect_axis,
        )
        checkpoint_axis = str(loaded.meta.get("axis", "") or "")
        if checkpoint_axis and checkpoint_axis not in arm:
            # Naming convention of the run matrix is "{arm}_{axis}"; a mismatch
            # here is the "wrong --lora" accident that cost a 2.5h generation run.
            logger.warning(
                "Checkpoint axis is %r but the output arm is %r; check --lora.",
                checkpoint_axis, arm,
            )
    else:
        # Worth saying out loud: an un-adapted run is the "no LoRA" control, not
        # an arm, and mixing the two up would invalidate the whole comparison.
        logger.warning("No --lora given; sampling the base model as the control arm")

    records = build_prompts(cfg, args.mode)
    logger.info(
        "Generating %d images for arm '%s' (%d steps, guidance %.1f, %s)",
        len(records), arm, cfg.eval.num_inference_steps, cfg.eval.guidance_scale,
        adapter.scheduler_name,
    )

    # Two batches of lookahead: enough that handing off a finished batch never
    # waits on the previous one, small enough to bound resident PIL images.
    writer = _ImageWriter(depth=2 * cfg.eval.generation_batch_size)
    try:
        for start in range(0, len(records), cfg.eval.generation_batch_size):
            chunk = records[start : start + cfg.eval.generation_batch_size]
            paths = [_image_path(output_dir, record) for record in chunk]
            if args.skip_existing and all(path.exists() for path in paths):
                # The manifest is rebuilt from scratch on every run, so a skipped
                # chunk still has to be stamped: leaving these out wrote a manifest
                # in which already-generated images carried no path, and
                # scripts/evaluate.py then read CMMD/CLIP-Score off whatever subset
                # the last resumed process happened to draw.
                for record, path in zip(chunk, paths):
                    _stamp_output(record, path, cfg.run_dir, chunk[0]["index"])
                continue
            # One generator for the batch, keyed on the first index: the pipeline
            # consumes the stream once for the whole batch, so a per-image generator
            # would only be honoured if the batch size were 1.
            generator = derive_generator(cfg.eval.seed, "gen", chunk[0]["index"])
            with module_forward_autocast(
                adapter.unet, adapter.device, cfg.eval.generation_autocast
            ):
                images = adapter.generate(
                    [record["prompt"] for record in chunk],
                    num_inference_steps=cfg.eval.num_inference_steps,
                    guidance_scale=cfg.eval.guidance_scale,
                    generator=generator,
                )
            for record, path, image in zip(chunk, paths, images):
                writer.submit(image, path)
                _stamp_output(record, path, cfg.run_dir, chunk[0]["index"])
            logger.info("  %d/%d", min(start + len(chunk), len(records)), len(records))
    finally:
        writer.close()
    # Past this point every submitted image is on disk and verified, so the
    # manifest is not claiming files that a failed write left missing.
    writer.raise_if_failed()

    manifest = {
        "run": cfg.name,
        "arm": arm,
        "mode": args.mode,
        "lora": args.lora,
        # The protocol travels with the images: an eval.json alone cannot say
        # whether two arms were sampled the same way.
        "protocol": {
            "prompt": cfg.eval.prompt,
            "include_class_names": cfg.eval.include_class_names,
            "guidance_scale": cfg.eval.guidance_scale,
            "num_inference_steps": cfg.eval.num_inference_steps,
            "scheduler": adapter.scheduler_name,
            "resolution": cfg.model.resolution,
            "seed": cfg.eval.seed,
            "generation_batch_size": cfg.eval.generation_batch_size,
            "generation_autocast": cfg.eval.generation_autocast,
            "weather_conditions": cfg.eval.weather_conditions,
            "model": adapter.describe(),
        },
        "images": records,
    }
    dump_json(manifest, output_dir / "manifest.json")
    logger.info("Wrote %s (%d images)", output_dir / "manifest.json", len(records))


def _stamp_output(record: dict, path: Path, run_dir: Path, batch_seed_index: int) -> None:
    """Record where the image lives and which batch generator drew it.

    Both the generating and the ``--skip-existing`` branch go through here, and
    that is the point: the two must write the same keys in the same order, so a
    resumed run's manifest is indistinguishable from an uninterrupted one's.
    ``batch_seed_index`` is the first index of the chunk — the key
    :func:`derive_generator` was given — not the record's own index.
    """
    record["path"] = str(path.relative_to(run_dir))
    record["batch_seed_index"] = batch_seed_index


def _image_path(output_dir: Path, record: dict) -> Path:
    """``generated/<condition>/NNNNN.png``. Grouped by condition because the
    CLIP-Score is reported per condition and CMMD reads one directory."""
    folder = "in_domain" if record["condition"] == "in_domain" else record["condition"]
    return output_dir / folder.replace(" ", "_") / f"{record['index']:05d}.png"


if __name__ == "__main__":
    main()
