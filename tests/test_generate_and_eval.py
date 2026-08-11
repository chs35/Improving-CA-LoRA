"""Sampling, class-name extraction and the CMMD / CLIP-Score plumbing.

Everything here runs on the toy adapter and the offline PixelEmbedder: no
downloads, no CLIP, no SDXL. What is being checked is that the stages connect and
that generation is reproducible from its seed, never that the numbers are good —
the toy model's images are noise and the pixel embedder's text side is a hash.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from calora.config import load_config
from calora.data import ClassNameSampler
from calora.data.cityscapes import (
    CITYSCAPES_LABEL_NAMES,
    CITYSCAPES_TRAIN_LABEL_IDS,
    extract_class_names,
)
from calora.eval.embeddings import PixelEmbedder, _features, list_images
from calora.eval.metrics import clip_score, cmmd
from calora.models.toy import ToyAdapter
from calora.utils.seeding import derive_generator


# -- toy sampling ------------------------------------------------------------------------


def test_generation_is_deterministic_in_its_seed():
    """The ablation compares arms image for image, so a repeat draw has to give
    back the identical pixels; a different seed has to give different ones."""
    adapter = ToyAdapter(seed=0, resolution=64)
    prompts = ["photorealistic first-person urban street view"]

    def sample(seed: int):
        return adapter.generate(
            prompts,
            num_inference_steps=4,
            guidance_scale=5.0,
            generator=derive_generator(seed, "gen", 0),
        )[0]

    first, again, other = sample(0), sample(0), sample(1)
    assert np.array_equal(np.asarray(first), np.asarray(again))
    assert not np.array_equal(np.asarray(first), np.asarray(other))


def test_generation_returns_one_image_per_prompt_at_the_model_resolution():
    adapter = ToyAdapter(seed=0, resolution=64)
    images = adapter.generate(["a", "b", "c"], num_inference_steps=3, guidance_scale=1.0)
    assert len(images) == 3
    assert all(image.size == (64, 64) for image in images)
    assert all(image.mode == "RGB" for image in images)


def test_guidance_scale_changes_the_sample():
    """Guidance is applied, not silently dropped: the toy path runs the same
    eps_uncond + g*(eps_cond - eps_uncond) formula the real samplers do."""
    adapter = ToyAdapter(seed=0, resolution=64)

    def sample(guidance: float):
        return np.asarray(
            adapter.generate(
                ["a street"],
                num_inference_steps=4,
                guidance_scale=guidance,
                generator=derive_generator(0, "gen", 0),
            )[0]
        )

    assert not np.array_equal(sample(1.0), sample(7.5))


def test_different_prompts_give_different_images():
    adapter = ToyAdapter(seed=0, resolution=64)
    images = adapter.generate(
        ["a foggy street", "a sunny street"],
        num_inference_steps=4,
        guidance_scale=5.0,
        generator=derive_generator(0, "gen", 0),
    )
    assert not np.array_equal(np.asarray(images[0]), np.asarray(images[1]))


def test_the_scheduler_is_named_for_the_manifest():
    """Two arms sampled under different schedulers are not comparable, so the
    sampler has to be recordable."""
    adapter = ToyAdapter(seed=0)
    assert adapter.scheduler_name == "ToyDDIM"
    assert adapter.describe()["scheduler"] == "ToyDDIM"


def test_an_adapter_without_a_sampler_says_so():
    from calora.models.adapter import DenoiserAdapter

    with pytest.raises(NotImplementedError, match="cannot generate"):
        DenoiserAdapter.generate(object(), ["a"])


# -- class names --------------------------------------------------------------------------


def _label_png(path, ids):
    """A synthetic labelIds map containing exactly ``ids``."""
    array = np.array(list(ids), dtype=np.uint8).reshape(1, -1)
    Image.fromarray(array, mode="L").save(path)
    return path


def test_extract_class_names_filters_void_and_fixes_the_order(tmp_path):
    # bicycle(33), road(7), sky(23) plus three void-ish ids that must not appear.
    path = _label_png(tmp_path / "labelIds.png", [33, 7, 23, 0, 4, 29])
    assert extract_class_names(path) == ["road", "sky", "bicycle"]


def test_extract_class_names_is_independent_of_pixel_layout(tmp_path):
    """The names go into a prompt, so their order must not depend on which pixel
    came first — otherwise one label map yields two different conditionings."""
    forward = _label_png(tmp_path / "a.png", [7, 26, 23])
    reversed_ = _label_png(tmp_path / "b.png", [23, 26, 7])
    assert extract_class_names(forward) == extract_class_names(reversed_)
    assert extract_class_names(forward) == ["road", "sky", "car"]


def test_extract_class_names_handles_an_all_void_map(tmp_path):
    assert extract_class_names(_label_png(tmp_path / "void.png", [0, 1, 2, 3])) == []


def test_the_train_19_table_is_the_official_one():
    assert len(CITYSCAPES_TRAIN_LABEL_IDS) == 19
    assert len(CITYSCAPES_LABEL_NAMES) == 34  # ids 0..33
    names = [CITYSCAPES_LABEL_NAMES[i] for i in CITYSCAPES_TRAIN_LABEL_IDS]
    assert names[:4] == ["road", "sidewalk", "building", "wall"]
    assert names[-3:] == ["train", "motorcycle", "bicycle"]
    # Nothing void-ish sneaks in: those are exactly the ids Cityscapes ignores.
    assert not {0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30} & set(
        CITYSCAPES_TRAIN_LABEL_IDS
    )


def test_the_dummy_class_name_sampler_is_deterministic_and_varies(repo_root):
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    sampler = ClassNameSampler(cfg.data, seed=0)
    assert sampler(3) == sampler(3)
    draws = [tuple(sampler(i)) for i in range(12)]
    assert len(set(draws)) > 1
    assert all(names for names in draws)


# -- prompt plan ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generate_images(repo_root):
    """scripts/generate_images.py as a module (it is not part of the package)."""
    import importlib.util
    import sys

    sys.path.insert(0, str(repo_root / "src"))
    spec = importlib.util.spec_from_file_location(
        "generate_images", repo_root / "scripts" / "generate_images.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_prompts_covers_both_families(generate_images, repo_root):
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    records = generate_images.build_prompts(cfg, "both")

    expected = cfg.eval.num_images + len(cfg.eval.weather_conditions) * cfg.eval.images_per_condition
    assert len(records) == expected
    assert [r["index"] for r in records] == list(range(expected))

    in_domain = [r for r in records if r["condition"] == "in_domain"]
    weather = [r for r in records if r["condition"] != "in_domain"]
    assert len(in_domain) == cfg.eval.num_images
    assert len(weather) == len(cfg.eval.weather_conditions) * cfg.eval.images_per_condition

    # in-domain is the plain training prompt; weather inserts the condition.
    for record in in_domain:
        assert record["prompt"].startswith(cfg.eval.prompt)
        assert " in " not in record["prompt"].removeprefix(cfg.eval.prompt).split(" with ")[0]
    for record in weather:
        assert f" in {record['condition']}" in record["prompt"]
    # "... with [Class names]" (Table 12), read off a label map per image.
    assert all(" with " in r["prompt"] and r["class_names"] for r in records)


def test_build_prompts_modes_are_exclusive(generate_images, repo_root):
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    only_in = generate_images.build_prompts(cfg, "in_domain")
    only_weather = generate_images.build_prompts(cfg, "weather")
    assert {r["condition"] for r in only_in} == {"in_domain"}
    assert {r["condition"] for r in only_weather} == set(cfg.eval.weather_conditions)


def test_build_prompts_is_reproducible(generate_images, repo_root):
    """The plan is rebuilt on a resumed run, so it has to come out identical or
    --skip-existing would silently mix two prompt sets in one directory."""
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    assert generate_images.build_prompts(cfg, "both") == generate_images.build_prompts(cfg, "both")


def test_build_prompts_can_omit_class_names(generate_images, repo_root):
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    cfg.eval.include_class_names = False
    records = generate_images.build_prompts(cfg, "in_domain")
    assert all(r["class_names"] == [] and " with " not in r["prompt"] for r in records)
    assert records[0]["prompt"] == cfg.eval.prompt


def test_generated_images_are_grouped_by_condition(generate_images, tmp_path, repo_root):
    """CMMD reads one directory and CLIP-Score is reported per condition, so the
    in-domain set must not be mixed in with the weather sets on disk."""
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    records = generate_images.build_prompts(cfg, "both")
    paths = [generate_images._image_path(tmp_path, record) for record in records]
    folders = {path.parent.name for path in paths}
    assert folders == {"in_domain", *(c.replace(" ", "_") for c in cfg.eval.weather_conditions)}
    assert len(set(paths)) == len(paths)


# -- the script as the operator runs it ----------------------------------------------------


def test_a_resumed_run_manifest_is_identical_to_an_uninterrupted_one(
    generate_images, repo_root, tmp_path, monkeypatch
):
    """--skip-existing must still describe the images it skipped.

    It did not: the skip branch left ``path`` off those records, so a run that was
    resumed wrote a manifest naming only the images the *last* process drew. That
    is silent — the PNGs are all on disk — and evaluate.py then reported "No
    in-domain images in the manifest" and scored a subset. Happened for real on
    runs/sdxl_cityscapes/A_t81_style: 900 images, 316 manifest entries.
    """
    monkeypatch.chdir(repo_root)  # data.augmentation_file is a repo-relative path
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    run_dir = tmp_path / cfg.name  # --set output_dir below, not the config's
    manifest_path = run_dir / "resume_arm" / "generated" / "manifest.json"

    batches: list[int] = []
    real_load_adapter = generate_images.load_adapter

    def counting_load_adapter(model_cfg):
        adapter = real_load_adapter(model_cfg)
        inner = adapter.generate

        def generate(prompts, **kwargs):
            batches.append(len(prompts))
            return inner(prompts, **kwargs)

        adapter.generate = generate
        return adapter

    monkeypatch.setattr(generate_images, "load_adapter", counting_load_adapter)

    def run(*extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "generate_images.py",
                "--config", str(repo_root / "configs" / "tiny_cpu.yaml"),
                "--run-name", "resume_arm",
                "--mode", "both",
                "--set", f"output_dir={tmp_path}",
                *extra,
            ],
        )
        generate_images.main()
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    first = run()
    assert sum(batches) == len(first["images"])
    assert all(
        (run_dir / record["path"]).exists() and "batch_seed_index" in record
        for record in first["images"]
    )

    batches.clear()
    second = run("--skip-existing")
    assert batches == []  # every chunk was already on disk, so nothing was re-drawn
    assert second == first  # ... and the manifest says so, key for key


# -- save hardening ------------------------------------------------------------------------


class _FlakyImage:
    """PIL-image stand-in whose first ``failures`` saves go wrong.

    Two flavours of wrong, both seen on the RunPod network volume: an outright
    ``EIO``, and a write that returns cleanly but leaves a truncated PNG behind.
    """

    def __init__(self, failures: int, *, truncate: bool = False):
        self.failures = failures
        self.truncate = truncate
        self.calls = 0

    def save(self, path) -> None:
        import errno
        import io

        self.calls += 1
        if self.calls > self.failures:
            Image.new("RGB", (8, 8), (7, 11, 13)).save(path)
            return
        if not self.truncate:
            raise OSError(errno.EIO, "Input/output error")
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (200, 100, 50)).save(buffer, format="PNG")
        data = buffer.getvalue()
        Path(path).write_bytes(data[: len(data) // 2])


def test_saving_retries_an_eio_with_backoff(generate_images, tmp_path):
    """A real EIO killed a 500-image arm at image 82; one retry recovers it."""
    image = _FlakyImage(1)
    delays: list[float] = []
    path = generate_images.save_image_verified(
        image, tmp_path / "in_domain" / "00000.png", backoff=0.25, sleep=delays.append
    )
    assert image.calls == 2
    assert delays == [0.25]
    with Image.open(path) as handle:
        handle.load()


def test_saving_rejects_and_replaces_a_truncated_file(generate_images, tmp_path):
    """--skip-existing would happily accept a half-written PNG on the resumed
    run, so a file that does not fully decode does not count as saved."""
    image = _FlakyImage(1, truncate=True)
    path = generate_images.save_image_verified(
        image, tmp_path / "00000.png", backoff=0.0, sleep=lambda _delay: None
    )
    assert image.calls == 2
    assert np.asarray(Image.open(path)).shape == (8, 8, 3)


def test_saving_gives_up_after_the_attempt_budget(generate_images, tmp_path):
    image = _FlakyImage(99)
    path = tmp_path / "00000.png"
    with pytest.raises(OSError, match="Giving up"):
        generate_images.save_image_verified(
            image, path, backoff=0.0, sleep=lambda _delay: None
        )
    assert image.calls == generate_images.SAVE_ATTEMPTS
    assert not path.exists()  # no partial file left for --skip-existing to trust


def test_the_writer_thread_saves_every_image_in_order(generate_images, tmp_path):
    saved: list[tuple] = []

    def saver(image, path):
        saved.append((image, path))
        return path

    writer = generate_images._ImageWriter(depth=2, saver=saver)
    jobs = [(f"image-{i}", tmp_path / f"{i:05d}.png") for i in range(5)]
    for image, path in jobs:
        writer.submit(image, path)
    writer.close()
    writer.raise_if_failed()

    assert saved == jobs
    assert not writer.is_alive()


def test_a_writer_failure_reaches_the_main_thread(generate_images, tmp_path):
    """Swallowing it would let the manifest claim images that were never written."""
    import errno

    def saver(image, path):
        raise OSError(errno.EIO, "Input/output error")

    writer = generate_images._ImageWriter(depth=1, saver=saver)
    writer.submit("image", tmp_path / "00000.png")
    writer.close()
    with pytest.raises(OSError, match="Input/output error"):
        writer.raise_if_failed()
    assert not writer.is_alive()


def test_the_writer_verifies_by_default(generate_images, tmp_path):
    writer = generate_images._ImageWriter(depth=1)
    path = tmp_path / "in_domain" / "00000.png"
    writer.submit(Image.new("RGB", (8, 8), (1, 2, 3)), path)
    writer.close()
    writer.raise_if_failed()
    assert np.asarray(Image.open(path)).shape == (8, 8, 3)


def test_both_scripts_share_one_saver(generate_images, evaluate):
    """Two copies of the retry logic would drift, and the reference export is on
    the same volume that killed the generation run."""
    from calora.utils import io_images

    assert generate_images.save_image_verified is io_images.save_image_verified
    assert evaluate.save_image_verified is io_images.save_image_verified


# -- reference export hardening ------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluate(repo_root):
    """scripts/evaluate.py as a module (it is not part of the package)."""
    import importlib.util
    import sys

    sys.path.insert(0, str(repo_root / "src"))
    spec = importlib.util.spec_from_file_location(
        "evaluate", repo_root / "scripts" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def eval_cfg(repo_root, tmp_path, monkeypatch):
    """tiny_cpu pointed at a scratch run dir; the dummy data is repo-relative."""
    monkeypatch.chdir(repo_root)
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    cfg.output_dir = str(tmp_path)
    return cfg


def _truncate(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


def test_export_reference_retries_a_failing_save(evaluate, eval_cfg, monkeypatch):
    """This is the exact call that died: `to_pil(...).save(...)` raised EIO on the
    mfs volume and took the whole eval with it."""
    flaky: list[_FlakyImage] = []

    def flaky_to_pil(_tensor):
        image = _FlakyImage(1)
        flaky.append(image)
        return image

    delays: list[float] = []
    real_save = evaluate.save_image_verified
    monkeypatch.setattr(evaluate, "to_pil", flaky_to_pil)
    monkeypatch.setattr(
        evaluate,
        "save_image_verified",
        lambda image, path: real_save(image, path, backoff=0.25, sleep=delays.append),
    )

    directory = evaluate.export_reference(eval_cfg)

    count = eval_cfg.eval.reference_num_images
    assert len(flaky) == count and all(image.calls == 2 for image in flaky)
    assert delays == [0.25] * count  # one retry each, none of them fatal
    for index in range(count):
        with Image.open(directory / f"{index:05d}.png") as handle:
            handle.load()


def test_export_reference_keeps_intact_files_and_redoes_corrupt_ones(
    evaluate, eval_cfg, monkeypatch
):
    """An eval retry must not re-export 500 images, and must not accept the
    truncated one the failed write left behind."""
    directory = evaluate.export_reference(eval_cfg)
    count = eval_cfg.eval.reference_num_images
    before = {index: (directory / f"{index:05d}.png").read_bytes() for index in range(count)}

    _truncate(directory / "00002.png")

    saved: list[Path] = []
    real_save = evaluate.save_image_verified
    monkeypatch.setattr(
        evaluate,
        "save_image_verified",
        lambda image, path: saved.append(path) or real_save(image, path),
    )
    evaluate.export_reference(eval_cfg)

    assert saved == [directory / "00002.png"]  # only the broken one was redrawn
    after = {index: (directory / f"{index:05d}.png").read_bytes() for index in range(count)}
    assert after == before  # ... and it came back byte-identical, so CMMD is unchanged


def test_export_reference_skips_the_dataset_when_everything_verifies(
    evaluate, eval_cfg, monkeypatch
):
    """The resumed path must not even walk the dataset root: on the pod that is
    a network stat of every Cityscapes frame for nothing."""
    evaluate.export_reference(eval_cfg)

    def refuse(*_args, **_kwargs):
        raise AssertionError("build_dataset must not run when the reference set is intact")

    monkeypatch.setattr(evaluate, "build_dataset", refuse)
    assert evaluate.export_reference(eval_cfg) == eval_cfg.run_dir / "reference"


# -- read hardening ------------------------------------------------------------------------


def _flaky_open(monkeypatch, failures: int) -> list:
    """Make the next ``failures`` ``Image.open`` calls raise EIO."""
    import errno

    calls: list = []
    real_open = Image.open

    def opener(fp, *args, **kwargs):
        calls.append(fp)
        if len(calls) <= failures:
            raise OSError(errno.EIO, "Input/output error")
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", opener)
    return calls


def test_reading_retries_an_eio_with_backoff(tmp_path, monkeypatch):
    from calora.utils.io_images import load_image_rgb

    path = tmp_path / "00000.png"
    Image.new("RGB", (8, 8), (9, 8, 7)).save(path)
    calls = _flaky_open(monkeypatch, 2)

    delays: list[float] = []
    image = load_image_rgb(path, sleep=delays.append)

    assert len(calls) == 3
    assert delays == [1.0, 2.0]
    assert image.mode == "RGB" and image.size == (8, 8)


def test_reading_gives_up_and_raises(tmp_path, monkeypatch):
    """A file that never comes back has to fail the eval, not silently drop an
    image out of the CMMD set."""
    from calora.utils.io_images import READ_ATTEMPTS, load_image_rgb

    path = tmp_path / "00000.png"
    Image.new("RGB", (8, 8), (1, 1, 1)).save(path)
    calls = _flaky_open(monkeypatch, 99)

    with pytest.raises(OSError, match="Input/output error"):
        load_image_rgb(path, sleep=lambda _delay: None)
    assert len(calls) == READ_ATTEMPTS


def test_the_embedders_recover_from_a_transient_read_error(tmp_path, monkeypatch):
    """The retry has to be on the path evaluate.py actually reads images through."""
    monkeypatch.setattr("time.sleep", lambda _delay: None)  # no real backoff in tests
    directory = _image_dir(tmp_path / "imgs", 3)
    paths = list_images(directory)
    expected = PixelEmbedder().embed_images(paths)

    _flaky_open(monkeypatch, 2)
    assert torch.equal(PixelEmbedder().embed_images(paths), expected)


# -- offline embedder ----------------------------------------------------------------------


def _image_dir(directory, count, *, shade=0):
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        value = (shade + 20 * index) % 256
        Image.new("RGB", (32, 32), (value, value // 2, 255 - value)).save(
            directory / f"{index:05d}.png"
        )
    return directory


def test_pixel_embedder_is_deterministic_and_normalised(tmp_path):
    directory = _image_dir(tmp_path / "imgs", 4)
    embedder = PixelEmbedder()
    first = embedder.embed_images(list_images(directory))
    assert torch.equal(first, embedder.embed_images(list_images(directory)))
    assert first.shape == (4, embedder.dim)
    assert torch.allclose(first.norm(dim=-1), torch.ones(4), atol=1e-5)

    texts = embedder.embed_texts(["a foggy street", "a sunny street"])
    assert torch.equal(texts, embedder.embed_texts(["a foggy street", "a sunny street"]))
    assert texts.shape == (2, embedder.dim)
    assert torch.allclose(texts.norm(dim=-1), torch.ones(2), atol=1e-5)
    # Distinct strings must not collide, or CLIP-Score would be a constant.
    assert not torch.allclose(texts[0], texts[1])


def test_pixel_embedder_shares_one_space_for_images_and_text(tmp_path):
    """clip_score dots the two, so a dimension mismatch would only surface here."""
    directory = _image_dir(tmp_path / "imgs", 2)
    embedder = PixelEmbedder()
    images = embedder.embed_images(list_images(directory))
    texts = embedder.embed_texts(["one", "two"])
    assert images.shape[1] == texts.shape[1]


def test_cmmd_runs_end_to_end_with_an_injected_embedder(tmp_path):
    reference = _image_dir(tmp_path / "ref", 6, shade=0)
    near = _image_dir(tmp_path / "near", 6, shade=4)
    far = _image_dir(tmp_path / "far", 6, shade=200)
    embedder = PixelEmbedder()

    close = cmmd(reference, near, embedder=embedder)
    distant = cmmd(reference, far, embedder=embedder)
    assert close["num_reference"] == 6 and close["num_generated"] == 6
    assert 0.0 <= close["cmmd"] < distant["cmmd"]
    assert cmmd(reference, reference, embedder=embedder)["cmmd"] == pytest.approx(0.0, abs=1e-9)


def test_clip_score_pairs_images_with_prompts(tmp_path):
    directory = _image_dir(tmp_path / "imgs", 3)
    paths = list_images(directory)
    embedder = PixelEmbedder()

    result = clip_score(paths, ["a", "b", "c"], embedder=embedder)
    assert result["num_pairs"] == 3
    assert result["clip_score"] >= 0.0  # the metric clamps the cosine at zero
    assert result == clip_score(paths, ["a", "b", "c"], embedder=embedder)

    with pytest.raises(ValueError, match="but"):
        clip_score(paths, ["a", "b"], embedder=embedder)


def test_pixel_embedder_is_flagged_as_unreportable():
    """The docstring is the only thing stopping a stub number reaching the report."""
    assert "never report" in PixelEmbedder.__doc__.lower()


# -- CLIP feature unwrapping ---------------------------------------------------------------
#
# transformers 4.x hands back a Tensor from get_image_features/get_text_features;
# 5.x hands back a ModelOutput whose pooler_output is the projected embedding.
# The metrics have to be identical on both, and no test here may touch the network.


class _Pooled:
    """A 5.x-shaped return: only ``pooler_output`` carries the projection."""

    def __init__(self, pooler_output, **extra):
        self.pooler_output = pooler_output
        for name, value in extra.items():
            setattr(self, name, value)


def test_features_passes_a_plain_tensor_through():
    """The 4.x path: normalise() is called straight on the return value there."""
    tensor = torch.randn(3, 768)
    assert _features(tensor) is tensor


def test_features_reads_pooler_output_when_that_is_all_there_is():
    pooled = torch.randn(2, 768)
    assert _features(_Pooled(pooled)) is pooled
    # Explicit None (rather than absent) is the shape observed on 5.14.1.
    assert _features(_Pooled(pooled, image_embeds=None, text_embeds=None)) is pooled


def test_features_prefers_the_embeds_fields_over_pooler_output():
    """If a later transformers restores them, they are the canonical projection
    and must win over pooler_output — which would then be the unprojected state."""
    embeds, pooled = torch.randn(2, 768), torch.randn(2, 1024)
    assert _features(_Pooled(pooled, image_embeds=embeds)) is embeds
    assert _features(_Pooled(pooled, text_embeds=embeds)) is embeds


def test_features_refuses_an_unrecognised_return_and_names_the_type():
    """Silently guessing here would put an unprojected tensor into CMMD."""

    class Mystery:
        pass

    with pytest.raises(TypeError, match="Mystery"):
        _features(Mystery())
