"""The A100 throughput settings: latent-space random crop, the on-disk latent
cache, the train/callback wall-clock split, TF32 and the config wiring.

What is pinned here is that each of them is either numerically inert or changes
the numbers only in the documented way: the latent crop reads the same pixels a
pixel crop would (exactly, for the patch-wise toy VAE), a cache read from disk
trains bit-identically to a fresh encode, and a corrupt cache costs a re-encode,
never a wrong latent.
"""

from __future__ import annotations

import time

import pytest
import torch

from calora.analysis.stability import build_stability_tracker, make_stability_callback
from calora.ca.measure import prepare_samples
from calora.ca.select import select_topk
from calora.config import LoraConfig_, ModelConfig, TrainConfig, load_config
from calora.data.cityscapes import CityscapesConceptDataset
from calora.data.transforms import hflip, load_full_frame, load_image, random_crop
from calora.models.projections import ProjectionTable
from calora.models.toy import ToyAdapter
from calora.train.latent_store import LatentStore, latents_digest
from calora.train.lora import (
    _draw_crop_offsets,
    _latent_window,
    _trim_to_multiple,
    lora_state_dict,
    train_lora,
)
from calora.utils.device import matmul_tf32_enabled, set_matmul_tf32

AXES = ["style", "viewpoint"]

pytest.importorskip("peft")

RESOLUTION = 64  # toy model resolution; the fake frames are 2:1 like Cityscapes


# -- fixtures ---------------------------------------------------------------------------------


def _wide_frame(path, width=256, height=128, seed=0):
    """A 2:1 frame with pixel-level texture, so every window is distinguishable."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def _fake_cityscapes(root, count=3):
    city_dir = root / "leftImg8bit" / "train" / "aachen"
    city_dir.mkdir(parents=True)
    for index in range(count):
        _wide_frame(city_dir / f"aachen_{index:06d}_000019_leftImg8bit.png", seed=index)
    return root


class _CountingCityscapes(CityscapesConceptDataset):
    """Records how the training loop touches the dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.item_reads = 0
        self.frame_reads = 0

    def __getitem__(self, index):
        self.item_reads += 1
        return super().__getitem__(index)

    def full_frame(self, index):
        self.frame_reads += 1
        return super().full_frame(index)


@pytest.fixture
def random_crop_dataset(tmp_path, augmenter):
    root = _fake_cityscapes(tmp_path / "cs")
    return _CountingCityscapes(
        root, augmenter, resolution=RESOLUTION, crop="random", hflip=True, seed=0,
        caption_template="photorealistic first-person urban street view",
    )


class _CountingAdapter(ToyAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encodes = 0

    def encode_images(self, images):
        self.encodes += images.shape[0]
        return super().encode_images(images)


def _train_config(**overrides) -> TrainConfig:
    cfg = TrainConfig(
        lora=LoraConfig_(rank=2, alpha=2),
        max_steps=4,
        batch_size=2,
        learning_rate=1e-2,
        checkpoint_every=0,
        log_every=1,
        stability_every=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _train(dataset, directory, **kwargs):
    adapter = _CountingAdapter(seed=0, resolution=RESOLUTION)
    table = ProjectionTable.from_model(adapter.unet)
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    config_overrides = kwargs.pop("config", {})
    state = train_lora(
        adapter, dataset, selection, _train_config(**config_overrides),
        output_dir=directory, **kwargs,
    )
    return state, lora_state_dict(adapter.unet), adapter


# -- full frames and latent windows -----------------------------------------------------------


def test_every_pixel_crop_is_a_window_of_the_full_frame(tmp_path):
    path = _wide_frame(tmp_path / "wide.png")
    full = load_full_frame(path, RESOLUTION)
    assert full.shape == (3, RESOLUTION, 2 * RESOLUTION)

    centred = load_image(path, RESOLUTION, crop="center")
    left = (full.shape[-1] - RESOLUTION) // 2
    assert torch.equal(centred, full[..., left:left + RESOLUTION])

    from PIL import Image
    with Image.open(path) as img:
        from calora.data.transforms import resize_short_side, to_tensor
        resized = resize_short_side(img.convert("RGB"), RESOLUTION)
        for offset_seed in range(4):
            generator = torch.Generator().manual_seed(offset_seed)
            window = to_tensor(random_crop(resized, RESOLUTION, generator=generator))
            matches = [
                start for start in range(RESOLUTION + 1)
                if torch.equal(window, full[..., start:start + RESOLUTION])
            ]
            assert matches, "a random crop that is not a window of the full frame"


def test_full_frame_and_caption_at_draw_no_crop(random_crop_dataset):
    """They run on decode threads in any order, so they must not touch the crop RNG."""
    before = random_crop_dataset._crop_draws
    first = random_crop_dataset.full_frame(1)
    assert torch.equal(first, random_crop_dataset.full_frame(1))
    assert random_crop_dataset.caption_at(1) == random_crop_dataset[1]["caption"]
    # Only the __getitem__ above drew a crop.
    assert random_crop_dataset._crop_draws == before + 1


@pytest.mark.parametrize("mirrored", [False, True])
def test_a_latent_window_is_the_encode_of_the_pixel_window(tmp_path, mirrored):
    """Exact for the toy VAE, whose stride equals its kernel (patch-wise encode).

    The real SDXL VAE has a receptive field wider than 8px, which is the
    documented approximation; this test pins the *indexing* — the offset scale,
    the vertical centring and the mirrored offset for flipped frames.
    """
    adapter = ToyAdapter(seed=0, resolution=RESOLUTION)
    frame = load_full_frame(_wide_frame(tmp_path / "wide.png"), RESOLUTION).unsqueeze(0)
    factor = 8
    size = RESOLUTION // factor
    source = hflip(frame) if mirrored else frame
    encoded = adapter.encode_images(source)
    span = encoded.shape[-1] - size
    for left in range(span + 1):
        pixels = frame[..., left * factor:left * factor + RESOLUTION]
        expected = adapter.encode_images(hflip(pixels) if mirrored else pixels)
        window = _latent_window(encoded, left, size, mirrored=mirrored)
        assert torch.allclose(window, expected, atol=1e-6), left


def test_latent_window_rejects_an_offset_outside_the_frame():
    latent = torch.zeros(1, 4, 8, 16)
    with pytest.raises(ValueError, match="outside"):
        _latent_window(latent, 9, 8)


def test_trim_to_multiple_is_symmetric_and_a_noop_when_aligned():
    image = torch.arange(3 * 10 * 21, dtype=torch.float32).reshape(3, 10, 21)
    trimmed = _trim_to_multiple(image, 8)
    assert trimmed.shape == (3, 8, 16)
    assert torch.equal(trimmed, image[:, 1:9, 2:18])
    aligned = torch.zeros(3, 16, 32)
    assert _trim_to_multiple(aligned, 8).shape == aligned.shape


def test_crop_offsets_are_in_range_reproducible_and_varied():
    cfg = _train_config(seed=0)
    draws = [_draw_crop_offsets([8, 0, 3], cfg, step, 0) for step in range(1, 40)]
    assert all(0 <= a <= 8 and b == 0 and 0 <= c <= 3 for a, b, c in draws)
    assert draws == [_draw_crop_offsets([8, 0, 3], cfg, step, 0) for step in range(1, 40)]
    assert len({a for a, _, _ in draws}) > 3


# -- training on the full-frame cache ----------------------------------------------------------


def test_random_crop_training_uses_the_full_frame_cache(random_crop_dataset, tmp_path):
    state, _, adapter = _train(random_crop_dataset, tmp_path / "run")

    assert state.step == 4 and torch.isfinite(torch.tensor(state.loss))
    n = len(random_crop_dataset)
    # One decode per frame, flip done on the tensor; the loop never reads items.
    assert random_crop_dataset.frame_reads == n
    assert random_crop_dataset.item_reads == 0
    # Both orientations encoded once each, and nothing per step.
    assert adapter.encodes == 2 * n


def test_the_loop_slides_the_window_and_mirrors_it_for_flips(
    random_crop_dataset, tmp_path, monkeypatch
):
    """A loop that always read offset 0 would pass every other test here."""
    import calora.train.lora as lora_module

    calls = []
    real = lora_module._latent_window

    def spy(latent, left, size, *, mirrored=False):
        calls.append((left, mirrored, size))
        return real(latent, left, size, mirrored=mirrored)

    monkeypatch.setattr(lora_module, "_latent_window", spy)
    _train(random_crop_dataset, tmp_path / "run", config={"max_steps": 12})

    assert len(calls) == 12 * 2  # max_steps x batch_size
    assert {size for _, _, size in calls} == {RESOLUTION // 8}
    assert len({left for left, _, _ in calls}) > 2
    assert {mirrored for _, mirrored, _ in calls} == {False, True}


def test_full_frame_training_is_reproducible(random_crop_dataset, tmp_path):
    state_a, weights_a, _ = _train(random_crop_dataset, tmp_path / "a")
    state_b, weights_b, _ = _train(random_crop_dataset, tmp_path / "b")
    assert [h["loss"] for h in state_a.history] == [h["loss"] for h in state_b.history]
    assert all(torch.equal(weights_a[k], weights_b[k]) for k in weights_a)


def test_disabling_the_cache_restores_per_step_pixel_crops(random_crop_dataset, tmp_path):
    _, _, adapter = _train(
        random_crop_dataset, tmp_path / "run", config={"cache_latents": False, "prefetch": False}
    )
    assert random_crop_dataset.frame_reads == 0
    assert random_crop_dataset.item_reads == 4 * 2  # max_steps x batch_size
    assert adapter.encodes == 4 * 2


def test_a_random_crop_dataset_without_full_frames_falls_back_loudly(augmenter, tmp_path, caplog):
    from calora.data.dummy import DummyConceptDataset

    class Shifting(DummyConceptDataset):
        crop = "random"

    dataset = Shifting(augmenter, num_items=4, resolution=RESOLUTION, seed=0)
    with caplog.at_level("WARNING"):
        _, _, adapter = _train(dataset, tmp_path / "run", config={"prefetch": False})
    assert "cannot serve uncropped frames" in caplog.text
    assert adapter.encodes == 4 * 2


# -- on-disk latent store ---------------------------------------------------------------------


def _store(tmp_path, dataset, **model_overrides):
    model = ModelConfig(kind="toy", resolution=RESOLUTION, **model_overrides)
    return LatentStore.for_run(tmp_path / "latent_cache", model, dataset)


def test_a_stored_cache_trains_bit_identically_without_encoding(random_crop_dataset, tmp_path):
    store = _store(tmp_path, random_crop_dataset)
    first, weights_first, adapter_first = _train(
        random_crop_dataset, tmp_path / "first", latent_store=store
    )
    assert store.path_for("full", True).exists()
    assert adapter_first.encodes == 2 * len(random_crop_dataset)

    random_crop_dataset.frame_reads = 0
    second, weights_second, adapter_second = _train(
        random_crop_dataset, tmp_path / "second", latent_store=store
    )
    assert adapter_second.encodes == 0
    assert random_crop_dataset.frame_reads == 0
    assert [h["loss"] for h in first.history] == [h["loss"] for h in second.history]
    assert all(torch.equal(weights_first[k], weights_second[k]) for k in weights_first)


def test_the_store_also_serves_a_centre_crop_cache(dataset, tmp_path):
    store = _store(tmp_path, dataset)
    first, weights_first, _ = _train(dataset, tmp_path / "first", latent_store=store)
    assert store.path_for("center", False).exists()
    second, weights_second, adapter = _train(dataset, tmp_path / "second", latent_store=store)
    assert adapter.encodes == 0
    assert first.loss == second.loss
    assert all(torch.equal(weights_first[k], weights_second[k]) for k in weights_first)


def test_a_store_for_another_configuration_is_a_miss(random_crop_dataset, tmp_path):
    latents = [torch.randn(1, 4, 8, 16) for _ in range(len(random_crop_dataset))]
    store = _store(tmp_path, random_crop_dataset)
    assert store.save("full", latents, None) is not None

    other_vae = _store(tmp_path, random_crop_dataset, vae_dtype="bfloat16")
    assert other_vae.path_for("full", False) != store.path_for("full", False)
    assert other_vae.load("full", count=len(latents), with_hflip=False) is None
    # Same file, wrong expectations: a different item count or flip is a miss too.
    assert store.load("full", count=len(latents) + 1, with_hflip=False) is None
    assert store.load("full", count=len(latents), with_hflip=True) is None
    loaded, flipped = store.load("full", count=len(latents), with_hflip=False)
    assert flipped is None and all(torch.equal(a, b) for a, b in zip(loaded, latents))


def test_a_corrupted_store_is_rejected_and_re_encoded(random_crop_dataset, tmp_path, caplog):
    """The network volume corrupted writes silently more than once; a flipped
    byte inside the tensor data must cost a re-encode, not a wrong latent."""
    store = _store(tmp_path, random_crop_dataset)
    _, weights_clean, _ = _train(random_crop_dataset, tmp_path / "clean", latent_store=store)
    path = store.path_for("full", True)

    data = bytearray(path.read_bytes())
    data[-100] ^= 0xFF  # well inside the tensor payload, past the JSON header
    path.write_bytes(bytes(data))
    assert store.load("full", count=len(random_crop_dataset), with_hflip=True) is None

    with caplog.at_level("WARNING"):
        _, weights_again, adapter = _train(
            random_crop_dataset, tmp_path / "again", latent_store=store
        )
    assert "sha256" in caplog.text
    assert adapter.encodes == 2 * len(random_crop_dataset)
    assert all(torch.equal(weights_clean[k], weights_again[k]) for k in weights_clean)
    # ... and the rewrite healed the file.
    assert store.load("full", count=len(random_crop_dataset), with_hflip=True) is not None


def test_an_unreadable_store_file_is_a_miss(random_crop_dataset, tmp_path):
    store = _store(tmp_path, random_crop_dataset)
    path = store.path_for("full", False)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a safetensors file")
    assert store.load("full", count=3, with_hflip=False) is None


def test_the_digest_covers_both_orientations():
    a = [torch.zeros(1, 2, 2, 2)]
    assert latents_digest(a, None) != latents_digest(a, [torch.ones(1, 2, 2, 2)])


# -- wall-clock split ------------------------------------------------------------------------


def test_callback_time_is_reported_apart_from_training(dataset, tmp_path):
    def slow_callback(state):
        time.sleep(0.05)

    state, _, _ = _train(dataset, tmp_path / "run", callbacks=[slow_callback])
    assert state.callback_elapsed >= 4 * 0.05
    assert state.train_elapsed == pytest.approx(state.elapsed - state.callback_elapsed)
    assert state.train_elapsed < state.elapsed
    entry = state.history[-1]
    assert {"train_elapsed", "callback_elapsed", "elapsed"} <= set(entry)
    # The final step's callback is inside the final elapsed, not after it.
    assert state.to_dict()["callback_elapsed"] == state.callback_elapsed


def test_drift_snapshots_record_their_own_cost(dataset, tmp_path):
    adapter = ToyAdapter(seed=0)
    table = ProjectionTable.from_model(adapter.unet)
    samples = prepare_samples(adapter, dataset, axes=AXES)
    trained = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    tracker = build_stability_tracker(
        adapter, samples, axes=AXES, timestep=81, fraction=0.2, trained_selection=trained
    )
    callback = make_stability_callback(
        adapter, samples, tracker, axes=AXES, every=1, max_steps=2, fraction=0.2
    )
    state = train_lora(
        adapter, dataset, trained, _train_config(max_steps=2, learning_rate=0.0),
        output_dir=tmp_path, callbacks=[callback],
    )
    seconds = [snapshot.seconds for snapshot in tracker.snapshots]
    assert len(seconds) == 2 and all(s > 0 for s in seconds)
    assert tracker.summary()["total_seconds"] == pytest.approx(sum(seconds))
    assert tracker.to_dict()["snapshots"][0]["seconds"] == seconds[0]
    assert state.callback_elapsed >= sum(seconds)


# -- TF32 and configs ------------------------------------------------------------------------


@pytest.fixture
def restore_tf32():
    before = matmul_tf32_enabled()
    yield
    set_matmul_tf32(before)


def test_allow_tf32_is_opt_in_and_only_ever_switched_on_by_the_loader(restore_tf32):
    from calora.models.loader import load_adapter

    assert ModelConfig().allow_tf32 is False
    set_matmul_tf32(False)
    load_adapter(ModelConfig(kind="toy", device="cpu", resolution=32))
    assert matmul_tf32_enabled() is False
    load_adapter(ModelConfig(kind="toy", device="cpu", resolution=32, allow_tf32=True))
    assert matmul_tf32_enabled() is True


def test_the_reproduction_config_carries_the_throughput_settings(repo_root):
    cfg = load_config(repo_root / "configs" / "sdxl_cityscapes.yaml")
    assert cfg.train.mixed_precision == "bf16"
    assert cfg.model.allow_tf32 is True
    assert cfg.model.enable_gradient_checkpointing is False
    assert cfg.model.dtype == "float32"  # fp32 master weights under autocast
    assert cfg.data.crop == "random" and cfg.data.hflip is True  # Table 12 unchanged
    assert cfg.train.cache_latents is True and cfg.train.latent_cache_dir
    assert cfg.train.stability_every == 0


def test_the_drift_config_differs_only_in_the_drift_cadence(repo_root):
    base = load_config(repo_root / "configs" / "sdxl_cityscapes.yaml")
    drift = load_config(repo_root / "configs" / "sdxl_cityscapes_drift.yaml")
    assert drift.train.stability_every == 500
    assert drift.name == base.name  # same run dir for generate/evaluate
    drift.train.stability_every = base.train.stability_every
    assert drift == base


def test_measurement_scripts_pin_their_own_precision_regime(repo_root):
    """measure_ca.py must stay full fp32 with checkpointing whatever the training
    settings of the config it is given say; the drift gate's reference must stay
    fp32. Static, like the crop/hflip call-site check: a missing line here would
    produce plausible numbers in a different regime."""
    measure = (repo_root / "scripts" / "measure_ca.py").read_text(encoding="utf-8")
    assert "set_matmul_tf32(False)" in measure
    assert "adapter.enable_gradient_checkpointing()" in measure
    gate = (repo_root / "scripts" / "check_drift_autocast_gate.py").read_text(encoding="utf-8")
    assert "set_matmul_tf32(False)" in gate
