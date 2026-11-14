"""LoRA wiring, CA drift tracking and the metric implementations."""

from __future__ import annotations

import itertools
import math
import threading

import pytest
import torch

from calora.analysis.stability import (
    StabilityTracker,
    build_stability_tracker,
    compare_selections,
    drift_autocast_context,
    jaccard,
    kind_mass,
    make_stability_callback,
    merged_lora,
    spearman,
)
from calora.ca.measure import prepare_samples
from calora.ca.select import select_all, select_topk
from calora.config import LoraConfig_, TrainConfig
from calora.eval.metrics import mmd_rbf
from calora.data.dummy import DummyConceptDataset
from calora.models.projections import ProjectionTable
from calora.models.toy import ToyAdapter
from calora.train.lora import (
    _ItemPrefetcher,
    _micro_batch_plan,
    attach_lora,
    lora_parameters,
    lora_state_dict,
    train_lora,
)
from calora.utils.seeding import derive_generator

AXES = ["style", "viewpoint"]

pytest.importorskip("peft")


# -- rank statistics ------------------------------------------------------------------------


def test_spearman_endpoints():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, a.flip(0)) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_value_based():
    a = torch.tensor([1.0, 2.0, 3.0])
    monotone = torch.tensor([10.0, 1000.0, 100000.0])
    assert spearman(a, monotone) == pytest.approx(1.0)


def test_spearman_handles_ties():
    a = torch.tensor([1.0, 1.0, 2.0, 2.0])
    b = torch.tensor([5.0, 5.0, 9.0, 9.0])
    assert spearman(a, b) == pytest.approx(1.0)


def test_jaccard_basics():
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a", "b"], ["c"]) == 0.0
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 1.0


def test_kind_mass_sums_to_one(table):
    scores = torch.rand(len(table))
    mass = kind_mass(scores, table, "kind")
    assert set(mass) == {"q", "k", "v", "out"}
    assert sum(mass.values()) == pytest.approx(1.0)


def test_compare_selections_is_symmetric_with_unit_diagonal(table):
    a = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    b = select_topk(torch.arange(len(table), dtype=torch.float64).flip(0), table, fraction=0.2)
    matrix = compare_selections({"a": a, "b": b})
    assert matrix["a"]["a"] == 1.0
    assert matrix["a"]["b"] == matrix["b"]["a"]


# -- LoRA wiring ------------------------------------------------------------------------------


def test_attach_lora_targets_exactly_the_selection(adapter, table):
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    attach_lora(adapter.unet, selection, LoraConfig_(rank=2, alpha=2))

    adapted = {
        name.rsplit(".lora_A", 1)[0]
        for name, _ in adapter.unet.named_modules()
        if name.endswith(".lora_A.default")
    }
    assert adapted == set(selection.names)
    assert len(lora_parameters(adapter.unet)) == 2 * len(selection)


def test_lora_only_trains_lora_parameters(adapter, table):
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    adapter.unet.requires_grad_(False)
    attach_lora(adapter.unet, selection, LoraConfig_(rank=2, alpha=2))

    trainable = {n for n, p in adapter.unet.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in name for name in trainable)


def test_attach_lora_rejects_an_empty_selection(adapter, table):
    empty = select_all(table)
    empty.names = []
    with pytest.raises(ValueError, match="empty"):
        attach_lora(adapter.unet, empty, LoraConfig_())


def test_merged_lora_restores_the_original_weights(adapter, table):
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    attach_lora(adapter.unet, selection, LoraConfig_(rank=2, alpha=2))

    # Give LoRA a non-zero delta so merging actually changes something.
    for name, param in adapter.unet.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                param.add_(torch.randn_like(param) * 0.1)

    target = adapter.unet.get_submodule(selection.names[0] + ".base_layer")
    before = target.weight.detach().clone()
    with merged_lora(adapter.unet):
        assert not torch.allclose(target.weight, before)
    assert torch.allclose(target.weight, before, atol=1e-6)


def test_projection_identities_survive_injection(adapter, table):
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    attach_lora(adapter.unet, selection, LoraConfig_(rank=2, alpha=2))
    with merged_lora(adapter.unet):
        assert ProjectionTable.from_model(adapter.unet).names == table.names


# -- training loop -------------------------------------------------------------------------------


def _train_config(**overrides) -> TrainConfig:
    cfg = TrainConfig(
        lora=LoraConfig_(rank=2, alpha=2),
        max_steps=3,
        batch_size=2,
        checkpoint_every=0,
        log_every=0,
        stability_every=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_training_updates_lora_and_writes_weights(adapter, table, dataset, tmp_path):
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    before = {k: v.clone() for k, v in lora_state_dict(adapter.unet).items()}

    state = train_lora(
        adapter, dataset, selection, _train_config(learning_rate=1e-2), output_dir=tmp_path
    )

    assert state.step == 3
    assert torch.isfinite(torch.tensor(state.loss))
    assert (tmp_path / "lora-final.safetensors").exists()

    after = lora_state_dict(adapter.unet)
    assert after and (not before or any(not torch.equal(after[k], before[k]) for k in after))


def test_training_is_reproducible(dataset, tmp_path):
    def run(seed_dir):
        adapter = ToyAdapter(seed=0)
        table = ProjectionTable.from_model(adapter.unet)
        selection = select_topk(
            torch.arange(len(table), dtype=torch.float64), table, fraction=0.2
        )
        state = train_lora(
            adapter, dataset, selection, _train_config(learning_rate=1e-2), output_dir=seed_dir
        )
        return state.loss, lora_state_dict(adapter.unet)

    loss_a, weights_a = run(tmp_path / "a")
    loss_b, weights_b = run(tmp_path / "b")
    assert loss_a == loss_b
    assert all(torch.equal(weights_a[k], weights_b[k]) for k in weights_a)


def test_the_latent_cache_does_not_change_the_numbers(dataset, tmp_path):
    """Caching is an optimisation, not a change of experiment.

    The VAE takes the latent distribution's *mode*, so encoding an image once up
    front and encoding it again on every step have to agree exactly — otherwise
    `train.cache_latents` would quietly become an experimental variable.
    """

    def run(directory, cache_latents):
        adapter = ToyAdapter(seed=0)
        table = ProjectionTable.from_model(adapter.unet)
        selection = select_topk(
            torch.arange(len(table), dtype=torch.float64), table, fraction=0.2
        )
        state = train_lora(
            adapter,
            dataset,
            selection,
            _train_config(learning_rate=1e-2, cache_latents=cache_latents),
            output_dir=directory,
        )
        return state.loss, lora_state_dict(adapter.unet)

    cached_loss, cached_weights = run(tmp_path / "cached", True)
    plain_loss, plain_weights = run(tmp_path / "plain", False)
    assert cached_loss == plain_loss
    assert all(torch.equal(cached_weights[k], plain_weights[k]) for k in cached_weights)


# -- dataset prefetch -----------------------------------------------------------------------


class _OrderSensitiveDataset:
    """A dataset whose reads are order-dependent, exactly like the real one.

    ``CityscapesConceptDataset._crop_generator`` derives the random crop from
    ``(seed, index, self._crop_draws)`` and bumps ``_crop_draws`` on every read,
    so which window an item gets depends on how many reads came before it. This
    stand-in reproduces that coupling, which is what lets the identity test below
    actually fail if the prefetcher ever reads out of order or from two threads.
    """

    crop = "random"
    hflip = False

    def __init__(self, base):
        self._base = base
        self._crop_draws = 0
        self.reads: list[int] = []
        self.reader_threads: set[str] = set()

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict:
        item = dict(self._base[index])
        generator = derive_generator(0, "cityscapes-crop", index, self._crop_draws)
        self._crop_draws += 1
        self.reads.append(index)
        self.reader_threads.add(threading.current_thread().name)
        shift = int(torch.randint(0, 8, (1,), generator=generator))
        item["image"] = item["image"].roll(shift, dims=-1)
        return item


def test_prefetching_does_not_change_the_numbers(augmenter, tmp_path):
    """``train.prefetch`` is an overlap, not an experimental variable.

    Five steps of the random-crop path, once sequentially and once with the
    worker thread. The crop of read *n* depends on *n*, so identical losses are
    only possible if the thread read the same items in the same order.
    """

    def run(directory, prefetch):
        adapter = ToyAdapter(seed=0)
        table = ProjectionTable.from_model(adapter.unet)
        selection = select_topk(
            torch.arange(len(table), dtype=torch.float64), table, fraction=0.2
        )
        dataset = _OrderSensitiveDataset(
            DummyConceptDataset(augmenter, num_items=4, resolution=64, seed=0)
        )
        state = train_lora(
            adapter,
            dataset,
            selection,
            _train_config(max_steps=5, learning_rate=1e-2, log_every=1, prefetch=prefetch),
            output_dir=directory,
        )
        return [entry["loss"] for entry in state.history], lora_state_dict(adapter.unet), dataset

    plain_losses, plain_weights, plain_data = run(tmp_path / "plain", False)
    fast_losses, fast_weights, fast_data = run(tmp_path / "prefetch", True)

    assert len(plain_losses) == 5
    assert fast_losses == plain_losses  # bit-identical, not approx
    assert all(torch.equal(fast_weights[k], plain_weights[k]) for k in plain_weights)
    assert fast_data.reads == plain_data.reads
    # ... and the flag really did move the reads off the training thread.
    assert plain_data.reader_threads == {"MainThread"}
    assert fast_data.reader_threads == {"calora-prefetch"}


def test_the_prefetcher_preserves_the_read_order_and_shuts_down():
    reads: list[int] = []

    class Recorder:
        def __len__(self) -> int:
            return 8

        def __getitem__(self, index: int) -> dict:
            reads.append(index)
            return {"index": index}

    plan = [[0, 3], [3, 1], [2, 2]]
    prefetcher = _ItemPrefetcher(Recorder(), iter(plan), depth=1)
    try:
        served = [[item["index"] for item in prefetcher.next()] for _ in plan]
    finally:
        prefetcher.close()

    assert served == plan
    assert reads == [index for batch in plan for index in batch]
    assert not prefetcher.is_alive()


def test_the_prefetcher_reraises_a_read_error_in_the_consumer():
    """A corrupt PNG has to kill the run, not the worker thread silently."""

    class Exploding:
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict:
            if index == 2:
                raise RuntimeError("decode failed")
            return {"index": index}

    prefetcher = _ItemPrefetcher(Exploding(), iter([[0], [1], [2], [3]]), depth=1)
    try:
        assert prefetcher.next()[0]["index"] == 0
        assert prefetcher.next()[0]["index"] == 1
        with pytest.raises(RuntimeError, match="decode failed"):
            prefetcher.next()
    finally:
        prefetcher.close()
    assert not prefetcher.is_alive()


def test_the_prefetcher_can_be_closed_mid_read():
    """The error path in ``train_lora`` must not leave a thread parked on a full
    queue for the rest of the process's life."""

    class Endless:
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict:
            return {"index": index}

    prefetcher = _ItemPrefetcher(Endless(), itertools.repeat([0, 1]), depth=1)
    assert prefetcher.next()
    prefetcher.close()
    assert not prefetcher.is_alive()


def test_the_micro_batch_plan_matches_what_the_loop_draws():
    from calora.train.lora import _draw_indices

    cfg = _train_config(max_steps=3, gradient_accumulation_steps=2)
    expected = [
        _draw_indices(7, cfg, step, micro)
        for step in range(1, cfg.max_steps + 1)
        for micro in range(cfg.gradient_accumulation_steps)
    ]
    assert list(_micro_batch_plan(7, cfg)) == expected


# -- text encoder offload -------------------------------------------------------------------


class _EncoderTracker:
    """Adapter stand-in that records where the encoders were asked to live."""

    device = torch.device("cpu")

    def __init__(self):
        self.encoded: list[str] = []
        self.offloads: list[bool] = []
        self.restores = 0

    def encode_prompts(self, prompts):
        self.encoded.extend(prompts)
        return {"prompt_embeds": torch.zeros(len(prompts), 2)}

    def offload_encoders(self, *, include_vae: bool = True) -> None:
        self.offloads.append(include_vae)

    def restore_encoders(self) -> None:
        self.restores += 1


def test_the_text_encoders_are_parked_once_the_captions_are_cached(augmenter):
    """The VAE stays resident (one encode per step with a random crop); the text
    encoders are finished after a handful of template captions."""
    from calora.train.lora import _InputCache, _park_text_encoders

    dataset = DummyConceptDataset(augmenter, num_items=4, resolution=32, seed=0)
    tracker = _EncoderTracker()
    cache = _InputCache(tracker)

    assert _park_text_encoders(tracker, cache, dataset)
    assert tracker.encoded == dataset.distinct_captions()
    assert tracker.offloads == [False]  # include_vae=False

    # Every caption a step can ask for is already cached, so nothing re-encodes.
    cache.conditioning([dataset[i]["caption"] for i in range(len(dataset))])
    assert tracker.encoded == dataset.distinct_captions()
    assert tracker.restores == 0


def test_an_unseen_caption_pulls_the_text_encoders_back(augmenter):
    """Correctness over VRAM: a caption nobody predicted is still encoded."""
    from calora.train.lora import _InputCache, _park_text_encoders

    dataset = DummyConceptDataset(augmenter, num_items=4, resolution=32, seed=0)
    tracker = _EncoderTracker()
    cache = _InputCache(tracker)
    _park_text_encoders(tracker, cache, dataset)

    cond = cache.conditioning(["a caption nobody predicted"])
    assert cond["prompt_embeds"].shape[0] == 1
    assert tracker.restores == 1
    assert tracker.offloads == [False, False]  # parked again straight afterwards


def test_a_dataset_that_cannot_list_its_captions_keeps_its_encoders(augmenter):
    from calora.train.lora import _InputCache, _park_text_encoders

    tracker = _EncoderTracker()
    cache = _InputCache(tracker)
    assert not _park_text_encoders(tracker, cache, [{"caption": "x"}])
    assert tracker.offloads == []


def test_distinct_captions_covers_every_item(augmenter):
    dataset = DummyConceptDataset(augmenter, num_items=8, resolution=16, seed=0)
    assert set(dataset.distinct_captions()) == {
        dataset[i]["caption"] for i in range(len(dataset))
    }


# -- horizontal flip -------------------------------------------------------------------------


def _flip_dataset(augmenter):
    from calora.data.dummy import DummyConceptDataset

    return DummyConceptDataset(augmenter, num_items=4, resolution=64, hflip=True, seed=0)


def test_the_latent_cache_does_not_change_the_numbers_with_hflip(augmenter, tmp_path):
    """The hflip case of the caching invariant.

    Both orientations are encoded up front, and the flip decision is drawn from
    ``(seed, step, micro)`` rather than from the read order, so the cached run
    has to reproduce the uncached run exactly. If the cache instead flipped the
    cached *latent* — ``VAE(flip(x)) != flip(VAE(x))`` — this is the test that
    would catch it.
    """
    dataset = _flip_dataset(augmenter)

    def run(directory, cache_latents):
        adapter = ToyAdapter(seed=0)
        table = ProjectionTable.from_model(adapter.unet)
        selection = select_topk(
            torch.arange(len(table), dtype=torch.float64), table, fraction=0.2
        )
        state = train_lora(
            adapter,
            dataset,
            selection,
            _train_config(learning_rate=1e-2, cache_latents=cache_latents),
            output_dir=directory,
        )
        return state.loss, lora_state_dict(adapter.unet)

    cached_loss, cached_weights = run(tmp_path / "cached", True)
    plain_loss, plain_weights = run(tmp_path / "plain", False)
    assert cached_loss == plain_loss
    assert all(torch.equal(cached_weights[k], plain_weights[k]) for k in cached_weights)


def test_flipping_latents_is_not_the_same_as_encoding_flipped_pixels(augmenter):
    """The reason the cache holds two orientations instead of one.

    The toy "VAE" is a strided convolution, exactly like the real one in the
    respect that matters: its receptive field grid is not mirror-symmetric.
    """
    from calora.data.transforms import hflip

    adapter = ToyAdapter(seed=0)
    image = _flip_dataset(augmenter)[0]["image"].unsqueeze(0)
    assert not torch.allclose(
        adapter.encode_images(hflip(image)), hflip(adapter.encode_images(image))
    )


def test_the_flip_draw_is_derived_and_reproducible():
    """Derived from (seed, step, micro), never the global RNG: the cached and
    uncached paths have to make the identical decision without talking."""
    from calora.train.lora import _draw_flips

    cfg = _train_config()
    assert _draw_flips(8, cfg, 3, 0) == _draw_flips(8, cfg, 3, 0)
    torch.manual_seed(1234)  # global RNG must not be an input
    assert _draw_flips(8, cfg, 3, 0) == _draw_flips(8, cfg, 3, 0)
    # Different steps see different draws, and both orientations do occur.
    draws = [_draw_flips(4, cfg, step, 0) for step in range(16)]
    flat = [flip for batch in draws for flip in batch]
    assert any(flat) and not all(flat)


# -- reloading a trained arm -----------------------------------------------------------------


def test_load_lora_round_trips_a_trained_arm(dataset, tmp_path):
    """generate_images.py --lora has to restore the exact model that was trained.

    The projections come back out of the checkpoint keys rather than a selection
    JSON, so a stale --selection cannot half-load an arm and still sample.
    """
    from calora.train.lora import load_lora

    trained = ToyAdapter(seed=0)
    table = ProjectionTable.from_model(trained.unet)
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    train_lora(
        trained, dataset, selection, _train_config(learning_rate=1e-1), output_dir=tmp_path
    )
    saved = lora_state_dict(trained.unet)

    restored = ToyAdapter(seed=0)
    reloaded = load_lora(restored.unet, tmp_path / "lora-final.safetensors", LoraConfig_(rank=2))
    assert set(reloaded.names) == set(selection.names)

    weights = lora_state_dict(restored.unet)
    assert set(weights) == set(saved)
    assert all(torch.equal(weights[k], saved[k]) for k in saved)


def test_load_lora_takes_the_rank_from_the_checkpoint(dataset, tmp_path, caplog):
    """The state dict is authoritative; a config that disagrees is a warning, not
    a silently wrong-shaped adapter."""
    from calora.train.lora import load_lora

    trained = ToyAdapter(seed=0)
    table = ProjectionTable.from_model(trained.unet)
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    train_lora(trained, dataset, selection, _train_config(), output_dir=tmp_path)

    restored = ToyAdapter(seed=0)
    with caplog.at_level("WARNING"):
        load_lora(restored.unet, tmp_path / "lora-final.safetensors", LoraConfig_(rank=16))
    assert "rank" in caplog.text
    lora_a = next(v for k, v in lora_state_dict(restored.unet).items() if "lora_A" in k)
    assert lora_a.shape[0] == 2  # the checkpoint's rank, not the config's


def test_load_lora_rejects_a_non_lora_file(adapter, tmp_path):
    from safetensors.torch import save_file

    from calora.train.lora import load_lora

    path = tmp_path / "not-lora.safetensors"
    save_file({"conv_in.weight": torch.zeros(2, 2)}, str(path))
    with pytest.raises(ValueError, match="LoRA checkpoint"):
        load_lora(adapter.unet, path, LoraConfig_(rank=2))


def test_alpha_travels_with_the_checkpoint(dataset, tmp_path):
    """``alpha`` scales every delta and is invisible in the tensors.

    Without it in the metadata a reload would fall back to whatever
    ``train.lora.alpha`` happens to say, and an arm trained at alpha=2 but
    reloaded at the default 64 would sample a 32x-amplified LoRA — a model that
    runs, produces images, and is silently the wrong one.
    """
    from safetensors import safe_open

    trained = ToyAdapter(seed=0)
    table = ProjectionTable.from_model(trained.unet)
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    train_lora(trained, dataset, selection, _train_config(), output_dir=tmp_path)

    with safe_open(str(tmp_path / "lora-final.safetensors"), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["lora_alpha"] == "2" and metadata["lora_rank"] == "2"


def test_a_reloaded_arm_generates_the_same_images(dataset, tmp_path):
    """The point of the round trip: sampling has to be indistinguishable from
    sampling the process that trained it, or the eval measures the wrong model.

    Reloaded with the *default* LoraConfig_ (alpha=64) on purpose: the checkpoint
    metadata has to override it, otherwise this test passes only when the caller
    already knew the answer.
    """
    import numpy as np

    from calora.train.lora import load_lora
    from calora.utils.seeding import derive_generator

    trained = ToyAdapter(seed=0, resolution=64)
    table = ProjectionTable.from_model(trained.unet)
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    train_lora(
        trained, dataset, selection, _train_config(learning_rate=1e-1), output_dir=tmp_path
    )

    restored = ToyAdapter(seed=0, resolution=64)
    load_lora(restored.unet, tmp_path / "lora-final.safetensors", LoraConfig_())

    def sample(a):
        return np.asarray(
            a.generate(
                ["a street"],
                num_inference_steps=3,
                guidance_scale=5.0,
                generator=derive_generator(0, "gen", 0),
            )[0]
        )

    assert np.array_equal(sample(trained), sample(restored))


def test_load_lora_warns_when_a_checkpoint_predates_the_alpha_metadata(dataset, tmp_path, caplog):
    from safetensors.torch import save_file

    from calora.train.lora import load_lora

    trained = ToyAdapter(seed=0)
    table = ProjectionTable.from_model(trained.unet)
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    attach_lora(trained.unet, selection, LoraConfig_(rank=2, alpha=2))
    path = tmp_path / "legacy.safetensors"
    save_file(lora_state_dict(trained.unet), str(path))  # no metadata at all

    restored = ToyAdapter(seed=0)
    with caplog.at_level("WARNING"):
        load_lora(restored.unet, path, LoraConfig_(rank=2, alpha=2))
    assert "lora_alpha" in caplog.text


def test_hflip_is_off_unless_the_dataset_asks_for_it(adapter, table, dataset, tmp_path):
    """A dataset built by a measurement call site (hflip=False) must train
    bit-identically to one built before hflip existed."""
    from calora.train.lora import _InputCache

    cache = _InputCache(adapter)
    cache.precompute_latents(dataset, with_hflip=False)
    assert cache.latents_flipped is None


# -- mixed precision ------------------------------------------------------------------------


def test_bf16_autocast_trains_on_cpu_with_a_finite_loss(adapter, table, dataset, tmp_path):
    """bf16 is the mode that works without CUDA: full fp32 exponent range, no scaler."""
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    state = train_lora(
        adapter,
        dataset,
        selection,
        _train_config(learning_rate=1e-2, mixed_precision="bf16"),
        output_dir=tmp_path,
    )

    assert state.step == 3
    assert torch.isfinite(torch.tensor(state.loss))
    # Autocast covers the forward only; anything that carries state stays fp32.
    assert all(p.dtype == torch.float32 for p in lora_parameters(adapter.unet))


def test_fp16_mixed_precision_is_rejected_on_cpu(adapter, table, dataset, tmp_path):
    """GradScaler is CUDA-only, so fp16 on CPU must fail loudly rather than silently."""
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    with pytest.raises(ValueError, match="fp16"):
        train_lora(
            adapter,
            dataset,
            selection,
            _train_config(mixed_precision="fp16"),
            output_dir=tmp_path,
        )
    # Rejected before the UNet was touched.
    assert not lora_parameters(adapter.unet)


def test_unknown_mixed_precision_is_rejected(adapter, table, dataset, tmp_path):
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    with pytest.raises(ValueError, match="mixed_precision"):
        train_lora(
            adapter,
            dataset,
            selection,
            _train_config(mixed_precision="float8"),
            output_dir=tmp_path,
        )


def test_base_weights_are_frozen_during_training(adapter, table, dataset, tmp_path):
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    target = selection.names[0]
    before = adapter.unet.get_submodule(target).weight.detach().clone()

    train_lora(adapter, dataset, selection, _train_config(learning_rate=1e-1), output_dir=tmp_path)

    after = adapter.unet.get_submodule(target + ".base_layer").weight.detach()
    assert torch.allclose(before, after)


# -- drift tracking ---------------------------------------------------------------------------


def test_drift_baseline_starts_at_perfect_overlap(adapter, table, dataset, tmp_path):
    """Baselining on the same samples is what makes step-0 drift exactly zero."""
    samples = prepare_samples(adapter, dataset, axes=AXES)
    trained = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    tracker = build_stability_tracker(
        adapter, samples, axes=AXES, timestep=81, fraction=0.2, trained_selection=trained
    )

    callback = make_stability_callback(
        adapter, samples, tracker, axes=AXES, every=1, max_steps=2, fraction=0.2
    )
    train_lora(
        adapter,
        dataset,
        trained,
        _train_config(max_steps=2, learning_rate=0.0),
        output_dir=tmp_path,
        callbacks=[callback],
    )

    assert len(tracker.snapshots) == 2
    # A zero learning rate cannot move the weights, so nothing may drift.
    for snapshot in tracker.snapshots:
        assert snapshot.jaccard_vs_baseline == 1.0
        assert snapshot.spearman_vs_baseline == pytest.approx(1.0)
        assert snapshot.entered == [] and snapshot.left == []


def test_drift_autocast_defaults_to_a_null_context():
    """``measure.drift_autocast="no"`` must not even enter ``torch.autocast``, so
    that the default drift numbers are bit-identical to before the option."""
    import contextlib

    assert isinstance(
        drift_autocast_context(torch.device("cpu"), "no"), contextlib.nullcontext
    )
    assert isinstance(
        drift_autocast_context(torch.device("cpu"), "bf16"), torch.autocast
    )
    with pytest.raises(ValueError, match="drift_autocast"):
        drift_autocast_context(torch.device("cpu"), "fp8")


def test_drift_autocast_bf16_measures_the_same_baseline(adapter, table, dataset, tmp_path):
    """bf16 forward/backward, fp32 accumulation.

    Baseline and snapshots share the mode, so a zero learning rate still has to
    report zero drift — a mismatch between the two would show up as spurious
    movement at step 0.
    """
    samples = prepare_samples(adapter, dataset, axes=AXES)
    trained = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    tracker = build_stability_tracker(
        adapter,
        samples,
        axes=AXES,
        timestep=81,
        fraction=0.2,
        trained_selection=trained,
        drift_autocast="bf16",
    )
    callback = make_stability_callback(
        adapter, samples, tracker, axes=AXES, every=1, max_steps=1, fraction=0.2,
        drift_autocast="bf16",
    )
    train_lora(
        adapter,
        dataset,
        trained,
        _train_config(max_steps=1, learning_rate=0.0),
        output_dir=tmp_path,
        callbacks=[callback],
    )

    snapshot = tracker.snapshots[0]
    assert math.isfinite(snapshot.mean_ca)
    assert snapshot.jaccard_vs_baseline == 1.0
    # The accumulators are fp32 regardless of the forward dtype.
    assert tracker.baseline_scores.dtype == torch.float64


def test_tracker_separates_drift_from_the_trained_set(table):
    scores = torch.arange(len(table), dtype=torch.float64)
    baseline = select_topk(scores, table, fraction=0.2)
    trained = select_topk(scores.flip(0), table, fraction=0.2)
    tracker = StabilityTracker(
        table, scores, baseline, timestep=81, trained_selection=trained
    )

    snapshot = tracker.record(100, scores, baseline)
    assert snapshot.jaccard_vs_baseline == 1.0
    assert snapshot.jaccard_vs_trained == 0.0
    assert tracker.summary()["final_jaccard_vs_trained"] == 0.0


def test_tracker_reports_which_projections_moved(table):
    scores = torch.arange(len(table), dtype=torch.float64)
    baseline = select_topk(scores, table, fraction=0.2)
    tracker = StabilityTracker(table, scores, baseline, timestep=81)

    shifted = select_topk(scores.flip(0), table, fraction=0.2)
    snapshot = tracker.record(500, scores.flip(0), shifted)

    assert snapshot.jaccard_vs_baseline == 0.0
    assert set(snapshot.entered) == set(shifted.names)
    assert set(snapshot.left) == set(baseline.names)
    assert tracker.to_dict()["summary"]["min_jaccard"] == 0.0


def test_tracker_rejects_a_mismatched_score_vector(table):
    with pytest.raises(ValueError, match="does not match"):
        StabilityTracker(table, torch.zeros(3), select_all(table), timestep=1)


# -- metrics ---------------------------------------------------------------------------------


def test_mmd_of_a_distribution_with_itself_is_zero():
    x = torch.randn(16, 8)
    assert mmd_rbf(x, x) == pytest.approx(0.0, abs=1e-9)


def test_mmd_grows_with_separation():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, 8, generator=generator)
    near = x + 0.1
    far = x + 5.0
    assert 0 < mmd_rbf(x, near) < mmd_rbf(x, far)


def test_mmd_is_symmetric():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(32, 8, generator=generator)
    y = torch.randn(32, 8, generator=generator)
    assert mmd_rbf(x, y) == pytest.approx(mmd_rbf(y, x))


def test_mmd_validates_shapes():
    with pytest.raises(ValueError, match="2-D"):
        mmd_rbf(torch.randn(4), torch.randn(4, 4))
    with pytest.raises(ValueError, match="dim mismatch"):
        mmd_rbf(torch.randn(4, 8), torch.randn(4, 16))


def test_miou_is_explicitly_out_of_scope():
    from calora.eval.metrics import segmentation_miou

    with pytest.raises(NotImplementedError, match="out of scope"):
        segmentation_miou()
