"""The self-generated CA measurement set (the paper's x0 = Phi_T2I(c; theta)).

Covers generation determinism and reuse, the dataset's caption guard, the latent
cache, measurement provenance (write, read, legacy files, refusal to pool), the
precision context and the drift x0 sources.
"""

from __future__ import annotations

import json

import pytest
import torch

from calora.analysis.drift_data import SelfGenDriftProvider, drift_samples, latents_fingerprint
from calora.ca.measure import measure_concept_awareness, prepare_samples
from calora.ca.precision import measurement_precision
from calora.ca.stats import (
    LEGACY_PROVENANCE,
    common_provenance,
    load_measurement,
    save_measurement,
)
from calora.config import load_config
from calora.data import build_measurement_dataset
from calora.data.selfgen import (
    SelfGeneratedConceptDataset,
    SelfGenMismatchError,
    generate_selfgen_set,
    manifest_sha256,
    provenance_for,
    selfgen_latents,
    spec_from_config,
    verify_selfgen_set,
)
from calora.models.loader import load_adapter
from calora.models.projections import ProjectionTable
from calora.utils.device import matmul_tf32_enabled


@pytest.fixture
def cfg(repo_root, tmp_path, monkeypatch):
    monkeypatch.chdir(repo_root)  # data.augmentation_file is repo-relative
    config = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    config.measure.selfgen.dir = str(tmp_path / "selfgen")
    config.measure.selfgen.num_images = 5
    config.measure.selfgen.batch_size = 2
    config.measure.num_samples = 5
    return config


def _generate(cfg, adapter, **kwargs):
    sg = cfg.measure.selfgen
    return generate_selfgen_set(
        adapter, spec_from_config(cfg, adapter), sg.dir,
        num_images=kwargs.pop("num_images", sg.num_images),
        **kwargs,
    )


def _pixels(directory, count):
    from PIL import Image

    out = []
    for i in range(count):
        with Image.open(directory / f"{i:05d}.png") as image:
            out.append(list(image.getdata()))
    return out


class _CountingAdapter:
    """Wraps an adapter and counts generate() calls."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def generate(self, *args, **kwargs):
        self.calls += 1
        return self.inner.generate(*args, **kwargs)


# -- generation ---------------------------------------------------------------------------


def test_generation_is_deterministic_and_the_noise_is_batch_size_invariant(cfg, tmp_path):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    first = _pixels(tmp_path / "selfgen", 5)
    cfg.measure.selfgen.dir = str(tmp_path / "again")
    _generate(cfg, adapter)
    assert _pixels(tmp_path / "again", 5) == first
    assert first[0] != first[1]

    # Another batch size is another spec (batched kernels differ in the last
    # bit), but each image's noise is its own: the pixels agree to rounding.
    cfg.measure.selfgen.dir = str(tmp_path / "other")
    cfg.measure.selfgen.batch_size = 5
    _generate(cfg, adapter)
    second = _pixels(tmp_path / "other", 5)
    worst = max(
        abs(a - b) for img_a, img_b in zip(first, second)
        for px_a, px_b in zip(img_a, img_b) for a, b in zip(px_a, px_b)
    )
    assert worst <= 2


def test_the_batch_size_is_part_of_the_spec(cfg):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    cfg.measure.selfgen.batch_size = 3
    with pytest.raises(SelfGenMismatchError, match="batch_size"):
        _generate(cfg, adapter)


def test_a_complete_set_is_not_generated_again(cfg):
    adapter = _CountingAdapter(load_adapter(cfg.model))
    _generate(cfg, adapter)
    assert adapter.calls == 3  # ceil(5 / 2)
    sha = manifest_sha256(cfg.measure.selfgen.dir)
    _generate(cfg, adapter)
    assert adapter.calls == 3
    assert manifest_sha256(cfg.measure.selfgen.dir) == sha


def test_a_corrupt_or_missing_image_is_regenerated_alone(cfg, tmp_path):
    adapter = _CountingAdapter(load_adapter(cfg.model))
    _generate(cfg, adapter)
    before = _pixels(tmp_path / "selfgen", 5)
    (tmp_path / "selfgen" / "00003.png").write_bytes(b"not a png")
    (tmp_path / "selfgen" / "00001.png").unlink()
    with pytest.raises(OSError):
        verify_selfgen_set(tmp_path / "selfgen", spec_from_config(cfg), num_images=5)

    adapter.calls = 0
    _generate(cfg, adapter)
    assert adapter.calls == 2  # the batches holding images 1 and 3, nothing else
    assert _pixels(tmp_path / "selfgen", 5) == before
    verify_selfgen_set(tmp_path / "selfgen", spec_from_config(cfg), num_images=5)


def test_a_set_of_another_spec_is_refused_not_mixed(cfg):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    cfg.measure.selfgen.guidance_scale = 7.5
    with pytest.raises(SelfGenMismatchError, match="guidance_scale"):
        _generate(cfg, adapter)
    with pytest.raises(SelfGenMismatchError):
        verify_selfgen_set(cfg.measure.selfgen.dir, spec_from_config(cfg), num_images=5)


def test_a_larger_set_serves_a_smaller_request_and_a_partial_one_does_not(cfg):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    verify_selfgen_set(cfg.measure.selfgen.dir, spec_from_config(cfg), num_images=3)
    with pytest.raises(FileNotFoundError, match="5 images, 7 requested"):
        verify_selfgen_set(cfg.measure.selfgen.dir, spec_from_config(cfg), num_images=7)


def test_a_templated_prompt_cannot_be_generated_from(cfg):
    cfg.data.caption_template = "a street in {city}"
    with pytest.raises(ValueError, match="one fixed prompt"):
        spec_from_config(cfg)


# -- dataset and latents --------------------------------------------------------------------


def test_measurement_dataset_serves_the_set_with_c_and_its_augmentations(cfg):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    dataset = build_measurement_dataset(
        cfg, num_items=4, crop="center", hflip=False, augs_per_sample=2
    )
    assert isinstance(dataset, SelfGeneratedConceptDataset)
    assert len(dataset) == 4
    item = dataset[2]
    assert item["caption"] == cfg.data.caption_template
    assert item["image"].shape == (3, cfg.model.resolution, cfg.model.resolution)
    assert set(item["augmented"]) == set(cfg.data.concept_axes)
    assert all(len(v) == 2 for v in item["augmented"].values())
    # Deterministic: the same item twice is the same pixels and prompts.
    again = dataset[2]
    assert torch.equal(item["image"], again["image"])
    assert item["augmented"] == again["augmented"]


def test_the_dataset_refuses_a_caption_the_set_was_not_generated_from(cfg):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    from calora.data import build_augmenter

    with pytest.raises(ValueError, match="must share one c"):
        SelfGeneratedConceptDataset(
            cfg.measure.selfgen.dir,
            build_augmenter(cfg.data),
            caption="sketch of first-person urban street view",
            num_items=2,
            resolution=cfg.model.resolution,
        )


def test_measurement_must_be_centre_cropped_and_unmirrored(cfg):
    with pytest.raises(ValueError, match="crop='center'"):
        build_measurement_dataset(cfg, num_items=2, crop="random", hflip=False)
    with pytest.raises(ValueError, match="hflip=False"):
        build_measurement_dataset(cfg, num_items=2, crop="center", hflip=True)


def test_more_samples_than_generated_images_is_refused(cfg):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    with pytest.raises(ValueError, match="at least that large"):
        build_measurement_dataset(cfg, num_items=6, crop="center", hflip=False)


def test_latents_are_encoded_once_and_match_a_fresh_encode(cfg, monkeypatch):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    dataset = build_measurement_dataset(cfg, num_items=5, crop="center", hflip=False)
    first = selfgen_latents(adapter, dataset, cfg.model)
    fresh = [adapter.encode_images(dataset[i]["image"].unsqueeze(0)) for i in range(5)]
    assert all(torch.equal(a, b) for a, b in zip(first, fresh))

    calls = []
    monkeypatch.setattr(adapter, "encode_images", lambda *a, **k: calls.append(1))
    second = selfgen_latents(adapter, dataset, cfg.model)
    assert not calls, "a cached set must not be re-encoded"
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_precomputed_latents_give_the_same_samples(cfg):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    dataset = build_measurement_dataset(cfg, num_items=3, crop="center", hflip=False)
    axes = cfg.data.concept_axes
    encoded = prepare_samples(adapter, dataset, axes=axes, num_samples=3)
    cached = prepare_samples(
        adapter, dataset, axes=axes, num_samples=3,
        latents=selfgen_latents(adapter, dataset, cfg.model),
    )
    for a, b in zip(encoded, cached):
        assert torch.equal(a.latents, b.latents)
        assert a.augmented == b.augmented


# -- provenance -----------------------------------------------------------------------------


def _measure(adapter, samples, timestep=81):
    table = ProjectionTable.from_model(adapter.unet, granularity="head")
    return measure_concept_awareness(
        adapter, table, samples, timestep=timestep, axes=["style", "viewpoint"], seed=0
    )


def test_provenance_round_trips_and_legacy_files_read_as_dataset(cfg, tmp_path):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    dataset = build_measurement_dataset(cfg, num_items=2, crop="center", hflip=False)
    samples = prepare_samples(adapter, dataset, axes=cfg.data.concept_axes, num_samples=2)
    result = _measure(adapter, samples)

    provenance = {**provenance_for(cfg), "precision": "tf32"}
    assert provenance["data_source"] == "selfgen"
    assert provenance["selfgen_manifest_sha256"] == manifest_sha256(cfg.measure.selfgen.dir)
    result.provenance = provenance
    save_measurement(result, tmp_path / "t0081.safetensors")
    assert load_measurement(tmp_path / "t0081.safetensors").provenance == provenance

    # A file from before the provenance header existed.
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(str(tmp_path / "t0081.safetensors"), framework="pt") as handle:
        metadata = dict(handle.metadata())
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}
    metadata.pop("provenance")
    save_file(tensors, str(tmp_path / "legacy.safetensors"), metadata=metadata)
    assert load_measurement(tmp_path / "legacy.safetensors").provenance == LEGACY_PROVENANCE


def test_measurements_of_different_x0_or_precision_are_not_pooled(cfg):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    dataset = build_measurement_dataset(cfg, num_items=2, crop="center", hflip=False)
    samples = prepare_samples(adapter, dataset, axes=cfg.data.concept_axes, num_samples=2)
    a, b = _measure(adapter, samples, 1), _measure(adapter, samples, 81)
    a.provenance = {"data_source": "selfgen", "selfgen_manifest_sha256": "x", "precision": "fp32"}
    b.provenance = dict(a.provenance)
    assert common_provenance({1: a, 81: b})["data_source"] == "selfgen"
    b.provenance = dict(LEGACY_PROVENANCE)
    with pytest.raises(ValueError, match="data_source"):
        common_provenance({1: a, 81: b})
    b.provenance = {**a.provenance, "precision": "bf16"}
    with pytest.raises(ValueError, match="precision"):
        common_provenance({1: a, 81: b})


def test_measure_ca_cli_writes_selfgen_provenance_and_refuses_mixing(cfg, repo_root, tmp_path):
    import subprocess
    import sys

    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    base = [
        sys.executable, str(repo_root / "scripts" / "measure_ca.py"),
        "--config", str(repo_root / "configs" / "tiny_cpu.yaml"),
        "--set", f"output_dir={tmp_path / 'runs'}", "name=cli",
        f"measure.selfgen.dir={cfg.measure.selfgen.dir}", "measure.selfgen.num_images=5",
        "measure.selfgen.batch_size=2",
        "measure.num_samples=4", "measure.timesteps=[81]",
    ]
    env = {"OMP_NUM_THREADS": "2", "PATH": "/usr/bin:/bin"}
    done = subprocess.run(base, cwd=repo_root, capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    path = tmp_path / "runs" / "cli" / "measurements" / "t0081.safetensors"
    provenance = load_measurement(path).provenance
    assert provenance["data_source"] == "selfgen"
    assert provenance["precision"] == "fp32"
    summary = json.loads((tmp_path / "runs" / "cli" / "measure_summary.json").read_text())
    assert summary["provenance"]["data_source"] == "selfgen"

    # Completing that run dir from real images must fail rather than mix.
    mixed = subprocess.run(
        [*base, "measure.source=dataset", "--skip-existing"],
        cwd=repo_root, capture_output=True, text=True, env=env,
    )
    assert mixed.returncode != 0
    assert "Use a fresh run name" in mixed.stderr


# -- precision ------------------------------------------------------------------------------


@pytest.mark.parametrize("precision, tf32", [("fp32", False), ("tf32", True), ("bf16", True)])
def test_measurement_precision_sets_and_restores_tf32(precision, tf32):
    from calora.utils.device import set_matmul_tf32

    set_matmul_tf32(False)
    with measurement_precision("cpu", precision):
        assert matmul_tf32_enabled() is tf32
        if precision == "bf16":
            assert torch.is_autocast_enabled("cpu")
    assert matmul_tf32_enabled() is False
    with pytest.raises(ValueError):
        with measurement_precision("cpu", "fp16"):
            pass


# -- drift x0 -------------------------------------------------------------------------------


def test_fixed_drift_reads_the_base_set_and_current_regenerates_it(cfg):
    adapter = load_adapter(cfg.model)
    _generate(cfg, adapter)
    cfg.train.stability_num_samples = 3

    cfg.measure.drift_source = "selfgen_fixed"
    fixed, provider, x0 = drift_samples(cfg, adapter)
    assert provider is None and x0["drift_source"] == "selfgen_fixed"
    assert len(fixed) == 3

    cfg.measure.drift_source = "selfgen_current"
    baseline, provider, x0 = drift_samples(cfg, adapter)
    assert isinstance(provider, SelfGenDriftProvider)
    # Step 0 is the base set itself, for both self-generated modes.
    assert latents_fingerprint(baseline) == latents_fingerprint(fixed)
    # Regenerating with unchanged weights reproduces the base set image by image:
    # same prompt, sampler and per-image seeds.
    regenerated = provider(0)
    assert all(torch.equal(a.latents, b.latents) for a, b in zip(regenerated, baseline))
    assert provider.last_generation_seconds > 0

    # Moved weights -> new x0, prompts and augmentations untouched.
    with torch.no_grad():
        for parameter in adapter.unet.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    moved = provider(1)
    assert latents_fingerprint(moved) != latents_fingerprint(baseline)
    assert [s.augmented for s in moved] == [s.augmented for s in baseline]
    assert [s.index for s in moved] == [s.index for s in baseline]


def test_dataset_drift_keeps_the_old_real_image_path(cfg):
    adapter = load_adapter(cfg.model)
    cfg.measure.drift_source = "dataset"
    cfg.train.stability_num_samples = 2
    samples, provider, x0 = drift_samples(cfg, adapter)
    assert provider is None and x0["data_source"] == "dataset"
    assert len(samples) == 2


def test_unknown_sources_are_rejected_at_load(repo_root):
    with pytest.raises(ValueError, match="measure.source"):
        load_config(repo_root / "configs" / "tiny_cpu.yaml", {"measure": {"source": "real"}})
    with pytest.raises(ValueError, match="drift_source"):
        load_config(repo_root / "configs" / "tiny_cpu.yaml", {"measure": {"drift_source": "x"}})
    with pytest.raises(ValueError, match="precision"):
        load_config(repo_root / "configs" / "tiny_cpu.yaml", {"measure": {"precision": "fp16"}})
