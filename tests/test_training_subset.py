"""Few-shot training: the 10-image subset, per-axis drift, and UNet-only autocast.

The subset has to reach the training loop and nothing else, and every arm of a run
has to draw the same images. Per-axis drift has to judge a run on the axis its
selection was made on, which the earlier single-axis tracker did not.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from calora.analysis.stability import (
    build_multi_axis_stability_tracker,
    build_stability_tracker,
    drift_view,
    make_stability_callback,
)
from calora.ca.measure import prepare_samples
from calora.ca.select import select_topk
from calora.config import EvalConfig, LoraConfig_, ModelConfig, TrainConfig, load_config
from calora.data import DummyConceptDataset, apply_training_subset
from calora.models.toy import ToyAdapter
from calora.train.latent_store import LatentStore
from calora.train.lora import _draw_indices, train_lora
from calora.utils.device import module_forward_autocast

AXES = ["style", "viewpoint"]


def _dataset(augmenter, num_items: int = 32) -> DummyConceptDataset:
    return DummyConceptDataset(augmenter, num_items=num_items, resolution=32)


def _train_config(**overrides) -> TrainConfig:
    values = dict(
        max_steps=2,
        batch_size=1,
        learning_rate=1e-2,
        checkpoint_every=0,
        log_every=1,
        lora=LoraConfig_(rank=2, alpha=2),
    )
    return TrainConfig(**{**values, **overrides})


# -- the subset -------------------------------------------------------------------------------


def test_the_draw_depends_only_on_the_seed(augmenter):
    first = apply_training_subset(_dataset(augmenter), 10, seed=0)
    again = apply_training_subset(_dataset(augmenter), 10, seed=0)
    other = apply_training_subset(_dataset(augmenter), 10, seed=1)
    assert first == again
    assert first["indices"] != other["indices"]
    assert len(first["indices"]) == len(set(first["indices"])) == 10


def test_the_restricted_dataset_serves_exactly_the_drawn_images(augmenter):
    full = _dataset(augmenter)
    subset = _dataset(augmenter)
    record = apply_training_subset(subset, 10, seed=0)

    assert len(subset) == 10
    assert record["keys"] == subset.item_keys()
    for position, source in enumerate(record["indices"]):
        assert subset[position]["key"] == full[source]["key"]
        assert torch.equal(subset[position]["image"], full[source]["image"])
        assert subset[position]["caption"] == full[source]["caption"]
    with pytest.raises(IndexError):
        subset[10]


@pytest.mark.parametrize("size", [0, 33])
def test_an_impossible_subset_size_is_refused(augmenter, size):
    with pytest.raises(ValueError, match="subset_size"):
        apply_training_subset(_dataset(augmenter), size)


def test_the_training_loop_only_draws_inside_the_subset(augmenter):
    subset = _dataset(augmenter)
    apply_training_subset(subset, 10, seed=0)
    cfg = _train_config()
    drawn = {i for step in range(500) for i in _draw_indices(len(subset), cfg, step, 0)}
    assert drawn == set(range(10))


def test_a_subset_gets_its_own_latent_cache_file(augmenter, tmp_path):
    model = ModelConfig(kind="toy", device="cpu", resolution=32)
    full = _dataset(augmenter)
    first, second = _dataset(augmenter), _dataset(augmenter)
    apply_training_subset(first, 10, seed=0)
    apply_training_subset(second, 10, seed=1)
    paths = {
        LatentStore.for_run(tmp_path, model, dataset).path_for("full", True)
        for dataset in (full, first, second)
    }
    assert len(paths) == 3


def test_training_runs_on_a_subset(augmenter, tmp_path):
    adapter = ToyAdapter(device="cpu", dtype=torch.float32, resolution=32)
    from calora.models.projections import ProjectionTable

    table = ProjectionTable.from_model(adapter.unet)
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    subset = _dataset(augmenter)
    apply_training_subset(subset, 10, seed=0)
    state = train_lora(adapter, subset, selection, _train_config(), output_dir=tmp_path)
    assert state.step == 2


def _write_cli_config(repo_root, tmp_path, *, subset_seed: int = 0, selfgen_dir=None):
    import yaml

    raw = yaml.safe_load((repo_root / "configs" / "tiny_cpu.yaml").read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path / "runs")
    raw["name"] = "subset"
    raw.setdefault("data", {})["augmentation_file"] = str(
        repo_root / "configs" / "prompt_augmentations.json"
    )
    raw["data"]["num_dummy_items"] = 16
    train = raw.setdefault("train", {})
    train.update(
        max_steps=2,
        checkpoint_every=0,
        log_every=1,
        subset_size=4,
        subset_seed=subset_seed,
        stability_every=2,
        stability_num_samples=6,
        latent_cache_dir="",
    )
    if selfgen_dir is not None:
        raw.setdefault("measure", {}).setdefault("selfgen", {})["dir"] = str(selfgen_dir)
    path = tmp_path / f"config_{subset_seed}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _selection_file(repo_root, config_path, axis):
    """A real head selection for the toy model, recorded on ``axis``."""
    from calora.config import load_config as load
    from calora.models.loader import load_adapter
    from calora.models.projections import ProjectionTable

    cfg = load(config_path)
    adapter = load_adapter(cfg.model)
    table = ProjectionTable.from_model(adapter.unet, granularity=cfg.select.granularity)
    selection = select_topk(
        torch.arange(len(table), dtype=torch.float64),
        table,
        fraction=0.02,
        budget_basis=cfg.select.budget_basis,
    )
    payload = selection.to_dict()
    payload["meta"] = {"timesteps": [81], "axis": axis} if axis else {"seed": 0}
    path = config_path.parent / f"selection_{axis or 'control'}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_train(repo_root, config_path, selection_path, run_name):
    return subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "train_lora.py"),
            "--config",
            str(config_path),
            "--selection",
            str(selection_path),
            "--run-name",
            run_name,
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "4"},
    )


def test_train_lora_end_to_end_records_the_subset_and_both_axes(
    repo_root, tmp_path, tiny_selfgen_dir
):
    config_path = _write_cli_config(repo_root, tmp_path, selfgen_dir=tiny_selfgen_dir)
    run_dir = tmp_path / "runs" / "subset"

    viewpoint = _run_train(
        repo_root, config_path, _selection_file(repo_root, config_path, "viewpoint"), "vp"
    )
    assert viewpoint.returncode == 0, viewpoint.stderr[-2000:]
    control = _run_train(
        repo_root, config_path, _selection_file(repo_root, config_path, None), "control"
    )
    assert control.returncode == 0, control.stderr[-2000:]

    record = json.loads((run_dir / "training_subset.json").read_text(encoding="utf-8"))
    assert record["size"] == 4
    summary = json.loads((run_dir / "vp" / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["training_subset"] == record

    # 6 drift samples out of 4 training images: drift reads its own x0 (the
    # self-generated set, regenerated per snapshot), never the training subset.
    assert "Drift samples: 6, x0 selfgen_current" in viewpoint.stdout + viewpoint.stderr

    vp = json.loads((run_dir / "vp" / "stability.json").read_text(encoding="utf-8"))
    assert vp["primary_axis"] == "viewpoint"
    assert set(vp["axes"]) == set(AXES)
    assert drift_view(vp)["axis"] == "viewpoint"
    assert drift_view(vp, "style")["axis"] == "style"
    assert len(drift_view(vp)["snapshots"]) == 1
    assert vp["x0"]["data_source"] == "selfgen"
    assert vp["x0"]["drift_source"] == "selfgen_current"
    assert drift_view(vp)["snapshots"][0]["generation_seconds"] > 0

    no_axis = json.loads((run_dir / "control" / "stability.json").read_text(encoding="utf-8"))
    assert no_axis["primary_axis"] is None
    assert set(no_axis["axes"]) == set(AXES)
    with pytest.raises(ValueError, match="no primary axis"):
        drift_view(no_axis)


def test_a_run_refuses_a_different_subset_than_its_first_arm(
    repo_root, tmp_path, tiny_selfgen_dir
):
    config_path = _write_cli_config(repo_root, tmp_path, selfgen_dir=tiny_selfgen_dir)
    selection = _selection_file(repo_root, config_path, "style")
    assert _run_train(repo_root, config_path, selection, "first").returncode == 0

    other_path = _write_cli_config(
        repo_root, tmp_path, subset_seed=1, selfgen_dir=tiny_selfgen_dir
    )
    other = _run_train(repo_root, other_path, selection, "second")
    assert other.returncode != 0
    assert "Arms of one run must share it" in other.stderr + other.stdout


def test_the_subset_does_not_reach_the_generation_prompts(repo_root):
    spec = importlib.util.spec_from_file_location(
        "generate_images", repo_root / "scripts" / "generate_images.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    full = module.build_prompts(cfg, "both")
    cfg.train.subset_size = 4
    assert module.build_prompts(cfg, "both") == full


def test_the_fewshot_config_sets_the_subset_and_keeps_the_rest(repo_root):
    cfg = load_config(repo_root / "configs" / "sdxl_cityscapes_fewshot10.yaml")
    base = load_config(repo_root / "configs" / "sdxl_cityscapes_drift.yaml")
    assert cfg.name == "sdxl_cityscapes_fewshot10"
    assert (cfg.train.subset_size, cfg.train.subset_seed) == (10, 0)
    assert cfg.train.stability_every == 500
    assert cfg.train.max_steps == base.train.max_steps == 10_000
    assert cfg.measure == base.measure
    assert cfg.select == base.select
    assert cfg.data == base.data


# -- per-axis drift ---------------------------------------------------------------------------


def _tracker_setup(adapter, table, dataset, primary_axis):
    samples = prepare_samples(adapter, dataset, axes=AXES)
    trained = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=0.2)
    tracker = build_multi_axis_stability_tracker(
        adapter,
        samples,
        axes=AXES,
        timestep=81,
        primary_axis=primary_axis,
        fraction=0.2,
        trained_selection=trained,
    )
    return samples, trained, tracker


def test_each_axis_matches_a_tracker_built_on_that_axis_alone(adapter, table, dataset):
    samples, trained, tracker = _tracker_setup(adapter, table, dataset, "viewpoint")
    assert tracker.primary_axis == "viewpoint"
    for axis in AXES:
        single = build_stability_tracker(
            adapter,
            samples,
            axes=AXES,
            timestep=81,
            selection_axis=axis,
            fraction=0.2,
            trained_selection=trained,
        )
        ours = tracker.trackers[axis]
        assert ours.axis == axis
        assert ours.baseline_selection.names == single.baseline_selection.names
        assert torch.equal(ours.baseline_scores, single.baseline_scores)
    assert not torch.equal(
        tracker.trackers["style"].baseline_scores, tracker.trackers["viewpoint"].baseline_scores
    )


def test_a_callback_records_every_axis_from_one_measurement(
    adapter, table, dataset, tmp_path, monkeypatch
):
    samples, trained, tracker = _tracker_setup(adapter, table, dataset, "style")

    import calora.ca.measure as measure_module

    calls = []
    real = measure_module.measure_concept_awareness

    def counting(*args, **kwargs):
        calls.append(kwargs["axes"])
        return real(*args, **kwargs)

    monkeypatch.setattr(measure_module, "measure_concept_awareness", counting)
    seen = []
    callback = make_stability_callback(
        adapter,
        samples,
        tracker,
        axes=AXES,
        every=1,
        max_steps=1,
        fraction=0.2,
        on_snapshot=seen.append,
    )
    train_lora(
        adapter,
        dataset,
        trained,
        _train_config(max_steps=1, learning_rate=0.0),
        output_dir=tmp_path,
        callbacks=[callback],
    )

    assert len(calls) == 1
    assert [snapshot.axis for snapshot in seen] == AXES
    for axis in AXES:
        snapshot = tracker.trackers[axis].snapshots[0]
        # A zero learning rate cannot move the weights, so each axis agrees with
        # its own baseline.
        assert snapshot.jaccard_vs_baseline == 1.0
        assert snapshot.spearman_vs_baseline == pytest.approx(1.0)
        assert snapshot.to_dict()["axis"] == axis


def test_an_unknown_primary_axis_is_refused(adapter, table, dataset):
    with pytest.raises(ValueError, match="not tracked"):
        _tracker_setup(adapter, table, dataset, "weather")


def test_drift_view_reads_files_from_before_per_axis_tracking():
    legacy = {"timestep": 81, "snapshots": []}
    assert drift_view(legacy) is legacy
    with pytest.raises(ValueError, match="predates"):
        drift_view(legacy, "style")


# -- UNet-only autocast -----------------------------------------------------------------------


class _Probe(nn.Module):
    def forward(self, x):
        return torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"), x


def test_only_the_wrapped_module_runs_under_autocast():
    probe, other = _Probe(), _Probe()
    device = torch.device("cpu")
    with module_forward_autocast(probe, device, "bf16"):
        enabled, dtype, _ = probe(torch.zeros(1))
        assert enabled and dtype == torch.bfloat16
        assert other(torch.zeros(1))[0] is False
    assert "forward" not in vars(probe)
    assert probe(torch.zeros(1))[0] is False


def test_no_means_the_module_is_not_touched():
    probe = _Probe()
    with module_forward_autocast(probe, torch.device("cpu"), "no"):
        assert "forward" not in vars(probe)
        assert probe(torch.zeros(1))[0] is False


def test_the_forward_is_restored_when_the_block_raises():
    probe = _Probe()
    with pytest.raises(RuntimeError):
        with module_forward_autocast(probe, torch.device("cpu"), "bf16"):
            raise RuntimeError
    assert "forward" not in vars(probe)


def test_sampling_works_with_only_the_unet_under_autocast(adapter):
    with module_forward_autocast(adapter.unet, adapter.device, "bf16"):
        images = adapter.generate(
            ["a street"], num_inference_steps=2, generator=torch.Generator().manual_seed(0)
        )
    assert len(images) == 1


def test_autocast_does_not_leak_past_a_real_diffusers_unet():
    """Euler returns latents in the model output's dtype, so an uncast bf16 output
    would turn the trajectory bf16 and the fp32 VAE would refuse to decode it."""
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        block_out_channels=(8, 16),
        layers_per_block=1,
        norm_num_groups=4,
        cross_attention_dim=8,
        attention_head_dim=2,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    ).eval()
    vae = diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(4,),
        norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
    ).eval()
    scheduler = diffusers.EulerDiscreteScheduler()
    scheduler.set_timesteps(2)

    latents = torch.randn(1, 4, 8, 8)
    context = torch.randn(1, 4, 8)
    device = torch.device("cpu")
    with torch.no_grad(), module_forward_autocast(unet, device, "bf16"):
        for timestep in scheduler.timesteps:
            as_tuple = unet(latents, timestep, encoder_hidden_states=context, return_dict=False)[0]
            assert as_tuple.dtype == torch.float32
            as_object = unet(sample=latents, timestep=timestep, encoder_hidden_states=context).sample
            assert as_object.dtype == torch.float32
            latents = scheduler.step(as_tuple, timestep, latents).prev_sample
            assert latents.dtype == torch.float32
        assert vae.decode(latents).sample.dtype == torch.float32


def test_generation_autocast_defaults_off_and_is_validated():
    assert EvalConfig().generation_autocast == "no"
    assert EvalConfig(generation_autocast="bf16").generation_autocast == "bf16"
    with pytest.raises(ValueError, match="generation_autocast"):
        EvalConfig(generation_autocast="fp16")
