from __future__ import annotations

import pytest
import torch
import yaml

from calora.config import Config, load_config, parse_cli_overrides, save_config
from calora.data import build_augmenter, build_dataset
from calora.data.prompts import PromptAugmenter
from calora.data.transforms import to_pil, to_tensor


# -- config ------------------------------------------------------------------------------


def test_repo_configs_all_load(repo_root):
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        cfg = load_config(path)
        assert isinstance(cfg, Config)
        assert cfg.name


def test_base_chain_is_merged(repo_root):
    smoke = load_config(repo_root / "configs" / "smoke_a100.yaml")
    parent = load_config(repo_root / "configs" / "sdxl_cityscapes.yaml")
    assert smoke.name == "smoke_a100"
    assert smoke.model.name == parent.model.name  # inherited
    assert smoke.measure.timesteps == [81]  # overridden
    assert smoke.train.lora.rank == parent.train.lora.rank  # nested inheritance


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"measure": {"num_sampels": 8}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(path)


def test_cli_overrides_are_nested_and_typed():
    assert parse_cli_overrides(["measure.num_samples=32", "model.device=cpu"]) == {
        "measure": {"num_samples": 32},
        "model": {"device": "cpu"},
    }
    assert parse_cli_overrides(["measure.timesteps=[1,81]"]) == {
        "measure": {"timesteps": [1, 81]}
    }
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_cli_overrides(["nope"])


def test_measure_config_rejects_a_nonpositive_aug_count(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"measure": {"augs_per_sample": 0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="augs_per_sample"):
        load_config(path)


def test_drift_autocast_is_off_by_default_and_validated(tmp_path):
    """The drift measurement's precision is part of the arm's provenance, so an
    unknown mode has to fail at config load, not at the first snapshot."""
    assert Config().measure.drift_autocast == "no"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"measure": {"drift_autocast": "fp16"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="drift_autocast"):
        load_config(path)


def test_primary_conflict_quantile_must_be_in_the_sweep(tmp_path):
    """Otherwise the headline conflict_count would come from an unreported threshold."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {"select": {"conflict_sim_quantiles": [0.1, 0.2], "conflict_primary_quantile": 0.3}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflict_primary_quantile"):
        load_config(path)


def test_repo_configs_sweep_includes_their_primary_quantile(repo_root):
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg.select.conflict_primary_quantile in cfg.select.conflict_sim_quantiles
        assert cfg.measure.augs_per_sample >= 1


def test_eval_config_rejects_an_empty_weather_list(tmp_path):
    """The paper's CLIP-Score is defined over weather-conditioned prompts, so an
    empty list would silently reduce the metric to the in-domain prompts."""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"eval": {"weather_conditions": []}}), encoding="utf-8")
    with pytest.raises(ValueError, match="weather_conditions"):
        load_config(path)


def test_eval_config_rejects_duplicate_weather_conditions(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"eval": {"weather_conditions": ["foggy", "rainy", "foggy"]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicates"):
        load_config(path)


@pytest.mark.parametrize(
    "key", ["num_images", "images_per_condition", "reference_num_images",
            "num_inference_steps", "generation_batch_size"],
)
def test_eval_config_rejects_nonpositive_counts(tmp_path, key):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"eval": {key: 0}}), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        load_config(path)


def test_the_fixed_training_prompt_needs_no_city_placeholder(repo_root):
    """Table 12's prompt has no {city}; the caption path must not require one."""
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    assert "{city}" not in cfg.data.caption_template
    built = build_dataset(cfg.data, resolution=32, seed=0)
    captions = {built[i]["caption"] for i in range(len(built))}
    assert captions == {cfg.data.caption_template}


def test_sdxl_config_matches_the_paper_table_12(repo_root):
    """Appendix A / Table 12 values, pinned so a later edit has to be deliberate."""
    cfg = load_config(repo_root / "configs" / "sdxl_cityscapes.yaml")
    assert cfg.data.caption_template == "photorealistic first-person urban street view"
    assert cfg.train.batch_size == 1
    assert cfg.train.gradient_accumulation_steps == 1
    assert cfg.data.hflip is True
    assert cfg.eval.guidance_scale == 5.0
    assert cfg.eval.num_inference_steps == 25
    assert cfg.eval.prompt == cfg.data.caption_template


def test_config_round_trips_through_yaml(tmp_path, repo_root):
    original = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    save_config(original, tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml") == original


# -- prompt augmentation --------------------------------------------------------------------


def test_augmentation_is_deterministic_per_item(augmenter):
    assert augmenter.augment("a street", 3) == augmenter.augment("a street", 3)


def test_different_items_get_different_augmentations(augmenter):
    variants = {augmenter.augment("a street", i)["style"] for i in range(8)}
    assert len(variants) > 1


def test_axes_produce_distinct_prompts(augmenter):
    result = augmenter.augment("a street", 0)
    assert set(result) == {"style", "viewpoint"}
    assert result["style"] != result["viewpoint"]


def test_v3_templates_replace_the_caption_rather_than_extending_it(augmenter):
    """The paper's c_Aug (Appendix B.2) rewrites the fixed training prompt.

    v1/v2 appended to a per-image caption via ``{caption}``; v3 does not, so the
    augmented prompt is the template verbatim and carries no trace of ``c``. The
    ``str.format(caption=...)`` call in ``ConceptAxis.render`` has to survive a
    template with no placeholder, which is what this pins down.
    """
    for axis, prompt in augmenter.augment("a street", 0).items():
        assert "a street" not in prompt
        assert prompt in augmenter.axes[axis].templates


def test_augmented_prompt_differs_from_the_base(augmenter):
    for index in range(8):
        for prompt in augmenter.augment("a street", index).values():
            assert prompt != "a street"


def test_augment_many_is_deterministic_per_item(augmenter):
    """The draw is keyed on (seed, axis, index) alone.

    Nothing about the timestep or the slot enters the key, which is what lets
    every timestep re-read the identical c_Aug set and stay paired — the
    augmenter is never even told which timestep is being measured.
    """
    first = augmenter.augment_many("a street", 3, 4)
    assert first == augmenter.augment_many("a street", 3, 4)
    assert first != augmenter.augment_many("a street", 4, 4)
    # augment() is the single-slot case of the same draw.
    assert augmenter.augment("a street", 3) == {
        axis: prompts[0] for axis, prompts in augmenter.augment_many("a street", 3, 1).items()
    }


def test_augment_many_draws_without_replacement(augmenter):
    # Read the budget off the axis rather than hard-coding it: the c_Aug set is
    # versioned data, and this property holds for whatever size it currently is.
    for index in range(4):
        for axis, template_count in {a: len(x.templates) for a, x in augmenter.axes.items()}.items():
            prompts = augmenter.augment_many("a street", index, template_count)[axis]
            assert len(prompts) == template_count
            assert len(set(prompts)) == template_count, f"{axis} repeated a template"


def test_augs_per_sample_of_four_fits_the_v3_sets(augmenter):
    """The configs draw 4 per axis; v3 ships 5 templates, so it stays a draw
    without replacement and no contribution is correlated with another."""
    for axis in augmenter.axis_names:
        assert len(augmenter.axes[axis].templates) >= 4
        assert len(set(augmenter.augment_many("a street", 0, 4)[axis])) == 4


def test_augment_many_warns_and_repeats_past_the_template_count(augmenter, caplog):
    available = len(augmenter.axes["style"].templates)
    with caplog.at_level("WARNING"):
        prompts = augmenter.augment_many("a street", 0, available + 2)["style"]
    assert len(prompts) == available + 2
    assert len(set(prompts)) == available  # every template once, then two repeats
    assert "with replacement" in caplog.text


def test_augment_many_rejects_a_nonpositive_count(augmenter):
    with pytest.raises(ValueError, match="m >= 1"):
        augmenter.augment_many("a street", 0, 0)


def test_datasets_carry_the_requested_number_of_slots(augmenter):
    from calora.data.dummy import DummyConceptDataset

    item = DummyConceptDataset(augmenter, num_items=2, resolution=32, augs_per_sample=3)[0]
    assert set(item["augmented"]) == {"style", "viewpoint"}
    assert all(len(prompts) == 3 for prompts in item["augmented"].values())


def test_missing_axis_is_reported(repo_root):
    with pytest.raises(KeyError, match="not in"):
        PromptAugmenter.from_file(
            repo_root / "configs" / "prompt_augmentations.json", axes=["nonexistent"]
        )


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        PromptAugmenter.from_file(tmp_path / "nope.json")


# -- datasets -------------------------------------------------------------------------------


def test_dummy_items_have_the_expected_schema(dataset):
    item = dataset[0]
    assert set(item) >= {"image", "caption", "augmented", "index", "city"}
    assert item["image"].shape == (3, 64, 64)
    assert item["image"].min() >= -1.0 and item["image"].max() <= 1.0


def test_dummy_images_are_deterministic_and_distinct(dataset):
    assert torch.equal(dataset[0]["image"], dataset[0]["image"])
    assert not torch.equal(dataset[0]["image"], dataset[1]["image"])


def test_dummy_images_are_not_flat_noise(dataset):
    """Structured content, so VAE latents are representative of real data."""
    image = dataset[0]["image"]
    horizontal_variation = (image[:, :, 1:] - image[:, :, :-1]).abs().mean()
    overall_variation = image.std()
    assert horizontal_variation < overall_variation


def test_build_dataset_dispatches_and_validates(repo_root):
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    built = build_dataset(cfg.data, resolution=64, seed=0)
    assert len(built) == cfg.data.num_dummy_items

    cfg.data.kind = "cityscapes"
    cfg.data.root = ""
    with pytest.raises(ValueError, match="data.root"):
        build_dataset(cfg.data, resolution=64, seed=0)

    cfg.data.kind = "nope"
    with pytest.raises(ValueError, match="Unknown data.kind"):
        build_dataset(cfg.data, resolution=64, seed=0)


def test_build_augmenter_honours_configured_axes(repo_root):
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    assert build_augmenter(cfg.data).axis_names == ["style", "viewpoint"]


def test_missing_cityscapes_root_gives_an_actionable_error(tmp_path, augmenter):
    from calora.data.cityscapes import scan_cityscapes

    with pytest.raises(FileNotFoundError, match="leftImg8bit"):
        scan_cityscapes(tmp_path, "train")


# -- transforms ------------------------------------------------------------------------------


def test_image_tensor_round_trip():
    from PIL import Image

    original = Image.new("RGB", (8, 8), (255, 0, 128))
    tensor = to_tensor(original)
    assert tensor.shape == (3, 8, 8)
    assert tensor[0].max() == pytest.approx(1.0)
    assert list(to_pil(tensor).getdata())[0] == (255, 0, 128)


def test_load_image_crops_to_a_square(tmp_path):
    from PIL import Image

    from calora.data.transforms import load_image

    path = tmp_path / "wide.png"
    Image.new("RGB", (2048, 1024), (10, 20, 30)).save(path)
    assert load_image(path, 256).shape == (3, 256, 256)


def _wide_gradient_image(path, width=2048, height=1024):
    """A frame whose columns are all distinguishable, so a crop reveals its offset."""
    import numpy as np
    from PIL import Image

    ramp = np.linspace(0, 255, width, dtype=np.uint8)
    array = np.repeat(ramp[None, :, None], height, axis=0).repeat(3, axis=2)
    Image.fromarray(array).save(path)
    return path


def test_load_image_rejects_an_unknown_crop_mode(tmp_path):
    from calora.data.transforms import load_image

    path = _wide_gradient_image(tmp_path / "wide.png")
    with pytest.raises(ValueError, match="crop mode"):
        load_image(path, 128, crop="diagonal")


def test_random_crop_moves_the_window_and_stays_reproducible(tmp_path):
    from calora.data.transforms import load_image
    from calora.utils.seeding import derive_generator

    path = _wide_gradient_image(tmp_path / "wide.png")
    centred = load_image(path, 128, crop="center")

    def sample(draw: int):
        return load_image(path, 128, crop="random", generator=derive_generator(0, "crop", 0, draw))

    windows = [sample(i) for i in range(6)]
    assert all(w.shape == (3, 128, 128) for w in windows)
    # Different draws see different pixels, and at least one is off-centre.
    assert any(not torch.equal(w, centred) for w in windows)
    assert any(not torch.equal(windows[0], w) for w in windows[1:])
    # Same derived generator, same window: the run is repeatable.
    assert torch.equal(sample(3), sample(3))


def _fake_cityscapes(root):
    """Minimal on-disk layout that ``scan_cityscapes`` accepts."""
    city_dir = root / "leftImg8bit" / "train" / "aachen"
    city_dir.mkdir(parents=True)
    for index in range(2):
        _wide_gradient_image(city_dir / f"aachen_{index:06d}_000019_leftImg8bit.png")
    return root


def test_cityscapes_center_crop_is_deterministic_but_random_is_not(tmp_path, augmenter):
    from calora.data.cityscapes import CityscapesConceptDataset

    root = _fake_cityscapes(tmp_path / "cs")
    centred = CityscapesConceptDataset(root, augmenter, resolution=128, crop="center")
    assert torch.equal(centred[0]["image"], centred[0]["image"])

    shifting = CityscapesConceptDataset(root, augmenter, resolution=128, crop="random", seed=0)
    reads = [shifting[0]["image"] for _ in range(6)]
    assert any(not torch.equal(reads[0], other) for other in reads[1:])

    with pytest.raises(ValueError, match="crop mode"):
        CityscapesConceptDataset(root, augmenter, resolution=128, crop="sideways")


def test_measurement_call_sites_can_force_a_centre_crop(tmp_path, repo_root):
    """CA pairing across timesteps and axes depends on identical pixels, so the
    measurement path overrides data.crop rather than trusting it."""
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    cfg.data.kind = "cityscapes"
    cfg.data.root = str(_fake_cityscapes(tmp_path / "cs"))
    cfg.data.crop = "random"

    assert build_dataset(cfg.data, resolution=128).crop == "random"
    assert build_dataset(cfg.data, resolution=128, crop="center").crop == "center"


# -- horizontal flip -------------------------------------------------------------------------


def test_hflip_override_mirrors_the_crop_override(tmp_path, repo_root):
    """A mirrored frame is as much a different x0 as a shifted crop is, so the
    measurement path forces it off the same way."""
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    cfg.data.kind = "cityscapes"
    cfg.data.root = str(_fake_cityscapes(tmp_path / "cs"))
    cfg.data.hflip = True

    assert build_dataset(cfg.data, resolution=128).hflip is True
    assert build_dataset(cfg.data, resolution=128, hflip=False).hflip is False


def test_hflip_does_not_make_getitem_nondeterministic(repo_root):
    """The flag only *permits* flipping; the draw happens in the training loop.

    If ``__getitem__`` flipped, the latent cache would have to be switched off
    for the same reason ``crop='random'`` switches it off — and caching both
    orientations would be impossible.
    """
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    cfg.data.hflip = True
    built = build_dataset(cfg.data, resolution=32, seed=0)
    assert torch.equal(built[0]["image"], built[0]["image"])


def test_measurement_scripts_pass_hflip_false(repo_root):
    """Static check of the three measurement call sites named in the invariants.

    They are three separate scripts, so nothing but reading them can prove the
    override is present at each; a missing one produces plausible numbers.
    """
    call_sites = [
        "measure_ca.py",
        "train_lora.py",
        "smoke_test.py",
        # The fp32-vs-bf16 drift gate re-measures the training callback's samples,
        # so it is a fourth measurement call site and carries the same override.
        "check_drift_autocast_gate.py",
    ]
    for name in call_sites:
        source = (repo_root / "scripts" / name).read_text(encoding="utf-8")
        assert 'crop="center"' in source, name
        assert "hflip=False" in source, name


def test_hflip_transform_mirrors_pixels():
    from calora.data.transforms import hflip

    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    assert torch.equal(hflip(image)[0, 0], torch.tensor([3.0, 2.0, 1.0, 0.0]))
    assert torch.equal(hflip(hflip(image)), image)
