"""Determinism, pairing and gradient hygiene of the measurement loop.

The project's conclusions rest on comparisons — between timesteps, between
concept axes, between training steps. Those are only meaningful if the loop is
reproducible and if the draws really are shared where they are claimed to be.
"""

from __future__ import annotations

import pytest
import torch

from calora.ca.measure import measure_concept_awareness, prepare_samples
from calora.models.toy import ToyAdapter

AXES = ["style", "viewpoint"]


def _measure(adapter, table, samples, timestep=81, **kwargs):
    return measure_concept_awareness(
        adapter, table, samples, timestep=timestep, axes=AXES, seed=0, **kwargs
    )


def test_repeated_runs_are_bit_identical(adapter, table, dataset):
    samples = prepare_samples(adapter, dataset, axes=AXES)
    first = _measure(adapter, table, samples)
    second = _measure(adapter, table, samples)

    for axis in AXES:
        assert torch.equal(first.ca(axis), second.ca(axis))
    assert torch.equal(first.similarity(*AXES), second.similarity(*AXES))


def test_seed_changes_the_noise_and_hence_the_result(adapter, table, dataset):
    samples = prepare_samples(adapter, dataset, axes=AXES)
    a = measure_concept_awareness(adapter, table, samples, timestep=81, axes=AXES, seed=0)
    b = measure_concept_awareness(adapter, table, samples, timestep=81, axes=AXES, seed=1)
    assert not torch.equal(a.ca("style"), b.ca("style"))


def test_pairing_across_timesteps_reuses_the_same_noise(adapter, table, dataset):
    """With pairing on, the noise depends on the sample index only."""
    samples = prepare_samples(adapter, dataset, axes=AXES)

    paired_a = _measure(adapter, table, samples, timestep=1, paired_across_timesteps=True)
    paired_b = _measure(adapter, table, samples, timestep=1, paired_across_timesteps=False)
    # At a fixed t both are deterministic, but the unpaired stream folds t in,
    # so the two must differ.
    assert not torch.equal(paired_a.ca("style"), paired_b.ca("style"))

    # ...and the unpaired variant redraws when t changes, the paired one does not.
    assert not torch.equal(
        _measure(adapter, table, samples, timestep=201, paired_across_timesteps=False).ca("style"),
        paired_b.ca("style"),
    )


def test_axes_share_one_sample_draw(adapter, table, dataset):
    """Both axes must see the same x_t, or their cosine is meaningless."""
    samples = prepare_samples(adapter, dataset, axes=AXES)
    joint = _measure(adapter, table, samples)
    style_only = measure_concept_awareness(
        adapter, table, samples, timestep=81, axes=["style"], seed=0
    )
    assert torch.allclose(joint.ca("style"), style_only.ca("style"))


def test_measurement_leaves_no_gradients_or_flags_behind(adapter, table, dataset):
    before = {name: p.requires_grad for name, p in adapter.unet.named_parameters()}
    samples = prepare_samples(adapter, dataset, axes=AXES)
    _measure(adapter, table, samples)

    after = {name: p.requires_grad for name, p in adapter.unet.named_parameters()}
    assert after == before
    assert all(p.grad is None for p in adapter.unet.parameters())
    assert not adapter.unet.training


def test_out_of_range_timestep_is_rejected(adapter, table, dataset):
    samples = prepare_samples(adapter, dataset, axes=AXES)
    with pytest.raises(ValueError, match="outside"):
        _measure(adapter, table, samples, timestep=5000)


def test_prepare_requires_every_axis(adapter, dataset):
    with pytest.raises(KeyError, match="augmented prompts"):
        prepare_samples(adapter, dataset, axes=["style", "nonexistent"])


def test_ca_is_positive_and_finite_on_the_toy_model(adapter, table, dataset):
    samples = prepare_samples(adapter, dataset, axes=AXES)
    result = _measure(adapter, table, samples)
    for axis in AXES:
        ca = result.ca(axis)
        assert torch.isfinite(ca).all()
        assert (ca > 0).all(), "every toy projection should carry some concept gradient"
    assert result.valid_count.min() == len(samples)


def test_direction_accumulation_can_be_disabled(adapter, table, dataset):
    samples = prepare_samples(adapter, dataset, axes=AXES)
    lean = _measure(adapter, table, samples, accumulate_direction=False)
    assert lean.cos_of_means == {}
    assert lean.per_axis["style"].direction_mean is None
    # CA itself must be unaffected by the memory option.
    full = _measure(adapter, table, samples)
    assert torch.allclose(lean.ca("style"), full.ca("style"))


def test_multiple_augmentations_multiply_the_contributions(adapter, table, augmenter):
    from calora.data.dummy import DummyConceptDataset

    dataset = DummyConceptDataset(augmenter, num_items=4, resolution=64, augs_per_sample=3, seed=0)
    samples = prepare_samples(adapter, dataset, axes=AXES)
    assert samples[0].augs_per_sample == 3
    assert len(set(samples[0].augmented["style"])) == 3, "slots must be distinct prompts"

    result = _measure(adapter, table, samples)
    assert result.num_items == 4
    assert result.num_samples == 12
    assert result.augs_per_sample == 3
    assert result.valid_count.min() == 12
    for axis in AXES:
        assert torch.isfinite(result.ca(axis)).all()


def test_mismatched_augmentation_slots_are_rejected(adapter, dataset):
    items = [dict(dataset[i]) for i in range(len(dataset))]
    items[1]["augmented"] = {axis: list(items[1]["augmented"][axis]) * 2 for axis in AXES}
    with pytest.raises(ValueError, match="differing numbers of augmentation slots"):
        prepare_samples(adapter, items, axes=AXES)


def test_a_bare_string_augmentation_is_read_as_one_slot(adapter, dataset):
    """The pre-multi-aug item contract stays readable."""
    items = [dict(dataset[i]) for i in range(2)]
    for item in items:
        item["augmented"] = {axis: item["augmented"][axis][0] for axis in AXES}
    samples = prepare_samples(adapter, items, axes=AXES)
    assert samples[0].augs_per_sample == 1


def test_split_half_partitions_by_sample_and_merges_back(adapter, table, dataset):
    samples = prepare_samples(adapter, dataset, axes=AXES)
    full = _measure(adapter, table, samples)
    split = _measure(adapter, table, samples, split_half=True)

    assert full.halves is None
    assert split.halves is not None
    # The halves are a partition of the samples, by index parity.
    assert [half.num_items for half in split.halves] == [2, 2]
    assert sum(half.num_items for half in split.halves) == full.num_items
    # ...and merging them reproduces the undivided measurement.
    for axis in AXES:
        assert torch.allclose(split.ca(axis), full.ca(axis), rtol=1e-10, atol=1e-7)
    assert torch.allclose(
        split.similarity(*AXES), full.similarity(*AXES), rtol=1e-10, atol=1e-6
    )


def test_split_half_halves_differ_from_each_other(adapter, table, dataset):
    """Otherwise the agreement statistic would be measuring nothing."""
    samples = prepare_samples(adapter, dataset, axes=AXES)
    split = _measure(adapter, table, samples, split_half=True)
    assert split.halves is not None
    assert not torch.allclose(split.halves[0].ca("style"), split.halves[1].ca("style"))


def test_batching_changes_the_estimand(adapter, table, augmenter):
    """Documented behaviour: batch_size > 1 is a different statistic."""
    from calora.data.dummy import DummyConceptDataset

    dataset = DummyConceptDataset(augmenter, num_items=4, resolution=64, seed=0)
    samples = prepare_samples(adapter, dataset, axes=AXES)
    per_sample = _measure(adapter, table, samples, batch_size=1)
    batched = _measure(adapter, table, samples, batch_size=4)

    assert per_sample.num_samples == 4
    assert batched.num_samples == 1
    assert not torch.allclose(per_sample.ca("style"), batched.ca("style"))
