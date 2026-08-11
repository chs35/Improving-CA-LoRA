"""Accumulator arithmetic, driven by hand-built gradients."""

from __future__ import annotations

import math

import pytest
import torch

from calora.ca.stats import (
    ConceptAwarenessAccumulator,
    load_measurement,
    save_measurement,
)
from calora.models.projections import ProjectionSpec, ProjectionTable

AXES = ["style", "viewpoint"]


def _table(numel: int = 4, count: int = 2) -> ProjectionTable:
    return ProjectionTable(
        [
            ProjectionSpec(
                name=f"attn1.to_q_{i}",
                kind="q",
                attn_index=1,
                block="down",
                block_index=0,
                shape=(numel,),
                numel=numel,
            )
            for i in range(count)
        ]
    )


def _grads(*vectors: list[float]) -> dict[str, torch.Tensor]:
    return {
        f"attn1.to_q_{i}": torch.tensor(v, dtype=torch.float32) for i, v in enumerate(vectors)
    }


def test_ca_is_the_ratio_of_norms():
    table = _table()
    acc = ConceptAwarenessAccumulator(table, AXES)
    with acc.sample() as scope:
        scope.set_denominator(_grads([2.0, 0, 0, 0], [1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0], [4.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 3.0, 0, 0], [0, 0, 0, 0]))

    result = acc.finalize(81)
    assert result.ca("style").tolist() == pytest.approx([0.5, 4.0])
    assert result.ca("viewpoint").tolist() == pytest.approx([1.5, 0.0])


def test_orthogonal_axes_give_zero_cosine():
    table = _table(count=1)
    acc = ConceptAwarenessAccumulator(table, AXES)
    with acc.sample() as scope:
        scope.set_denominator(_grads([1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0]))

    result = acc.finalize(81)
    assert float(result.similarity(*AXES, source="cos_of_means")[0]) == pytest.approx(0.0)
    assert float(result.similarity(*AXES, source="mean_of_cos")[0]) == pytest.approx(0.0)


def test_opposed_axes_give_negative_one():
    table = _table(count=1)
    acc = ConceptAwarenessAccumulator(table, AXES)
    with acc.sample() as scope:
        scope.set_denominator(_grads([1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 2.0, 0, 0]))
        scope.add_axis("viewpoint", _grads([-3.0, -6.0, 0, 0]))

    result = acc.finalize(81)
    assert float(result.similarity(*AXES, source="mean_of_cos")[0]) == pytest.approx(-1.0, abs=1e-6)


def test_mean_of_cos_and_cos_of_means_diverge_when_directions_cancel():
    """Two samples pulling opposite ways: the mean vector vanishes, per-sample agreement does not."""
    table = _table(count=1)
    acc = ConceptAwarenessAccumulator(table, AXES)
    for sign in (1.0, -1.0):
        with acc.sample() as scope:
            scope.set_denominator(_grads([1.0, 0, 0, 0]))
            scope.add_axis("style", _grads([sign, 0, 0, 0]))
            scope.add_axis("viewpoint", _grads([sign, 0, 0, 0]))

    result = acc.finalize(81)
    # Per sample the axes agree perfectly...
    assert float(result.similarity(*AXES, source="mean_of_cos")[0]) == pytest.approx(1.0)
    # ...but both mean vectors are ~0, so cos-of-means is undefined and reported as 0.
    assert abs(float(result.similarity(*AXES, source="cos_of_means")[0])) < 1e-6


def test_zero_denominator_is_excluded_not_divided_by():
    table = _table(count=2)
    acc = ConceptAwarenessAccumulator(table, AXES)
    # Projection 1 has a zero L_Diff gradient in the first sample only.
    with acc.sample() as scope:
        scope.set_denominator(_grads([1.0, 0, 0, 0], [0.0, 0, 0, 0]))
        scope.add_axis("style", _grads([2.0, 0, 0, 0], [5.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([2.0, 0, 0, 0], [5.0, 0, 0, 0]))
    with acc.sample() as scope:
        scope.set_denominator(_grads([1.0, 0, 0, 0], [2.0, 0, 0, 0]))
        scope.add_axis("style", _grads([4.0, 0, 0, 0], [6.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([4.0, 0, 0, 0], [6.0, 0, 0, 0]))

    result = acc.finalize(81)
    assert result.valid_count.tolist() == [2.0, 1.0]
    assert float(result.ca("style")[0]) == pytest.approx(3.0)  # (2/1 + 4/1) / 2
    assert float(result.ca("style")[1]) == pytest.approx(3.0)  # only the valid sample: 6/2
    assert math.isfinite(float(result.ca("style")[1]))


def test_missing_gradients_are_treated_as_absent():
    table = _table(count=2)
    acc = ConceptAwarenessAccumulator(table, AXES)
    with acc.sample() as scope:
        scope.set_denominator({"attn1.to_q_0": torch.tensor([1.0, 0, 0, 0])})
        scope.add_axis("style", {"attn1.to_q_0": torch.tensor([2.0, 0, 0, 0])})
        scope.add_axis("viewpoint", {"attn1.to_q_0": torch.tensor([2.0, 0, 0, 0])})

    result = acc.finalize(81)
    assert float(result.ca("style")[0]) == pytest.approx(2.0)
    assert float(result.ca("style")[1]) == 0.0
    assert result.valid_count.tolist() == [1.0, 0.0]


def test_an_aborted_sample_leaves_the_accumulator_completely_untouched():
    """``add_axis`` must be all-or-nothing across axes.

    Each axis's ``add_axis`` follows its own backward pass, so the second one can
    fail (OOM, NaN guard, interrupt) after the first has already been recorded.
    If any part of the first axis had landed eagerly, that axis would be averaged
    over one more sample than the other and every cross-axis similarity would be
    computed from mismatched sample sets.
    """
    table = _table(count=2)
    acc = ConceptAwarenessAccumulator(table, AXES)
    with acc.sample() as scope:
        scope.set_denominator(_grads([1.0, 0, 0, 0], [1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 1.0, 0, 0], [0, 1.0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0], [1.0, 0, 0, 0]))
    before = acc.finalize(81)

    with pytest.raises(RuntimeError, match="second backward"):
        with acc.sample() as scope:
            scope.set_denominator(_grads([2.0, 0, 0, 0], [2.0, 0, 0, 0]))
            scope.add_axis("style", _grads([9.0, 0, 0, 0], [9.0, 0, 0, 0]))
            raise RuntimeError("second backward blew up")

    after = acc.finalize(81)
    assert after.num_samples == before.num_samples
    assert torch.equal(after.valid_count, before.valid_count)
    assert torch.equal(after.gd_norm_mean, before.gd_norm_mean)
    for axis in AXES:
        assert torch.equal(after.ca(axis), before.ca(axis))
        assert torch.equal(after.per_axis[axis].ca_std, before.per_axis[axis].ca_std)
        assert torch.equal(after.per_axis[axis].gc_norm_mean, before.per_axis[axis].gc_norm_mean)
        directions_after = after.per_axis[axis].direction_mean
        directions_before = before.per_axis[axis].direction_mean
        assert directions_after is not None and directions_before is not None
        for name in table.names:
            assert torch.equal(directions_after[name], directions_before[name])
    for source in ("cos_of_means", "mean_of_cos"):
        assert torch.equal(
            after.similarity(*AXES, source=source), before.similarity(*AXES, source=source)
        )


def test_commit_requires_every_axis():
    acc = ConceptAwarenessAccumulator(_table(count=1), AXES)
    scope = acc.sample()
    scope.set_denominator(_grads([1.0, 0, 0, 0]))
    scope.add_axis("style", _grads([1.0, 0, 0, 0]))
    with pytest.raises(RuntimeError, match="missing concept axes"):
        scope.commit()


def test_add_axis_before_denominator_is_rejected():
    acc = ConceptAwarenessAccumulator(_table(count=1), AXES)
    scope = acc.sample()
    with pytest.raises(RuntimeError, match="set_denominator"):
        scope.add_axis("style", _grads([1.0, 0, 0, 0]))


def test_more_slots_than_configured_is_rejected():
    """augs_per_sample is the contract; a stray extra add_axis is a bug, not a slot."""
    acc = ConceptAwarenessAccumulator(_table(count=1), AXES)  # augs_per_sample=1
    scope = acc.sample()
    scope.set_denominator(_grads([1.0, 0, 0, 0]))
    scope.add_axis("style", _grads([1.0, 0, 0, 0]))
    with pytest.raises(RuntimeError, match="already added"):
        scope.add_axis("style", _grads([1.0, 0, 0, 0]))


def test_axes_must_contribute_the_same_number_of_slots():
    acc = ConceptAwarenessAccumulator(_table(count=1), AXES, augs_per_sample=2)
    scope = acc.sample()
    scope.set_denominator(_grads([1.0, 0, 0, 0]))
    scope.add_axis("style", _grads([1.0, 0, 0, 0]))
    scope.add_axis("style", _grads([1.0, 0, 0, 0]))
    scope.add_axis("viewpoint", _grads([1.0, 0, 0, 0]))
    with pytest.raises(RuntimeError, match="different numbers of augmentation slots"):
        scope.commit()


def test_rejects_a_nonpositive_augs_per_sample():
    with pytest.raises(ValueError, match="augs_per_sample"):
        ConceptAwarenessAccumulator(_table(count=1), AXES, augs_per_sample=0)


# -- multiple augmentation slots per sample ------------------------------------------------


def test_ca_averages_over_every_slot_not_just_the_first():
    """Each (sample, slot) is its own ratio sample; the shared denominator is not re-counted."""
    acc = ConceptAwarenessAccumulator(_table(count=1), AXES, augs_per_sample=2)
    with acc.sample() as scope:
        scope.set_denominator(_grads([2.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0]))  # ratio 0.5
        scope.add_axis("style", _grads([3.0, 0, 0, 0]))  # ratio 1.5
        scope.add_axis("viewpoint", _grads([0, 2.0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 6.0, 0, 0]))

    result = acc.finalize(81)
    assert result.num_items == 1
    assert result.num_samples == 2  # contributions, not images
    assert result.augs_per_sample == 2
    assert float(result.ca("style")[0]) == pytest.approx(1.0)  # (0.5 + 1.5) / 2
    assert float(result.ca("viewpoint")[0]) == pytest.approx(2.0)  # (1 + 3) / 2
    assert result.valid_count.tolist() == [2.0]
    # gd_norm is a per-image quantity and must not be counted once per slot.
    assert float(result.gd_norm_mean[0]) == pytest.approx(2.0)


def test_direction_sums_every_slot():
    acc = ConceptAwarenessAccumulator(_table(count=1), AXES, augs_per_sample=2)
    with acc.sample() as scope:
        scope.set_denominator(_grads([2.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([3.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0]))

    direction = acc.finalize(81).per_axis["style"].direction_mean
    assert direction is not None
    # (1/2 + 3/2) summed, divided by the two valid contributions.
    assert torch.allclose(direction["attn1.to_q_0"], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_paired_cosine_matches_slots_position_by_position():
    """Slot j of one axis pairs with slot j of the other, not every slot with every slot."""
    acc = ConceptAwarenessAccumulator(_table(count=1), AXES, augs_per_sample=2)
    with acc.sample() as scope:
        scope.set_denominator(_grads([1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([0, 1.0, 0, 0]))
        scope.add_axis("viewpoint", _grads([1.0, 0, 0, 0]))  # slot 0: cos = 1
        scope.add_axis("viewpoint", _grads([-1.0, 0, 0, 0]))  # slot 1: cos = 0

    result = acc.finalize(81)
    # Cross-pairing slot 0 with slot 1 would give (1 + 0 - 1 + 0) / 4 = 0 instead.
    assert float(result.similarity(*AXES, source="mean_of_cos")[0]) == pytest.approx(0.5)


def test_memory_estimate_grows_with_augs_per_sample():
    table = _table(numel=100, count=3)
    one = ConceptAwarenessAccumulator(table, AXES, augs_per_sample=1).estimate_memory_bytes()
    four = ConceptAwarenessAccumulator(table, AXES, augs_per_sample=4).estimate_memory_bytes()

    # Only the unit vectors held for the pairwise cosines scale with m; the
    # persistent direction sums and the in-place slot accumulation do not.
    assert four["persistent_accumulators"] == one["persistent_accumulators"]
    assert four["transient_per_sample"] == one["transient_per_sample"] + 3 * 300 * 4 * 2


# -- within-axis direction concentration -------------------------------------------------


def test_concentration_is_one_when_every_slot_pushes_the_same_way():
    """``||E[ghat]|| / E[||ghat||]`` = 1 exactly when the slots are collinear.

    The two style slots differ in magnitude but not direction, which is the
    whole point: concentration is about agreement, not about size.
    """
    acc = ConceptAwarenessAccumulator(_table(count=2), AXES, augs_per_sample=2)
    with acc.sample() as scope:
        scope.set_denominator(_grads([2.0, 0, 0, 0], [1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0], [0, 0, 0, 0]))  # ghat = +0.5
        scope.add_axis("style", _grads([3.0, 0, 0, 0], [0, 0, 0, 0]))  # ghat = +1.5
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0], [0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 2.0, 0, 0], [0, 0, 0, 0]))

    concentration = acc.finalize(81).direction_concentration("style")
    assert concentration.dtype == torch.float64
    assert float(concentration[0]) == pytest.approx(1.0)
    # Projection 1 never saw a concept gradient, so there is no direction to
    # concentrate: 0, not a ratio of two numerical zeros.
    assert float(concentration[1]) == 0.0


def test_opposing_slots_drag_the_concentration_below_one():
    acc = ConceptAwarenessAccumulator(_table(count=1), AXES, augs_per_sample=2)
    with acc.sample() as scope:
        scope.set_denominator(_grads([2.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0]))  # ghat = +0.5
        scope.add_axis("style", _grads([-3.0, 0, 0, 0]))  # ghat = -1.5
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 3.0, 0, 0]))

    result = acc.finalize(81)
    # ||(0.5 - 1.5) / 2|| / ((0.5 + 1.5) / 2) = 0.5 / 1.0
    assert float(result.direction_concentration("style")[0]) == pytest.approx(0.5)
    assert float(result.direction_concentration("viewpoint")[0]) == pytest.approx(1.0)


def test_concentration_falls_when_two_samples_disagree():
    """Cancellation across samples reads the same way as cancellation across slots."""
    acc = ConceptAwarenessAccumulator(_table(count=1), AXES)
    for sign in (1.0, -1.0):
        with acc.sample() as scope:
            scope.set_denominator(_grads([1.0, 0, 0, 0]))
            scope.add_axis("style", _grads([sign, 0, 0, 0]))
            scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0]))

    result = acc.finalize(81)
    # The mean vector vanishes while CA stays 1 — exactly the situation that
    # makes cos_of_means meaningless (see the sibling test on that pair).
    assert float(result.direction_concentration("style")[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(result.direction_concentration("viewpoint")[0]) == pytest.approx(1.0)


def test_summary_carries_the_mean_concentration_per_axis():
    acc = ConceptAwarenessAccumulator(_table(count=2), AXES)
    with acc.sample() as scope:
        scope.set_denominator(_grads([2.0, 0, 0, 0], [1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0], [2.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0], [0, 3.0, 0, 0]))

    summary = acc.finalize(81).summary()
    assert summary["direction_concentration_mean"] == {
        "style": pytest.approx(1.0),
        "viewpoint": pytest.approx(1.0),
    }


def test_concentration_requires_direction_accumulation():
    acc = ConceptAwarenessAccumulator(_table(count=2), AXES, accumulate_direction=False)
    with acc.sample() as scope:
        scope.set_denominator(_grads([1.0, 0, 0, 0], [1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0], [2.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0], [0, 3.0, 0, 0]))
    result = acc.finalize(81)

    with pytest.raises(ValueError, match="accumulate_direction"):
        result.direction_concentration("style")
    with pytest.raises(KeyError, match="measured axes"):
        result.direction_concentration("nonexistent")
    # Nothing to report, so the key is absent rather than present and empty.
    assert "direction_concentration_mean" not in result.summary()


def test_concentration_survives_a_save_load_round_trip(tmp_path):
    """It is derived, not serialised — but only if the vectors were kept."""
    acc = ConceptAwarenessAccumulator(_table(count=1), AXES, augs_per_sample=2)
    with acc.sample() as scope:
        scope.set_denominator(_grads([2.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([-3.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0]))
    result = acc.finalize(81)

    path = tmp_path / "concentration.safetensors"
    save_measurement(result, path, save_direction_vectors=True)
    assert torch.allclose(
        load_measurement(path).direction_concentration("style"),
        result.direction_concentration("style"),
    )

    lean = tmp_path / "no_vectors.safetensors"
    save_measurement(result, lean)
    with pytest.raises(ValueError, match="save_direction_vectors"):
        load_measurement(lean).direction_concentration("style")


# -- merging -------------------------------------------------------------------------------


def _feed(acc: ConceptAwarenessAccumulator, sample_index: int, slots: int = 1) -> None:
    """A reproducible pseudo-sample, distinct per index and slot."""
    scale = 1.0 + sample_index
    with acc.sample() as scope:
        scope.set_denominator(_grads([scale, 0.5, 0, 0], [0.25 * scale, 0, 1.0, 0]))
        for slot in range(slots):
            offset = 1.0 + 0.5 * slot
            scope.add_axis("style", _grads([offset, scale, 0, 0], [0, offset, scale, 0]))
        for slot in range(slots):
            offset = 1.0 + 0.5 * slot
            scope.add_axis("viewpoint", _grads([-scale, offset, 0, 0], [offset, 0, -scale, 0]))


@pytest.mark.parametrize("slots", [1, 3])
def test_merging_two_halves_reproduces_the_single_accumulator(slots):
    """The invariant split-half rests on: every buffer is a plain sum over samples."""
    table = _table(count=2)
    whole = ConceptAwarenessAccumulator(table, AXES, augs_per_sample=slots)
    halves = [
        ConceptAwarenessAccumulator(table, AXES, augs_per_sample=slots) for _ in range(2)
    ]
    for index in range(6):
        _feed(whole, index, slots)
        _feed(halves[index % 2], index, slots)

    expected = whole.finalize(81)
    # finalize() must not consume state: the halves are reported *and* merged.
    half_results = [half.finalize(81) for half in halves]
    merged = halves[0].merge_(halves[1]).finalize(81)

    assert merged.num_items == expected.num_items == 6
    assert merged.num_samples == expected.num_samples == 6 * slots
    assert sum(h.num_items for h in half_results) == 6
    assert torch.allclose(merged.valid_count, expected.valid_count, rtol=1e-10)
    assert torch.allclose(merged.gd_norm_mean, expected.gd_norm_mean, rtol=1e-10)
    for axis in AXES:
        assert torch.allclose(merged.ca(axis), expected.ca(axis), rtol=1e-10)
        assert torch.allclose(
            merged.per_axis[axis].ca_std, expected.per_axis[axis].ca_std, rtol=1e-10, atol=1e-7
        )
        assert torch.allclose(
            merged.per_axis[axis].gc_norm_mean, expected.per_axis[axis].gc_norm_mean, rtol=1e-10
        )
        directions = merged.per_axis[axis].direction_mean
        reference = expected.per_axis[axis].direction_mean
        assert directions is not None and reference is not None
        for name in table.names:
            # The direction buffers are deliberately fp32 — they are as large as
            # the attention parameters — so reordering the summation moves the
            # last few ulps. Everything checked above is fp64 and exact.
            assert torch.allclose(directions[name], reference[name], rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        merged.similarity(*AXES, source="mean_of_cos"),
        expected.similarity(*AXES, source="mean_of_cos"),
        rtol=1e-10,
        atol=1e-7,
    )
    assert torch.allclose(  # derived from the fp32 direction means
        merged.similarity(*AXES, source="cos_of_means"),
        expected.similarity(*AXES, source="cos_of_means"),
        rtol=1e-6,
        atol=1e-6,
    )


def test_merging_incompatible_accumulators_is_rejected():
    table = _table(count=2)
    base = ConceptAwarenessAccumulator(table, AXES)
    with pytest.raises(ValueError, match="itself"):
        base.merge_(base)
    with pytest.raises(ValueError, match="augs_per_sample"):
        base.merge_(ConceptAwarenessAccumulator(table, AXES, augs_per_sample=2))
    with pytest.raises(ValueError, match="names"):
        base.merge_(ConceptAwarenessAccumulator(_table(count=3), AXES))
    with pytest.raises(ValueError, match="axes"):
        base.merge_(ConceptAwarenessAccumulator(table, ["style"]))


def test_memory_estimate_scales_with_axes_and_options():
    table = _table(numel=100, count=3)
    full = ConceptAwarenessAccumulator(table, AXES).estimate_memory_bytes()
    lean = ConceptAwarenessAccumulator(
        table, AXES, accumulate_direction=False, compute_paired_cosine=False
    ).estimate_memory_bytes()

    assert full["attention_params"] == 300
    assert full["persistent_accumulators"] == 300 * 4 * 2
    assert lean["total"] == 0


def test_save_load_round_trip(tmp_path):
    table = _table(count=2)
    acc = ConceptAwarenessAccumulator(table, AXES)
    with acc.sample() as scope:
        scope.set_denominator(_grads([1.0, 0, 0, 0], [2.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 1.0, 0, 0], [1.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0], [1.0, 1.0, 0, 0]))
    result = acc.finalize(201)

    path = tmp_path / "t0201.safetensors"
    save_measurement(result, path)
    loaded = load_measurement(path)

    assert loaded.timestep == 201
    assert loaded.num_samples == 1
    assert loaded.names == result.names
    assert loaded.axes == result.axes
    assert torch.allclose(loaded.ca("style"), result.ca("style"))
    assert torch.allclose(loaded.similarity(*AXES), result.similarity(*AXES))
    assert loaded.table().names == table.names


def test_save_can_include_direction_vectors(tmp_path):
    table = _table(count=1)
    acc = ConceptAwarenessAccumulator(table, AXES)
    with acc.sample() as scope:
        scope.set_denominator(_grads([2.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([0, 1.0, 0, 0]))
    result = acc.finalize(1)

    path = tmp_path / "with_dirs.safetensors"
    save_measurement(result, path, save_direction_vectors=True)
    loaded = load_measurement(path)

    direction = loaded.per_axis["style"].direction_mean
    assert direction is not None
    assert torch.allclose(direction["attn1.to_q_0"], torch.tensor([0.5, 0.0, 0.0, 0.0]))


def test_similarity_lookup_is_order_insensitive_and_errors_clearly():
    table = _table(count=1)
    acc = ConceptAwarenessAccumulator(table, AXES)
    with acc.sample() as scope:
        scope.set_denominator(_grads([1.0, 0, 0, 0]))
        scope.add_axis("style", _grads([1.0, 0, 0, 0]))
        scope.add_axis("viewpoint", _grads([1.0, 0, 0, 0]))
    result = acc.finalize(1)

    assert torch.allclose(
        result.similarity("style", "viewpoint"), result.similarity("viewpoint", "style")
    )
    with pytest.raises(KeyError):
        result.similarity("style", "nonexistent")
    with pytest.raises(ValueError):
        result.similarity("style", "viewpoint", source="bogus")


def test_rejects_bad_configuration():
    with pytest.raises(ValueError, match="granularity"):
        ConceptAwarenessAccumulator(_table(), AXES, granularity="nope")
    with pytest.raises(ValueError, match="(?i)at least one"):
        ConceptAwarenessAccumulator(_table(), [])
    with pytest.raises(ValueError, match="Duplicate"):
        ConceptAwarenessAccumulator(_table(), ["style", "style"])
