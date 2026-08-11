"""Head-granularity units: slicing, measurement, selection and persistence.

The paper's unit is the theoretical attention head — a row-slice of
``to_q``/``to_k``/``to_v`` and a column-slice of ``to_out.0`` — not the fused
``nn.Linear`` this project measured first. These tests pin the properties that
distinguish the two, and check that the module path still behaves exactly as it
did (the granularity ablation depends on it).

The toy model is used with ``heads >= 2`` throughout: at ``H = 1`` a head unit
*is* the matrix and none of the slicing is exercised.
"""

from __future__ import annotations

import pytest
import torch

from calora.ca.measure import measure_concept_awareness, prepare_samples
from calora.ca.select import (
    DEFAULT_BUDGET_BASIS,
    aggregate_across_timesteps,
    rank_timesteps,
    select_all,
    select_random,
    select_topk,
    Selection,
)
from calora.ca.stats import load_measurement, save_measurement
from calora.config import SelectConfig
from calora.data.dummy import DummyConceptDataset
from calora.models.projections import (
    HEAD_SLICE_DIMS,
    ProjectionTable,
    attention_head_counts,
    enumerate_attention_heads,
    head_unit_name,
    parse_unit_name,
)
from calora.models.toy import ToyAdapter
from calora.train.lora import resolve_target_modules

AXES = ["style", "viewpoint"]


@pytest.fixture
def head_table(adapter: ToyAdapter) -> ProjectionTable:
    return ProjectionTable.from_model(adapter.unet, granularity="head")


def _measure(adapter, table, dataset, timestep: int, **kwargs):
    samples = prepare_samples(adapter, dataset, axes=AXES)
    return measure_concept_awareness(
        adapter, table, samples, timestep=timestep, axes=AXES, seed=0, **kwargs
    )


# --------------------------------------------------------------------------------------
# Unit definition
# --------------------------------------------------------------------------------------


def test_toy_model_has_at_least_two_heads(adapter: ToyAdapter):
    """The premise of every test below: H=1 would exercise no slicing at all."""
    counts = attention_head_counts(adapter.unet)
    assert counts, "no attention module exposed a head count"
    assert min(counts.values()) >= 2


def test_head_counts_come_from_the_module_not_a_constant():
    """``H`` is read off ``Attention.heads``; changing it changes the table."""
    for heads in (2, 4):
        adapter = ToyAdapter(seed=0, base_channels=32, heads=heads)
        table = ProjectionTable.from_model(adapter.unet, granularity="head")
        assert set(attention_head_counts(adapter.unet).values()) == {heads}
        assert len(table) == 40 * heads
        assert set(table.head_counts.values()) == {heads}


def test_heads_partition_the_matrix(table: ProjectionTable, head_table: ProjectionTable):
    """U = P * H, and the head units of a matrix sum to exactly its parameters."""
    counts = head_table.head_counts
    assert len(head_table) == sum(counts[name] for name in table.names)
    assert head_table.total_params == table.total_params
    assert head_table.module_names == table.names

    per_module: dict[str, int] = {}
    for spec in head_table:
        per_module[spec.module_name] = per_module.get(spec.module_name, 0) + spec.numel
    for spec in table:
        assert per_module[spec.name] == spec.numel
        assert per_module[spec.name] == spec.numel


def test_slice_direction_is_row_for_qkv_and_col_for_out(head_table: ProjectionTable):
    """``to_out.0`` reads from ``inner_dim``, so its heads own columns, not rows."""
    seen = set()
    for spec in head_table:
        assert spec.slice_dim == HEAD_SLICE_DIMS[spec.kind]
        seen.add((spec.kind, spec.slice_dim))
    assert seen == {("q", "row"), ("k", "row"), ("v", "row"), ("out", "col")}


def test_slice_weight_takes_the_expected_band(adapter: ToyAdapter, head_table: ProjectionTable):
    params = dict(adapter.unet.named_parameters())
    for spec in head_table:
        weight = params[spec.weight_name]
        sliced = spec.slice_weight(weight)
        assert tuple(sliced.shape) == tuple(spec.shape)
        head, dim = spec.head_index, spec.shape[0 if spec.slice_dim == "row" else 1]
        expected = (
            weight[head * dim : (head + 1) * dim]
            if spec.slice_dim == "row"
            else weight[:, head * dim : (head + 1) * dim]
        )
        assert torch.equal(sliced, expected)


def test_cross_attention_kv_is_still_row_sliced(head_table: ProjectionTable):
    """to_k/to_v of attn2 take a wider input but still write inner_dim rows."""
    cross_kv = [s for s in head_table if s.is_cross_attention and s.kind in ("k", "v")]
    assert cross_kv
    assert all(s.slice_dim == "row" for s in cross_kv)


def test_slices_are_complete(adapter: ToyAdapter, head_table: ProjectionTable):
    r"""``sum_h ||g[h]||^2 == ||g||^2``.

    The identity that *does* hold. The one that does not — a parameter-weighted
    combination of head CAs reproducing the matrix CA — is the subject of
    :func:`test_head_ca_is_not_a_decomposition_of_module_ca`: CA is a mean of
    ratios, which is not linear in the slices.
    """
    generator = torch.Generator().manual_seed(0)
    params = dict(adapter.unet.named_parameters())
    for spec in ProjectionTable.from_model(adapter.unet):
        fake_grad = torch.randn(params[spec.weight_name].shape, generator=generator)
        heads = [s for s in head_table if s.module_name == spec.name]
        assert len(heads) == heads[0].num_heads >= 2
        total = sum(float(h.slice_weight(fake_grad).pow(2).sum()) for h in heads)
        assert total == pytest.approx(float(fake_grad.pow(2).sum()), rel=1e-6)


def test_unit_names_round_trip():
    name = head_unit_name("down_blocks.0...attn1.to_out.0", 13)
    assert name.endswith("@h13")
    assert parse_unit_name(name) == ("down_blocks.0...attn1.to_out.0", 13)
    assert parse_unit_name("down_blocks.0...attn1.to_q") == ("down_blocks.0...attn1.to_q", None)
    with pytest.raises(ValueError, match="Malformed head unit name"):
        parse_unit_name("attn1.to_q@hx")


def test_head_names_never_collide_with_module_names(
    table: ProjectionTable, head_table: ProjectionTable
):
    """``@`` cannot occur in a module path, so peft can never match a head name."""
    assert not set(table.names) & set(head_table.names)
    assert all("@" not in name for name in table.names)


def test_mixed_granularity_table_is_rejected(table: ProjectionTable, head_table: ProjectionTable):
    with pytest.raises(ValueError, match="mixes module-granularity and head-granularity"):
        ProjectionTable([table.specs[0], head_table.specs[0]])


def test_records_round_trip_preserves_head_metadata(head_table: ProjectionTable):
    restored = ProjectionTable.from_records(head_table.to_records())
    assert restored.granularity == "head"
    assert restored.names == head_table.names
    assert [s.slice_dim for s in restored] == [s.slice_dim for s in head_table]
    assert [s.head_index for s in restored] == [s.head_index for s in head_table]
    assert [s.num_heads for s in restored] == [s.num_heads for s in head_table]
    assert [s.param_count for s in restored] == [s.param_count for s in head_table]


def test_missing_head_count_is_an_error():
    """A model with attention-shaped names but no ``heads`` attribute must fail loudly."""
    from torch import nn

    class _Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(4, 4, bias=False)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = _Attention()

    with pytest.raises(ValueError, match="No head count"):
        enumerate_attention_heads(_Model())


def test_indivisible_head_dim_is_an_error():
    from torch import nn

    class _Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = 3
            self.to_q = nn.Linear(4, 4, bias=False)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = _Attention()

    with pytest.raises(ValueError, match="not divisible by heads"):
        enumerate_attention_heads(_Model())


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


def test_head_measurement_produces_one_statistic_per_unit(
    adapter: ToyAdapter, head_table: ProjectionTable, augmenter
):
    dataset = DummyConceptDataset(augmenter, num_items=3, resolution=64, seed=0)
    result = _measure(adapter, head_table, dataset, 81)
    assert result.unit_granularity == "head"
    assert list(result.names) == head_table.names
    assert result.num_projections == len(head_table)
    for axis in AXES:
        assert result.ca(axis).shape == (len(head_table),)
    # The invariant that survives the refactor: both similarities always logged.
    assert result.similarity(*AXES, source="cos_of_means").shape == (len(head_table),)
    assert result.similarity(*AXES, source="mean_of_cos").shape == (len(head_table),)
    assert result.direction_concentration("style").shape == (len(head_table),)


def test_head_ca_is_the_ratio_of_head_norms(
    adapter: ToyAdapter, head_table: ProjectionTable, augmenter
):
    r"""``CA_head = E[||g_c[h]|| / ||g_d[h]||]`` — computed at head level.

    Checked against a hand-rolled single-sample measurement: one L_Diff backward
    and one L_Concept backward, sliced by hand. With one sample the expectation
    is the ratio itself, so any mis-slicing shows up immediately.
    """
    dataset = DummyConceptDataset(augmenter, num_items=1, resolution=64, seed=0)
    samples = prepare_samples(adapter, dataset, axes=AXES)
    result = measure_concept_awareness(
        adapter, head_table, samples, timestep=81, axes=AXES, seed=0
    )

    module_table = ProjectionTable.from_model(adapter.unet)
    module_result = measure_concept_awareness(
        adapter, module_table, samples, timestep=81, axes=AXES, seed=0
    )

    # With a single sample and slot, the head ratio and the matrix ratio are both
    # exact, so the norm identity below is a statement about the same gradients.
    ca_head = result.ca("style").to(torch.float64)
    gc_head = result.per_axis["style"].gc_norm_mean.to(torch.float64)
    gd_head = result.gd_norm_mean.to(torch.float64)
    assert torch.allclose(ca_head, gc_head / gd_head, atol=1e-5)

    # sum_h ||g[h]||^2 == ||g||^2 for both numerator and denominator.
    for values, module_values in (
        (gc_head, module_result.per_axis["style"].gc_norm_mean.to(torch.float64)),
        (gd_head, module_result.gd_norm_mean.to(torch.float64)),
    ):
        per_module: dict[str, float] = {}
        for spec, value in zip(head_table.specs, values):
            per_module[spec.module_name] = per_module.get(spec.module_name, 0.0) + float(value) ** 2
        expected = torch.tensor(
            [per_module[name] for name in module_table.names], dtype=torch.float64
        )
        assert torch.allclose(expected.sqrt(), module_values, rtol=1e-4, atol=1e-6)


def test_head_ca_is_not_a_decomposition_of_module_ca(
    adapter: ToyAdapter, head_table: ProjectionTable, augmenter
):
    """No parameter-weighted combination of head CAs reproduces the matrix CA.

    CA is a mean of *ratios*; the slices enter through two separate norms, so the
    map from head CAs to matrix CA is non-linear. Stated as a test because the
    tempting shortcut — deriving head CA from the already-measured module CA — is
    exactly what forced a full re-measurement.
    """
    dataset = DummyConceptDataset(augmenter, num_items=4, resolution=64, seed=0)
    samples = prepare_samples(adapter, dataset, axes=AXES)
    head = measure_concept_awareness(
        adapter, head_table, samples, timestep=81, axes=AXES, seed=0
    )
    module_table = ProjectionTable.from_model(adapter.unet)
    module = measure_concept_awareness(
        adapter, module_table, samples, timestep=81, axes=AXES, seed=0
    )

    weighted: dict[str, float] = {}
    for spec, value in zip(head_table.specs, head.ca("style").tolist()):
        weighted[spec.module_name] = weighted.get(spec.module_name, 0.0) + value * spec.numel
    combined = torch.tensor(
        [weighted[s.name] / s.numel for s in module_table.specs], dtype=torch.float64
    )
    assert not torch.allclose(combined, module.ca("style").to(torch.float64), atol=1e-3)


def test_head_and_module_measurements_share_the_denominator_draws(
    adapter: ToyAdapter, head_table: ProjectionTable, augmenter
):
    """Granularity must not disturb the pairing: same samples, same noise, same t."""
    dataset = DummyConceptDataset(augmenter, num_items=3, resolution=64, seed=0)
    samples = prepare_samples(adapter, dataset, axes=AXES)
    module_table = ProjectionTable.from_model(adapter.unet)
    a = measure_concept_awareness(
        adapter, module_table, samples, timestep=81, axes=AXES, seed=0
    )
    b = measure_concept_awareness(
        adapter, module_table, samples, timestep=81, axes=AXES, seed=0
    )
    head = measure_concept_awareness(
        adapter, head_table, samples, timestep=81, axes=AXES, seed=0
    )
    assert torch.equal(a.gd_norm_mean, b.gd_norm_mean)
    assert a.num_samples == head.num_samples and a.num_items == head.num_items


def test_split_half_works_at_head_granularity(
    adapter: ToyAdapter, head_table: ProjectionTable, augmenter
):
    """Halves still split by sample and still merge to the full result."""
    dataset = DummyConceptDataset(augmenter, num_items=4, resolution=64, seed=0)
    result = _measure(adapter, head_table, dataset, 81, split_half=True)
    assert result.halves is not None
    for half in result.halves:
        assert half.unit_granularity == "head"
        assert list(half.names) == head_table.names
        assert half.num_items == 2
    assert result.num_items == sum(h.num_items for h in result.halves)
    assert result.num_samples == sum(h.num_samples for h in result.halves)


def test_paired_across_timesteps_holds_at_head_granularity(
    adapter: ToyAdapter, head_table: ProjectionTable, augmenter
):
    """Different t, same draws: the seed must not depend on the timestep."""
    dataset = DummyConceptDataset(augmenter, num_items=2, resolution=64, seed=0)
    samples = prepare_samples(adapter, dataset, axes=AXES)
    kwargs = dict(axes=AXES, seed=0, paired_across_timesteps=True)
    at_1 = measure_concept_awareness(adapter, head_table, samples, timestep=1, **kwargs)
    at_81 = measure_concept_awareness(adapter, head_table, samples, timestep=81, **kwargs)
    again = measure_concept_awareness(adapter, head_table, samples, timestep=1, **kwargs)
    assert torch.equal(at_1.gd_norm_mean, again.gd_norm_mean)
    assert not torch.equal(at_1.gd_norm_mean, at_81.gd_norm_mean)


def test_elementwise_ratio_still_works_at_head_granularity(
    adapter: ToyAdapter, head_table: ProjectionTable, augmenter
):
    """``measure.granularity`` (how the ratio is read) is orthogonal to the unit."""
    dataset = DummyConceptDataset(augmenter, num_items=2, resolution=64, seed=0)
    matrix = _measure(adapter, head_table, dataset, 81, granularity="matrix")
    elementwise = _measure(adapter, head_table, dataset, 81, granularity="elementwise")
    assert matrix.unit_granularity == elementwise.unit_granularity == "head"
    assert matrix.granularity == "matrix" and elementwise.granularity == "elementwise"
    assert not torch.allclose(matrix.ca("style"), elementwise.ca("style"))


# --------------------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------------------


def test_drift_tracks_head_units(
    adapter: ToyAdapter, head_table: ProjectionTable, augmenter, tmp_path
):
    """Jaccard/Spearman are set/rank operations over unit names, so the logic is
    granularity-agnostic — but the tracker must actually build a head table."""
    pytest.importorskip("peft")
    from calora.analysis.stability import build_stability_tracker, make_stability_callback
    from calora.config import LoraConfig_, TrainConfig
    from calora.train.lora import train_lora

    dataset = DummyConceptDataset(augmenter, num_items=4, resolution=64, seed=0)
    samples = prepare_samples(adapter, dataset, axes=AXES, num_samples=2)
    # Type A: every head of every matrix, so the arm is trainable today.
    trained = select_all(head_table)

    tracker = build_stability_tracker(
        adapter, samples, axes=AXES, timestep=81, fraction=0.1,
        budget_basis="unit_count", trained_selection=trained, granularity="head",
    )
    assert tracker.table.granularity == "head"
    assert len(tracker.baseline_selection) == 8
    assert all(parse_unit_name(n)[1] is not None for n in tracker.baseline_selection.names)

    callback = make_stability_callback(
        adapter, samples, tracker, axes=AXES, every=1, max_steps=1,
        fraction=0.1, budget_basis="unit_count",
    )
    train_lora(
        adapter,
        dataset,
        trained,
        TrainConfig(
            lora=LoraConfig_(rank=2, alpha=2), max_steps=1, batch_size=1,
            learning_rate=0.0, log_every=0, checkpoint_every=0,
        ),
        output_dir=tmp_path,
        callbacks=[callback],
    )
    assert len(tracker.snapshots) == 1
    # A zero learning rate cannot move the weights, so nothing may drift.
    assert tracker.snapshots[0].jaccard_vs_baseline == 1.0
    assert tracker.to_dict()["granularity"] == "head"
    assert tracker.to_dict()["num_units"] == len(head_table)


def test_drift_refuses_a_granularity_mismatch(
    adapter: ToyAdapter, table: ProjectionTable, augmenter
):
    """A module-granularity arm tracked with head units would compare name sets
    that never intersect, reporting total drift from step 0."""
    from calora.analysis.stability import build_stability_tracker

    dataset = DummyConceptDataset(augmenter, num_items=2, resolution=64, seed=0)
    samples = prepare_samples(adapter, dataset, axes=AXES, num_samples=2)
    module_arm = select_topk(
        torch.arange(len(table), dtype=torch.float64), table, fraction=0.1
    )
    with pytest.raises(ValueError, match="does not match the trained selection"):
        build_stability_tracker(
            adapter, samples, axes=AXES, timestep=81, fraction=0.1,
            trained_selection=module_arm, granularity="head",
        )


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def test_measurement_records_its_unit_granularity(
    adapter: ToyAdapter, head_table: ProjectionTable, augmenter, tmp_path
):
    dataset = DummyConceptDataset(augmenter, num_items=2, resolution=64, seed=0)
    result = _measure(adapter, head_table, dataset, 81)
    path = tmp_path / "t0081.safetensors"
    save_measurement(result, path)

    loaded = load_measurement(path, expect_granularity="head")
    assert loaded.unit_granularity == "head"
    assert list(loaded.names) == head_table.names
    assert loaded.table().granularity == "head"
    assert loaded.summary()["unit_granularity"] == "head"


def test_loading_the_wrong_granularity_is_refused(
    adapter: ToyAdapter, table: ProjectionTable, augmenter, tmp_path
):
    """Head and module files look identical; only the metadata separates them."""
    dataset = DummyConceptDataset(augmenter, num_items=2, resolution=64, seed=0)
    path = tmp_path / "t0081.safetensors"
    save_measurement(_measure(adapter, table, dataset, 81), path)

    assert load_measurement(path, expect_granularity="module").unit_granularity == "module"
    with pytest.raises(ValueError, match="unit granularity 'module' .* 'head' was expected"):
        load_measurement(path, expect_granularity="head")


def test_legacy_files_read_back_as_module_units(
    adapter: ToyAdapter, table: ProjectionTable, augmenter, tmp_path
):
    """Files written before the field existed are module measurements, and load as such."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    dataset = DummyConceptDataset(augmenter, num_items=2, resolution=64, seed=0)
    path = tmp_path / "t0081.safetensors"
    save_measurement(_measure(adapter, table, dataset, 81), path)

    with safe_open(str(path), framework="pt") as fh:
        metadata = dict(fh.metadata())
        tensors = {key: fh.get_tensor(key) for key in fh.keys()}
    metadata.pop("unit_granularity")
    legacy = tmp_path / "legacy.safetensors"
    save_file(tensors, str(legacy), metadata=metadata)

    assert load_measurement(legacy, expect_granularity="module").unit_granularity == "module"


# --------------------------------------------------------------------------------------
# Selection and scoring
# --------------------------------------------------------------------------------------


def test_selection_pipeline_runs_unchanged_on_a_head_table(
    adapter: ToyAdapter, head_table: ProjectionTable, augmenter
):
    """Aggregation, top-k and timestep ranking need only the new table injected."""
    dataset = DummyConceptDataset(augmenter, num_items=3, resolution=64, seed=0)
    samples = prepare_samples(adapter, dataset, axes=AXES)
    results = {
        t: measure_concept_awareness(
            adapter, head_table, samples, timestep=t, axes=AXES, seed=0, split_half=True
        )
        for t in (1, 81)
    }

    aggregated = aggregate_across_timesteps(results, axis="style")
    assert aggregated.shape == (len(head_table),)

    selection = select_topk(
        aggregated, head_table, fraction=0.1, budget_basis="unit_count"
    )
    assert selection.granularity == "head"
    assert selection.units_total == len(head_table)
    assert len(selection) == 8  # floor(0.1 * 80)
    assert all(parse_unit_name(name)[1] is not None for name in selection.names)

    ranked = rank_timesteps(
        results, head_table, axis_pair=(AXES[0], AXES[1]), fraction=0.1,
        budget_basis="unit_count",
    )
    assert {s.timestep for s in ranked} == {1, 81}
    assert all(s.num_selected == 8 for s in ranked)

    # Split halves rank on the same table with no special handling.
    halves = rank_timesteps(
        {t: r.halves[0] for t, r in results.items()},
        head_table,
        axis_pair=(AXES[0], AXES[1]),
        fraction=0.1,
        budget_basis="unit_count",
    )
    assert len(halves) == 2


def test_unit_count_budget_takes_the_floor(head_table: ProjectionTable):
    scores = torch.arange(len(head_table), dtype=torch.float64)
    for fraction, expected in ((0.02, 1), (0.1, 8), (0.5, 40)):
        selection = select_topk(
            scores, head_table, fraction=fraction, budget_basis="unit_count"
        )
        assert len(selection) == expected
    # Never empty, even when the floor would be zero.
    assert len(select_topk(scores, head_table, fraction=0.001, budget_basis="unit_count")) == 1


def test_unit_count_budget_ignores_unit_width(head_table: ProjectionTable):
    """The count basis buys k units whatever they cost; the parameter basis does not."""
    scores = torch.arange(len(head_table), dtype=torch.float64)
    counted = select_topk(scores, head_table, fraction=0.1, budget_basis="unit_count")
    parameterised = select_topk(scores, head_table, fraction=0.1, budget_basis="attn_params")
    assert len(counted) == 8
    assert counted.params_selected != parameterised.params_selected
    # Parameter statistics are reported under both bases (the report needs them).
    assert counted.params_selected > 0
    assert counted.to_dict()["unit_fraction_achieved"] == pytest.approx(0.1)


def test_selection_json_round_trips_head_granularity(head_table: ProjectionTable):
    scores = torch.arange(len(head_table), dtype=torch.float64)
    selection = select_topk(scores, head_table, fraction=0.1, budget_basis="unit_count")
    restored = Selection.from_dict(selection.to_dict(), head_table)
    assert restored.granularity == "head"
    assert restored.names == selection.names
    assert restored.indices == selection.indices
    assert restored.units_total == selection.units_total


def test_legacy_selection_json_loads_as_module(table: ProjectionTable):
    """No ``granularity`` key and no ``@h`` in the names -> a module selection."""
    restored = Selection.from_dict({"names": table.names[:3]}, table)
    assert restored.granularity == "module"
    assert restored.as_target_modules() == table.names[:3]


def test_random_and_all_baselines_carry_the_granularity(head_table: ProjectionTable):
    assert select_random(head_table, fraction=0.1, budget_basis="unit_count").granularity == "head"
    everything = select_all(head_table)
    assert everything.granularity == "head"
    assert len(everything) == len(head_table)


def test_breakdown_groups_head_units_by_kind_and_block(head_table: ProjectionTable):
    """Grouping keys come from the owning matrix, so the report reads the same."""
    everything = select_all(head_table)
    assert set(everything.breakdown["kind"]) == {"q", "k", "v", "out"}
    assert sum(everything.breakdown["kind"].values()) == len(head_table)


# --------------------------------------------------------------------------------------
# Training interface (type A / type B)
# --------------------------------------------------------------------------------------


def test_fully_selected_matrices_map_to_full_matrix_lora(
    adapter: ToyAdapter, head_table: ProjectionTable, table: ProjectionTable
):
    """All H heads chosen: the slice is the whole matrix, so every one is a target."""
    targets = resolve_target_modules(adapter.unet, select_all(head_table))
    assert targets == sorted(table.names)


def test_partially_selected_matrices_get_a_sliced_adapter(
    adapter: ToyAdapter, head_table: ProjectionTable
):
    """A partial matrix is trained as a slice, never widened to the whole matrix."""
    from calora.train.lora import plan_sliced_lora

    partial = Selection.from_dict(
        {"names": [head_table.names[0]], "granularity": "head"}, head_table
    )
    plans = plan_sliced_lora(adapter.unet, partial)
    assert len(plans) == 1
    assert plans[0].head_indices == (head_table[0].head_index,)
    assert not plans[0].is_full
    assert resolve_target_modules(adapter.unet, partial) == [head_table[0].module_name]


def test_peft_targets_still_refuse_a_partial_matrix(head_table: ProjectionTable):
    """``as_target_modules`` is the peft-only view: peft cannot slice, so a partial
    matrix must raise there rather than be silently widened."""
    partial = Selection.from_dict(
        {"names": [head_table.names[0]], "granularity": "head"}, head_table
    )
    counts = {head_table[0].module_name: head_table[0].num_heads}
    with pytest.raises(NotImplementedError, match="only partly selected"):
        partial.as_target_modules(counts)


def test_head_selection_needs_head_counts_to_resolve(head_table: ProjectionTable):
    selection = select_all(head_table)
    with pytest.raises(ValueError, match="needs head_counts"):
        selection.as_target_modules()


def test_head_groups_expose_the_selected_slices(head_table: ProjectionTable):
    names = [head_table.names[0], head_table.names[1], head_table.names[4]]
    selection = Selection.from_dict({"names": names, "granularity": "head"}, head_table)
    groups = selection.head_groups()
    assert sum(len(v) for v in groups.values()) == 3
    assert all(heads == sorted(heads) for heads in groups.values())
    assert selection.module_names() == list(dict.fromkeys(parse_unit_name(n)[0] for n in names))


def test_module_selection_has_no_head_groups(table: ProjectionTable):
    selection = select_topk(
        torch.arange(len(table), dtype=torch.float64), table, fraction=0.1
    )
    assert selection.head_groups() == {}
    assert selection.as_target_modules() == selection.names


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def test_config_defaults_to_head_units_and_a_count_budget():
    cfg = SelectConfig()
    assert cfg.granularity == "head"
    assert cfg.budget_basis == DEFAULT_BUDGET_BASIS["head"] == "unit_count"


def test_config_auto_budget_follows_granularity():
    assert SelectConfig(granularity="module").budget_basis == "attn_params"
    assert SelectConfig(granularity="head").budget_basis == "unit_count"
    # An explicit basis is never overridden, and resolution is idempotent.
    explicit = SelectConfig(granularity="head", budget_basis="attn_params")
    assert explicit.budget_basis == "attn_params"
    assert SelectConfig(**{**explicit.__dict__}).budget_basis == "attn_params"


def test_config_rejects_unknown_granularity_and_basis():
    with pytest.raises(ValueError, match="select.granularity"):
        SelectConfig(granularity="layer")
    with pytest.raises(ValueError, match="select.budget_basis"):
        SelectConfig(budget_basis="nonsense")
