"""Head-slice LoRA: the paper's type A / type B adapters.

The properties pinned here are the ones that make a head-granularity arm mean
what it claims:

* **equivalence** — selecting every head of a matrix must be numerically the same
  experiment as a standard full-matrix LoRA (forward *and* backward), otherwise
  the granularity ablation compares two implementations rather than two units;
* **locality** — a selected head's adapter must never touch another head's
  parameters, otherwise the 2% budget is fiction;
* **provenance** — the checkpoint has to say which heads it covers and on which
  axis, because nothing in the tensor shapes reveals either.

The toy model is used with ``heads >= 2`` throughout: at ``H = 1`` a slice is the
whole matrix and none of this is exercised.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from calora.analysis.stability import build_stability_tracker, make_stability_callback, merged_lora
from calora.ca.measure import prepare_samples
from calora.ca.select import Selection, select_all, select_topk
from calora.config import LoraConfig_, TrainConfig
from calora.models.projections import ProjectionTable, head_unit_name, parse_unit_name
from calora.models.sliced_lora import (
    SlicedLoRALinear,
    head_slice_index,
    lora_type_for_kind,
)
from calora.models.toy import ToyAdapter
from calora.train.lora import (
    attach_lora,
    load_lora,
    lora_state_dict,
    plan_sliced_lora,
    save_lora,
    train_lora,
)

AXES = ["style", "viewpoint"]

pytest.importorskip("peft")


@pytest.fixture
def head_table(adapter: ToyAdapter) -> ProjectionTable:
    return ProjectionTable.from_model(adapter.unet, granularity="head")


def _selection(table: ProjectionTable, names: list[str]) -> Selection:
    return Selection.from_dict({"names": names, "granularity": "head"}, table)


def _heads_of(table: ProjectionTable, module_name: str) -> list[str]:
    return [spec.name for spec in table if spec.module_name == module_name]


def _first_module(table: ProjectionTable, kind: str) -> str:
    return next(spec.module_name for spec in table if spec.kind == kind)


def _batch(adapter: ToyAdapter, batch: int = 2):
    """A fixed (latents, timesteps, conditioning) triple to push through the UNet."""
    torch.manual_seed(7)
    latents = torch.randn(adapter.latent_shape(batch, adapter.resolution))
    timesteps = torch.full((batch,), 81, dtype=torch.long)
    cond = adapter.encode_prompts(["a street"] * batch)
    return latents, timesteps, cond


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


# --------------------------------------------------------------------------------------
# 1. Equivalence with a standard full LoRA when every head is selected
# --------------------------------------------------------------------------------------


def _copy_lora_weights(source: nn.Module, target: nn.Module, module_name: str) -> None:
    """Give the peft adapter and the sliced adapter identical A and B."""
    peft_a = source.get_submodule(f"{module_name}.lora_A.default").weight
    peft_b = source.get_submodule(f"{module_name}.lora_B.default").weight
    sliced = target.get_submodule(module_name)
    assert sliced.lora_A.shape == peft_a.shape and sliced.lora_B.shape == peft_b.shape
    with torch.no_grad():
        # Non-zero B: with the standard B=0 init every implementation agrees.
        peft_b.copy_(torch.randn_like(peft_b) * 0.1)
        sliced.lora_A.copy_(peft_a)
        sliced.lora_B.copy_(peft_b)


@pytest.mark.parametrize("kind", ["q", "out"])
def test_full_head_selection_matches_a_standard_full_lora(kind: str):
    """Type B (``to_q``, rows) and type A (``to_out.0``, columns), forward and backward.

    Every head selected means the slice *is* the matrix, so the sliced module has
    to reduce to plain ``W0 x + s B A x``. If it does not, no comparison between
    the head and module granularities means anything.
    """
    peft_side, sliced_side = ToyAdapter(seed=0), ToyAdapter(seed=0)
    module_table = ProjectionTable.from_model(peft_side.unet)
    head_table = ProjectionTable.from_model(sliced_side.unet, granularity="head")
    module_name = _first_module(head_table, kind)

    attach_lora(
        peft_side.unet,
        Selection.from_dict({"names": [module_name]}, module_table),
        LoraConfig_(rank=4, alpha=8),
    )
    attach_lora(
        sliced_side.unet,
        _selection(head_table, _heads_of(head_table, module_name)),
        LoraConfig_(rank=4, alpha=8),
    )
    assert isinstance(sliced_side.unet.get_submodule(module_name), SlicedLoRALinear)
    _copy_lora_weights(peft_side.unet, sliced_side.unet, module_name)

    latents, timesteps, cond = _batch(peft_side)
    unadapted = ToyAdapter(seed=0).predict(latents, timesteps, cond).detach()
    outputs = []
    grads = []
    for side in (peft_side, sliced_side):
        prediction = side.predict(latents, timesteps, cond)
        prediction.pow(2).mean().backward()
        outputs.append(prediction.detach())
        grads.append(
            {
                key: param.grad.detach().clone()
                for key, param in side.unet.named_parameters()
                if "lora_" in key and param.grad is not None
            }
        )

    # Agreeing on the base model's output would prove nothing.
    assert not torch.allclose(outputs[0], unadapted, rtol=1e-6, atol=1e-7)
    assert torch.allclose(outputs[0], outputs[1], rtol=1e-6, atol=1e-7)
    peft_grads, sliced_grads = grads
    # Key spellings differ (peft nests the adapter name); the two A/B pairs do not.
    assert len(peft_grads) == len(sliced_grads) == 2
    for letter in ("lora_A", "lora_B"):
        a = next(v for k, v in peft_grads.items() if letter in k)
        b = next(v for k, v in sliced_grads.items() if letter in k)
        assert torch.allclose(a, b, rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------------------
# 2. Locality: the delta is exactly zero outside the selected heads
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["q", "out"])
def test_training_leaves_unselected_head_slices_bit_identical(kind: str, dataset, tmp_path):
    """The 2% budget is only real if the untouched heads are *exactly* untouched."""
    trained = ToyAdapter(seed=0)
    head_table = ProjectionTable.from_model(trained.unet, granularity="head")
    module_name = _first_module(head_table, kind)
    heads = _heads_of(head_table, module_name)
    assert len(heads) >= 2, "H must be >= 2 for a partial selection to exist"
    chosen = heads[:1]
    selection = _selection(head_table, chosen)

    pristine = ToyAdapter(seed=0).unet.get_submodule(module_name).weight.detach().clone()
    train_lora(
        trained, dataset, selection, _train_config(learning_rate=1e-1), output_dir=tmp_path
    )

    module = trained.unet.get_submodule(module_name)
    assert not torch.equal(module.lora_B, torch.zeros_like(module.lora_B)), "B never moved"
    with merged_lora(trained.unet):
        effective = module.base_layer.weight.detach().clone()
    delta = effective - pristine

    spec = head_table[chosen[0]]
    index = head_slice_index([spec.head_index], spec.shape[0 if spec.slice_dim == "row" else 1])
    mask = torch.zeros_like(delta, dtype=torch.bool)
    if spec.slice_dim == "row":
        mask[index] = True
    else:
        mask[:, index] = True
    assert torch.count_nonzero(delta[~mask]) == 0, "LoRA leaked outside the selected head"
    assert torch.count_nonzero(delta[mask]) > 0, "the selected head was not trained"


def test_type_b_forward_only_changes_the_selected_output_rows(head_table, adapter):
    """The forward-side statement of the same property, before any merging."""
    module_name = _first_module(head_table, "q")
    heads = _heads_of(head_table, module_name)
    attach_lora(adapter.unet, _selection(head_table, heads[:1]), LoraConfig_(rank=2, alpha=2))
    module = adapter.unet.get_submodule(module_name)
    with torch.no_grad():
        module.lora_B.copy_(torch.randn_like(module.lora_B))

    x = torch.randn(2, 5, module.base_layer.in_features)
    delta = module(x) - module.base_layer(x)
    selected = module._slice_index
    keep = torch.ones(delta.shape[-1], dtype=torch.bool)
    keep[selected] = False
    assert torch.count_nonzero(delta[..., keep]) == 0
    assert torch.count_nonzero(delta[..., selected]) > 0


def test_head_slice_index_is_head_ordered_and_contiguous_per_head():
    assert head_slice_index([0, 2], 3).tolist() == [0, 1, 2, 6, 7, 8]


def test_lora_type_follows_the_slice_direction():
    assert [lora_type_for_kind(k) for k in ("q", "k", "v")] == ["B", "B", "B"]
    assert lora_type_for_kind("out") == "A"


# --------------------------------------------------------------------------------------
# 3. Attaching changes nothing (B = 0)
# --------------------------------------------------------------------------------------


def test_attaching_a_sliced_lora_is_output_preserving(adapter, head_table):
    latents, timesteps, cond = _batch(adapter)
    before = adapter.predict(latents, timesteps, cond).detach()

    selection = select_topk(
        torch.arange(len(head_table), dtype=torch.float64), head_table, fraction=0.1,
        budget_basis="unit_count",
    )
    attach_lora(adapter.unet, selection, LoraConfig_(rank=4, alpha=4))
    after = adapter.predict(latents, timesteps, cond).detach()
    assert torch.equal(before, after)


def test_attaching_freezes_the_wrapped_weights(adapter, head_table):
    """Like peft's injection: only A and B are trainable afterwards."""
    module_name = _first_module(head_table, "q")
    attach_lora(
        adapter.unet, _selection(head_table, _heads_of(head_table, module_name)[:1]),
        LoraConfig_(rank=2, alpha=2),
    )
    module = adapter.unet.get_submodule(module_name)
    assert not module.base_layer.weight.requires_grad
    assert module.lora_A.requires_grad and module.lora_B.requires_grad


def test_attach_refuses_to_stack_two_sliced_adapters(adapter, head_table):
    selection = _selection(head_table, head_table.names[:2])
    attach_lora(adapter.unet, selection, LoraConfig_(rank=2, alpha=2))
    with pytest.raises(RuntimeError, match="already carries"):
        attach_lora(adapter.unet, selection, LoraConfig_(rank=2, alpha=2))


def test_a_selection_naming_a_unit_the_model_lacks_is_refused(adapter, head_table):
    ghost = _selection(head_table, head_table.names[:1])
    ghost.names = [head_unit_name(parse_unit_name(head_table.names[0])[0], 99)]
    with pytest.raises(KeyError, match="absent from the model"):
        plan_sliced_lora(adapter.unet, ghost)


# --------------------------------------------------------------------------------------
# 4. Checkpoints
# --------------------------------------------------------------------------------------


def _train_a_head_arm(tmp_path, dataset, *, axis: str = "style", num_units: int = 3):
    trained = ToyAdapter(seed=0)
    head_table = ProjectionTable.from_model(trained.unet, granularity="head")
    # Units of several matrices and both slice directions, so the round trip has
    # something to get wrong.
    names = [spec.name for spec in head_table if spec.kind == "q"][:num_units]
    names += [spec.name for spec in head_table if spec.kind == "out"][:num_units]
    selection = _selection(head_table, sorted(names, key=head_table.index_of))
    state = train_lora(
        trained,
        dataset,
        selection,
        _train_config(learning_rate=1e-1),
        output_dir=tmp_path,
        extra_metadata={"axis": axis},
    )
    return trained, selection, state


def test_checkpoint_round_trip_restores_the_heads_and_the_numbers(dataset, tmp_path):
    trained, selection, _ = _train_a_head_arm(tmp_path, dataset)
    saved = lora_state_dict(trained.unet)
    assert any(key.endswith(".lora_head_indices") for key in saved)

    restored = ToyAdapter(seed=0)
    reloaded = load_lora(
        restored.unet,
        tmp_path / "lora-final.safetensors",
        LoraConfig_(rank=2, alpha=2),
        expect_granularity="head",
        expect_axis="style",
    )
    assert reloaded.granularity == "head"
    assert set(reloaded.names) == set(selection.names)

    weights = lora_state_dict(restored.unet)
    assert set(weights) == set(saved)
    assert all(torch.equal(weights[k], saved[k]) for k in saved)

    latents, timesteps, cond = _batch(trained)
    assert torch.equal(
        trained.predict(latents, timesteps, cond).detach(),
        restored.predict(latents, timesteps, cond).detach(),
    )


def test_checkpoint_metadata_records_rank_alpha_granularity_and_axis(dataset, tmp_path):
    from safetensors import safe_open

    _train_a_head_arm(tmp_path, dataset, axis="viewpoint")
    with safe_open(str(tmp_path / "lora-final.safetensors"), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["lora_rank"] == "2"
    assert metadata["lora_alpha"] == "2"
    assert metadata["granularity"] == "head"
    assert metadata["axis"] == "viewpoint"


def test_loading_refuses_a_granularity_mismatch(dataset, tmp_path):
    _train_a_head_arm(tmp_path, dataset)
    restored = ToyAdapter(seed=0)
    with pytest.raises(ValueError, match="head-granularity adapter"):
        load_lora(
            restored.unet,
            tmp_path / "lora-final.safetensors",
            LoraConfig_(rank=2, alpha=2),
            expect_granularity="module",
        )


def test_loading_refuses_an_axis_mismatch(dataset, tmp_path):
    _train_a_head_arm(tmp_path, dataset, axis="style")
    restored = ToyAdapter(seed=0)
    with pytest.raises(ValueError, match="axis"):
        load_lora(
            restored.unet,
            tmp_path / "lora-final.safetensors",
            LoraConfig_(rank=2, alpha=2),
            expect_axis="viewpoint",
        )


def test_a_module_checkpoint_cannot_be_loaded_as_head(dataset, tmp_path):
    """The accident this guards: two arms whose files differ only in what a unit is."""
    trained = ToyAdapter(seed=0)
    table = ProjectionTable.from_model(trained.unet)
    module_selection = select_topk(
        torch.arange(len(table), dtype=torch.float64), table, fraction=0.2
    )
    train_lora(
        trained, dataset, module_selection, _train_config(), output_dir=tmp_path
    )
    restored = ToyAdapter(seed=0)
    with pytest.raises(ValueError, match="module-granularity adapter"):
        load_lora(
            restored.unet,
            tmp_path / "lora-final.safetensors",
            LoraConfig_(rank=2, alpha=2),
            expect_granularity="head",
        )


def test_a_head_checkpoint_without_granularity_metadata_is_refused(dataset, tmp_path):
    """Head-index tensors but no metadata: written by something that did not know
    about one of the two, so neither reading can be trusted."""
    from safetensors.torch import save_file

    trained, _, _ = _train_a_head_arm(tmp_path, dataset)
    path = tmp_path / "no-metadata.safetensors"
    save_file(lora_state_dict(trained.unet), str(path))

    restored = ToyAdapter(seed=0)
    with pytest.raises(ValueError, match="lora_head_indices"):
        load_lora(restored.unet, path, LoraConfig_(rank=2, alpha=2))


def test_save_lora_refuses_a_granularity_the_model_contradicts(adapter, head_table, tmp_path):
    attach_lora(
        adapter.unet, _selection(head_table, head_table.names[:2]), LoraConfig_(rank=2, alpha=2)
    )
    with pytest.raises(ValueError, match="ground truth"):
        save_lora(
            adapter.unet, tmp_path / "wrong.safetensors", {"granularity": "module"},
            cfg=LoraConfig_(rank=2, alpha=2),
        )


# --------------------------------------------------------------------------------------
# 5. The drift path on an adapted model
# --------------------------------------------------------------------------------------


def test_head_units_keep_their_names_under_a_sliced_adapter(adapter, head_table):
    """``PROJECTION_PATTERN`` matches ``.base_layer``, so re-measurement enumerates
    the same units before and after attachment — the basis of the drift Jaccard."""
    selection = _selection(head_table, head_table.names[:3])
    attach_lora(adapter.unet, selection, LoraConfig_(rank=2, alpha=2))
    with merged_lora(adapter.unet):
        adapted = ProjectionTable.from_model(adapter.unet, granularity="head")
    assert adapted.names == head_table.names
    assert adapted.head_counts == head_table.head_counts


def test_merged_lora_restores_the_base_weights_of_a_sliced_adapter(adapter, head_table):
    selection = _selection(head_table, head_table.names[:2])
    attach_lora(adapter.unet, selection, LoraConfig_(rank=2, alpha=2))
    module = adapter.unet.get_submodule(parse_unit_name(head_table.names[0])[0])
    with torch.no_grad():
        module.lora_B.add_(torch.randn_like(module.lora_B) * 0.1)

    before = module.base_layer.weight.detach().clone()
    with merged_lora(adapter.unet):
        assert not torch.allclose(module.base_layer.weight, before)
    assert torch.allclose(module.base_layer.weight, before, atol=1e-6)


def test_drift_snapshots_run_at_head_granularity(dataset, tmp_path):
    """End to end: baseline, training with sliced adapters attached, snapshots."""
    trained = ToyAdapter(seed=0)
    head_table = ProjectionTable.from_model(trained.unet, granularity="head")
    samples = prepare_samples(trained, dataset, axes=AXES, num_samples=2)
    selection = select_topk(
        torch.arange(len(head_table), dtype=torch.float64),
        head_table,
        fraction=0.1,
        budget_basis="unit_count",
    )
    tracker = build_stability_tracker(
        trained,
        samples,
        axes=AXES,
        timestep=81,
        fraction=0.1,
        budget_basis="unit_count",
        trained_selection=selection,
        granularity="head",
    )
    callback = make_stability_callback(
        trained, samples, tracker, axes=AXES, every=2, max_steps=2,
        fraction=0.1, budget_basis="unit_count",
    )
    train_lora(
        trained,
        dataset,
        selection,
        _train_config(max_steps=2, learning_rate=1e-1),
        output_dir=tmp_path,
        callbacks=[callback],
    )
    assert tracker.to_dict()["granularity"] == "head"
    assert len(tracker.snapshots) == 1
    # Drift is compared over head unit names, not matrix names.
    assert tracker.baseline_selection.granularity == "head"
    assert all(
        parse_unit_name(name)[1] is not None for name in tracker.baseline_selection.names
    )
    assert tracker.snapshots[0].num_selected == len(tracker.baseline_selection)
    # The trained weights are back in their unmerged state afterwards.
    assert all(
        not module.merged
        for module in trained.unet.modules()
        if isinstance(module, SlicedLoRALinear)
    )


def test_bf16_autocast_trains_a_sliced_arm(dataset, tmp_path):
    """The A100 arms run under bf16 autocast; the index_add path has to survive a
    reduced-precision forward with fp32 LoRA state."""
    trained = ToyAdapter(seed=0)
    head_table = ProjectionTable.from_model(trained.unet, granularity="head")
    selection = _selection(head_table, head_table.names[:3])
    state = train_lora(
        trained,
        dataset,
        selection,
        _train_config(learning_rate=1e-2, mixed_precision="bf16"),
        output_dir=tmp_path,
    )
    assert torch.isfinite(torch.tensor(state.loss))
    assert all(
        module.lora_A.dtype == torch.float32 and module.lora_B.dtype == torch.float32
        for module in trained.unet.modules()
        if isinstance(module, SlicedLoRALinear)
    )


def test_a_wrapped_projection_still_exposes_the_linear_interface(adapter, head_table):
    """diffusers processors read ``attn.to_q.weight.dtype``; peft forwards it too."""
    module_name = _first_module(head_table, "q")
    attach_lora(
        adapter.unet, _selection(head_table, _heads_of(head_table, module_name)[:1]),
        LoraConfig_(rank=2, alpha=2),
    )
    module = adapter.unet.get_submodule(module_name)
    assert module.weight is module.base_layer.weight
    assert module.in_features == module.base_layer.in_features
    assert module.out_features == module.base_layer.out_features


def test_select_all_at_head_granularity_trains_every_matrix(adapter, head_table, dataset, tmp_path):
    """The degenerate case: all heads everywhere, i.e. full attention LoRA
    expressed through the sliced modules."""
    plans = plan_sliced_lora(adapter.unet, select_all(head_table))
    assert len(plans) == len(head_table.module_names)
    assert all(plan.is_full for plan in plans)
    assert {plan.lora_type for plan in plans} == {"A", "B"}
