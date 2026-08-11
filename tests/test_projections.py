from __future__ import annotations

import pytest
import torch
from torch import nn

from calora.models.projections import ProjectionTable, enumerate_attention_projections
from calora.models.toy import ToyAdapter


def test_enumerates_all_qkvout(table: ProjectionTable):
    assert len(table) == 40
    counts = {k: len(v) for k, v in table.group_indices("kind").items()}
    assert counts == {"q": 10, "k": 10, "v": 10, "out": 10}


def test_names_are_module_paths_and_weights_resolve(adapter: ToyAdapter, table: ProjectionTable):
    modules = dict(adapter.unet.named_modules())
    params = dict(adapter.unet.named_parameters())
    for spec in table:
        assert isinstance(modules[spec.module_path], nn.Linear)
        assert params[spec.weight_name].shape == torch.Size(spec.shape)


def test_self_and_cross_attention_are_distinguished(table: ProjectionTable):
    groups = table.group_indices("attn")
    assert set(groups) == {"attn1", "attn2"}
    assert len(groups["attn1"]) == len(groups["attn2"]) == 20
    assert all(table[i].is_cross_attention for i in groups["attn2"])


def test_ordering_is_stable_across_instantiations():
    first = ProjectionTable.from_model(ToyAdapter(seed=0).unet).names
    second = ProjectionTable.from_model(ToyAdapter(seed=7).unet).names
    assert first == second


def test_filter_preserves_binding(adapter: ToyAdapter, table: ProjectionTable):
    only_q = table.filter(kinds=["q"])
    assert len(only_q) == 10
    assert all(spec.kind == "q" for spec in only_q)
    assert set(only_q.parameters) == set(only_q.names)


def test_records_round_trip(table: ProjectionTable):
    restored = ProjectionTable.from_records(table.to_records())
    assert restored.names == table.names
    assert restored.total_params == table.total_params
    assert [s.module_path for s in restored] == [s.module_path for s in table]


def test_peft_base_layer_keeps_canonical_names():
    """After LoRA injection the identifiers must not change (the drift path depends on it)."""
    peft = pytest.importorskip("peft")
    adapter = ToyAdapter(seed=0)
    before = ProjectionTable.from_model(adapter.unet)

    config = peft.LoraConfig(r=2, lora_alpha=2, target_modules=before.names[:4], bias="none")
    peft.inject_adapter_in_model(config, adapter.unet)

    after = ProjectionTable.from_model(adapter.unet)
    assert after.names == before.names
    wrapped = [s for s in after if s.module_path.endswith(".base_layer")]
    assert len(wrapped) == 4
    assert all(s.name == s.module_path[: -len(".base_layer")] for s in wrapped)


def test_rejects_model_without_attention():
    with pytest.raises(ValueError, match="No attention projections"):
        enumerate_attention_projections(nn.Sequential(nn.Linear(4, 4)))


def test_bind_rejects_mismatched_model(table: ProjectionTable):
    with pytest.raises(KeyError, match="missing"):
        table.bind(nn.Sequential(nn.Linear(4, 4)))
