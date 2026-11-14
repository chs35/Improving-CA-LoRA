"""The fp32-vs-bf16 drift-measurement equivalence gate (scripts/check_drift_autocast_gate.py).

The gate's verdict on the real arm is an empirical question about SDXL that no CPU
test can answer. What is tested here is the machinery around it: that the timestep
is taken from the arm's own artefacts, that both autocast modes are measured on the
same merged weights with the same samples and seed, that the verdict is exactly the
conjunction of the two pre-registered thresholds, and that the JSON lands on disk
with the fields the plan says it must carry.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys

import pytest
import torch

from calora.ca.measure import prepare_samples
from calora.ca.select import select_topk
from calora.config import LoraConfig_, load_config
from calora.models.projections import ProjectionTable
from calora.train.lora import attach_lora, save_lora
from calora.utils.io import dump_json

AXES = ["style", "viewpoint"]

pytest.importorskip("peft")


@pytest.fixture(scope="module")
def gate(repo_root):
    """scripts/check_drift_autocast_gate.py as a module (it is not part of the package)."""
    sys.path.insert(0, str(repo_root / "src"))
    spec = importlib.util.spec_from_file_location(
        "check_drift_autocast_gate", repo_root / "scripts" / "check_drift_autocast_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- timestep resolution -----------------------------------------------------------------


def test_the_timestep_comes_from_the_arms_stability_json(gate, repo_root, tmp_path):
    """The gate has to bound the error at the ``t`` the snapshots actually used, so
    it reads the arm's own record of it rather than the config default."""
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    dump_json({"timestep": 201, "snapshots": []}, tmp_path / "stability.json")
    dump_json({"selection": {"meta": {"timesteps": [481]}}}, tmp_path / "train_summary.json")

    timestep, source = gate.resolve_timestep(tmp_path, cfg, None)
    assert timestep == 201
    assert "stability.json" in source


def test_the_timestep_falls_back_to_the_selection_then_the_config(gate, repo_root, tmp_path):
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")

    timestep, source = gate.resolve_timestep(tmp_path, cfg, None)
    assert timestep == cfg.measure.timesteps[0]
    assert source.startswith("config:")

    dump_json({"selection": {"meta": {"timesteps": [481, 81]}}}, tmp_path / "train_summary.json")
    timestep, source = gate.resolve_timestep(tmp_path, cfg, None)
    assert timestep == 481
    assert "train_summary.json" in source


def test_an_explicit_timestep_wins(gate, repo_root, tmp_path):
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    dump_json({"timestep": 201}, tmp_path / "stability.json")
    assert gate.resolve_timestep(tmp_path, cfg, 41) == (41, "cli")


# -- the comparison itself ---------------------------------------------------------------


def _adapted(adapter, table, *, fraction=0.2):
    """A LoRA-adapted UNet with a non-zero delta, i.e. a stand-in for a trained arm.

    ``init_lora_weights="gaussian"`` leaves ``lora_B`` at zero, which would make the
    merged weight identical to the base one and the measurement blind to whether
    ``merged_lora`` was entered at all.
    """
    selection = select_topk(torch.arange(len(table), dtype=torch.float64), table, fraction=fraction)
    attach_lora(adapter.unet, selection, LoraConfig_(rank=2, alpha=2))
    with torch.no_grad():
        for name, param in adapter.unet.named_parameters():
            if "lora_B" in name:
                generator = torch.Generator().manual_seed(0)
                param.add_(torch.randn(param.shape, generator=generator) * 0.1)
    return selection


def test_run_gate_reports_both_modes_and_a_thresholded_verdict(gate, adapter, table, dataset):
    """The end-to-end shape of the comparison on the toy model.

    Whether bf16 agrees with fp32 on a 5-layer toy UNet says nothing about SDXL, so
    nothing is asserted about the *values* beyond their being well-formed. What is
    asserted is that ``pass`` is the conjunction of the two pre-registered
    thresholds and nothing else.
    """
    from calora.analysis.stability import merged_lora

    samples = prepare_samples(adapter, dataset, axes=AXES, num_samples=2)
    _adapted(adapter, table)

    with merged_lora(adapter.unet):
        merged_table = ProjectionTable.from_model(adapter.unet)
        payload = gate.run_gate(
            adapter,
            merged_table,
            samples,
            timestep=81,
            axes=AXES,
            selection_axis="style",
            fraction=0.2,
            budget_basis="attn_params",
            seed=0,
        )

    assert set(payload["modes"]) == {"no", "bf16"}
    assert payload["timestep"] == 81
    assert payload["num_samples"] == 2
    assert payload["seed"] == 0
    assert payload["thresholds"] == {
        "spearman": gate.SPEARMAN_THRESHOLD,
        "jaccard": gate.JACCARD_THRESHOLD,
    }
    assert math.isfinite(payload["spearman"]) and -1.0 <= payload["spearman"] <= 1.0
    assert 0.0 <= payload["jaccard"] <= 1.0
    assert payload["pass"] == (
        payload["spearman"] >= gate.SPEARMAN_THRESHOLD
        and payload["jaccard"] >= gate.JACCARD_THRESHOLD
    )

    for mode, summary in payload["modes"].items():
        assert summary["drift_autocast"] == mode
        assert summary["num_selected"] >= 1
        assert summary["seconds"] > 0.0
        # Per-axis CA summaries, both axes, no full [P] tensors anywhere.
        assert all(f"ca/{axis}" in summary["measurement"] for axis in AXES)
        assert 0 < len(summary["top_projections"]) <= gate.TOP_N_REPORTED
        ranked = [entry["score"] for entry in summary["top_projections"]]
        assert ranked == sorted(ranked, reverse=True)
        assert all(entry["name"] in table.names for entry in summary["top_projections"])
    # Serialisable as written: the script dumps this dict verbatim.
    json.dumps(payload)


def test_the_two_modes_are_a_paired_comparison(gate, adapter, table, dataset, monkeypatch):
    """Same samples, same seed, same weights — only the autocast context differs.

    Recorded off the ``measure_concept_awareness`` calls themselves, because the
    estimator noise only cancels if every other argument is identical; a stray
    reseed or a second sample subset would make the gate compare two unrelated
    draws and could pass or fail for the wrong reason.
    """
    from calora.analysis.stability import merged_lora

    calls: list[dict] = []
    original = gate.measure_concept_awareness

    def spy(adapter_, table_, samples_, **kwargs):
        calls.append({"samples": id(samples_), "table": id(table_), **kwargs})
        return original(adapter_, table_, samples_, **kwargs)

    monkeypatch.setattr(gate, "measure_concept_awareness", spy)

    samples = prepare_samples(adapter, dataset, axes=AXES, num_samples=2)
    _adapted(adapter, table)
    with merged_lora(adapter.unet):
        merged_table = ProjectionTable.from_model(adapter.unet)
        gate.run_gate(
            adapter,
            merged_table,
            samples,
            timestep=81,
            axes=AXES,
            selection_axis="style",
            fraction=0.2,
            budget_basis="attn_params",
            seed=7,
        )

    assert len(calls) == 2
    assert calls[0] == calls[1]
    # ... and the arguments are the drift callback's, not measure_ca.py's.
    assert calls[0]["seed"] == 7
    assert calls[0]["timestep"] == 81
    assert calls[0]["accumulate_direction"] is False
    assert calls[0]["compute_paired_cosine"] is False


def test_the_verdict_needs_both_thresholds(gate, monkeypatch):
    """A high rank correlation with a churning top-k must not pass, nor the reverse.

    Driven through ``measure_mode`` rather than reimplementing the rule, so the test
    fails if the script ever loosens the conjunction.
    """
    from calora.ca.select import Selection

    def fake(names, rho_scores):
        def measure_mode(adapter, table, samples, mode, **kwargs):
            selection = Selection.from_dict({"names": names[mode]})
            return (
                torch.tensor(rho_scores[mode], dtype=torch.float64),
                selection,
                {"drift_autocast": mode, "seconds": 1.0, "num_selected": len(names[mode]),
                 "measurement": {"num_samples": 4}},
            )

        return measure_mode

    class _Table:
        names = ["a", "b", "c", "d"]
        granularity = "module"

        def __len__(self):
            return 4

    def verdict(names, scores):
        monkeypatch.setattr(gate, "measure_mode", fake(names, scores))
        return gate.run_gate(
            object(),
            _Table(),
            [0] * 4,
            timestep=81,
            axes=AXES,
            selection_axis="style",
            fraction=0.2,
            budget_basis="attn_params",
            seed=0,
        )

    monotone = {"no": [1.0, 2.0, 3.0, 4.0], "bf16": [1.0, 2.0, 3.0, 4.0]}
    agreed = {"no": ["a", "b"], "bf16": ["a", "b"]}
    churned = {"no": ["a", "b"], "bf16": ["c", "d"]}

    both = verdict(agreed, monotone)
    assert both["spearman"] == pytest.approx(1.0) and both["jaccard"] == 1.0
    assert both["pass"] is True

    # Perfect ranking agreement, disjoint top-k: the Jaccard half must veto.
    ranked_only = verdict(churned, monotone)
    assert ranked_only["spearman"] == pytest.approx(1.0)
    assert ranked_only["jaccard"] == 0.0
    assert ranked_only["pass"] is False

    # Identical top-k, reversed ranking: the Spearman half must veto.
    overlap_only = verdict(agreed, {"no": [1.0, 2.0, 3.0, 4.0], "bf16": [4.0, 3.0, 2.0, 1.0]})
    assert overlap_only["jaccard"] == 1.0
    assert overlap_only["spearman"] == pytest.approx(-1.0)
    assert overlap_only["pass"] is False


# -- the script as the operator runs it --------------------------------------------------


def test_main_writes_the_gate_json_next_to_the_checkpoint(
    gate, repo_root, tmp_path, monkeypatch, tiny_selfgen_dir
):
    """End to end under tiny_cpu.yaml: a saved arm in, one JSON verdict out."""
    from calora.models.loader import load_adapter

    monkeypatch.chdir(repo_root)  # data.augmentation_file is a repo-relative path
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")

    # A checkpoint the script's own adapter can load: same toy model, same LoRA cfg.
    trainer = load_adapter(cfg.model)
    table = ProjectionTable.from_model(trainer.unet)
    _adapted(trainer, table, fraction=cfg.select.fraction)
    lora_path = tmp_path / "arm" / "lora-final.safetensors"
    save_lora(trainer.unet, lora_path, cfg=LoraConfig_(rank=2, alpha=2))

    arm_dir = tmp_path / cfg.name / "gate_arm"
    arm_dir.mkdir(parents=True)
    dump_json({"timestep": 81, "snapshots": []}, arm_dir / "stability.json")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_drift_autocast_gate.py",
            "--config", str(repo_root / "configs" / "tiny_cpu.yaml"),
            "--lora", str(lora_path),
            "--run-name", "gate_arm",
            # The checkpoint above is a module-granularity arm, and load_lora
            # refuses to load one under a head-granularity config.
            "--set", f"output_dir={tmp_path}", "select.granularity=module",
            f"measure.selfgen.dir={tiny_selfgen_dir}",
        ],
    )
    gate.main()

    payload = json.loads((arm_dir / gate.GATE_FILENAME).read_text(encoding="utf-8"))
    assert payload["arm"] == "gate_arm"
    assert payload["lora"] == str(lora_path)
    # The timestep came from the arm's stability.json, not measure.timesteps[0]=1.
    assert payload["timestep"] == 81
    assert "stability.json" in payload["timestep_source"]
    assert payload["num_samples"] == cfg.train.stability_num_samples
    assert payload["seed"] == cfg.measure.seed
    assert set(payload["modes"]) == {"no", "bf16"}
    assert isinstance(payload["pass"], bool)
    assert payload["pass"] == (
        payload["spearman"] >= gate.SPEARMAN_THRESHOLD
        and payload["jaccard"] >= gate.JACCARD_THRESHOLD
    )
    assert payload["thresholds"]["spearman"] == gate.SPEARMAN_THRESHOLD
    assert payload["thresholds"]["jaccard"] == gate.JACCARD_THRESHOLD


def test_main_gates_the_base_model_when_no_lora_is_given(
    gate, repo_root, tmp_path, monkeypatch, tiny_selfgen_dir
):
    """The gate is run once *before* any arm exists, so ``--lora`` is optional.

    With no checkpoint the base weights are what both modes measure: no ``load_lora``
    and no ``merged_lora`` round-trip may happen (asserted by making either fatal),
    the verdict lands under ``<run_dir>/base/``, and the JSON carries the same schema
    as an arm's with ``lora`` recorded as null.
    """
    cfg = load_config(repo_root / "configs" / "tiny_cpu.yaml")
    monkeypatch.chdir(repo_root)  # data.augmentation_file is a repo-relative path

    def forbidden(*args, **kwargs):
        raise AssertionError("the base-model gate must not touch LoRA weights")

    monkeypatch.setattr(gate, "load_lora", forbidden)
    monkeypatch.setattr(gate, "merged_lora", forbidden)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_drift_autocast_gate.py",
            "--config", str(repo_root / "configs" / "tiny_cpu.yaml"),
            "--set", f"output_dir={tmp_path}", f"measure.selfgen.dir={tiny_selfgen_dir}",
        ],
    )
    gate.main()

    payload = json.loads(
        (tmp_path / cfg.name / "base" / gate.GATE_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["arm"] == "base"
    assert payload["lora"] is None
    # No arm artefacts to read a timestep off: the config default is the last resort.
    assert payload["timestep"] == cfg.measure.timesteps[0]
    assert payload["timestep_source"].startswith("config:")
    # ... and the run is at the config's drift granularity, as an arm's gate would be.
    assert payload["granularity"] == cfg.select.granularity
    assert payload["num_samples"] == cfg.train.stability_num_samples
    assert payload["seed"] == cfg.measure.seed
    assert set(payload["modes"]) == {"no", "bf16"}
    assert isinstance(payload["pass"], bool)
    assert payload["pass"] == (
        payload["spearman"] >= gate.SPEARMAN_THRESHOLD
        and payload["jaccard"] >= gate.JACCARD_THRESHOLD
    )
    assert payload["thresholds"]["spearman"] == gate.SPEARMAN_THRESHOLD
    assert payload["thresholds"]["jaccard"] == gate.JACCARD_THRESHOLD
    for mode, summary in payload["modes"].items():
        assert summary["drift_autocast"] == mode
        assert summary["num_selected"] >= 1
        assert all(f"ca/{axis}" in summary["measurement"] for axis in cfg.data.concept_axes)


def test_main_refuses_a_missing_checkpoint(gate, repo_root, tmp_path, monkeypatch):
    """A ``--lora`` that was *asked for* must exist: the gate is then a statement about
    a trained arm, and silently falling back to the base model would be meaningless."""
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_drift_autocast_gate.py",
            "--config", str(repo_root / "configs" / "tiny_cpu.yaml"),
            "--lora", str(tmp_path / "nope.safetensors"),
            "--set", f"output_dir={tmp_path}",
        ],
    )
    with pytest.raises(SystemExit, match="No LoRA checkpoint"):
        gate.main()
