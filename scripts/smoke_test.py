"""Run the entire pipeline in one process on CPU, with no downloads.

    python scripts/smoke_test.py

measure -> score timesteps -> select (4 arms) -> train (sliced type A/B LoRA) ->
checkpoint round trip -> CA drift -> generate -> evaluate. Takes a few seconds on
the toy model. Use it as the gate before moving to a GPU: if this fails, the A100
run will fail too, only slower.

Every number printed here is meaningless — a randomly initialised toy UNet and a
pixel-hash embedder. What is being checked is that the stages connect.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Also importable as a module, not just runnable as a script: the generation
# stage below reuses generate_images.py rather than copying it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from calora.analysis.drift_data import drift_samples as build_drift_samples  # noqa: E402
from calora.analysis.stability import build_stability_tracker, make_stability_callback  # noqa: E402
from calora.ca.measure import measure_concept_awareness, prepare_samples  # noqa: E402
from calora.ca.select import (  # noqa: E402
    Selection,
    aggregate_across_timesteps,
    axis_scores,
    rank_timesteps,
    select_all,
    select_random,
    select_topk,
    timestep_rank_agreement,
)
from calora.config import load_config  # noqa: E402
from calora.data import build_dataset, build_measurement_dataset, to_pil  # noqa: E402
from calora.data.selfgen import (  # noqa: E402
    SelfGeneratedConceptDataset,
    generate_selfgen_set,
    selfgen_latents,
    spec_from_config,
)
from calora.eval.embeddings import PixelEmbedder  # noqa: E402
from calora.eval.metrics import clip_score, cmmd  # noqa: E402
from calora.models.loader import load_adapter  # noqa: E402
from calora.models.projections import ProjectionTable  # noqa: E402
from calora.models.sliced_lora import lora_type_for_kind  # noqa: E402
from calora.train.lora import (  # noqa: E402
    load_lora,
    plan_sliced_lora,
    resolve_target_modules,
    train_lora,
)
from calora.utils.io import dump_json, load_json  # noqa: E402
from calora.utils.logging import get_logger, setup_logging  # noqa: E402
from calora.utils.plotting import plot_loss_curve  # noqa: E402
from calora.utils.seeding import derive_generator  # noqa: E402
from generate_images import build_prompts  # noqa: E402
from generate_images import _image_path as image_path  # noqa: E402

logger = get_logger("smoke")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tiny_cpu.yaml"


def _cover_both_lora_types(
    selection: Selection, table: ProjectionTable, scores: torch.Tensor
) -> Selection:
    """Extend a head selection until it covers a row- *and* a column-sliced matrix.

    Only for the smoke run. A real arm is whatever CA says, and may well be all
    ``to_q``; but then the smoke test would silently stop exercising the type A
    (``to_out.0``, column-sliced) path, which is half of what this stage is for.
    The additions are the highest-CA units of the missing type, so the result is
    still a CA-ordered set — just not a budgeted one.
    """
    chosen = {table.index_of(name) for name in selection.names}
    present = {lora_type_for_kind(table[i].kind) for i in chosen}
    order = torch.argsort(scores.to(torch.float64), descending=True).tolist()
    for lora_type in ("A", "B"):
        if lora_type in present:
            continue
        for index in order:
            if lora_type_for_kind(table[index].kind) == lora_type:
                chosen.add(index)
                break
    indices = sorted(chosen)
    params = sum(table[i].numel for i in indices)
    return Selection(
        names=[table[i].name for i in indices],
        indices=indices,
        scores=[float(scores[i]) for i in indices],
        fraction_requested=selection.fraction_requested,
        fraction_achieved=params / table.total_params,
        params_selected=params,
        params_total=table.total_params,
        budget_basis=selection.budget_basis,
        strategy=f"{selection.strategy}+both_lora_types",
        meta=dict(selection.meta),
        granularity=table.granularity,
        units_total=len(table),
    )


def main() -> int:
    setup_logging("INFO")
    cfg = load_config(CONFIG)
    axes = cfg.data.concept_axes

    logger.info("=== 1. model and projections ===")
    adapter = load_adapter(cfg.model)
    # Every stage runs at select.granularity (head by default: the paper's unit),
    # training included — a head arm is trained with the sliced type A/B adapters
    # of calora.models.sliced_lora. The module table is kept alongside as the
    # granularity-ablation control and for the partition assertion below.
    table = ProjectionTable.from_model(adapter.unet, granularity=cfg.select.granularity)
    module_table = ProjectionTable.from_model(adapter.unet, granularity="module")
    logger.info("%s", table.summary())
    if table.granularity != "module":
        logger.info("  module-granularity control: %s", module_table.summary())
        assert table.total_params == module_table.total_params, (
            "head units must partition the same parameters as the matrices they slice"
        )

    logger.info("=== 2. data ===")
    dataset = build_dataset(cfg.data, resolution=cfg.model.resolution, seed=cfg.measure.seed)
    # Measurement x0 is the paper's: images the (base) model generates itself
    # from c, made once and verified on every later read. Generated into a
    # scratch dir so the smoke run leaves nothing behind; a second pass must find
    # the set complete and generate nothing.
    selfgen_root = Path(tempfile.mkdtemp(prefix="calora-smoke-selfgen-"))
    cfg.measure.selfgen.dir = str(selfgen_root)
    spec = spec_from_config(cfg, adapter)
    generate_selfgen_set(
        adapter, spec, selfgen_root,
        num_images=cfg.measure.selfgen.num_images, batch_size=cfg.measure.selfgen.batch_size,
    )
    calls = []
    original_generate = adapter.generate
    adapter.generate = lambda *a, **k: calls.append(1) or original_generate(*a, **k)
    generate_selfgen_set(
        adapter, spec, selfgen_root,
        num_images=cfg.measure.selfgen.num_images, batch_size=cfg.measure.selfgen.batch_size,
    )
    adapter.generate = original_generate
    assert not calls, "a complete self-generated set must not be generated again"
    # Measurement always reads a centre-cropped, un-mirrored view; training reads
    # `dataset` and may flip.
    measure_dataset = build_measurement_dataset(
        cfg,
        num_items=cfg.measure.num_samples,
        crop="center",
        hflip=False,
        augs_per_sample=cfg.measure.augs_per_sample,
    )
    assert isinstance(measure_dataset, SelfGeneratedConceptDataset)
    latents = selfgen_latents(adapter, measure_dataset, cfg.model)
    samples = prepare_samples(
        adapter, measure_dataset, axes=axes, num_samples=cfg.measure.num_samples, latents=latents
    )
    logger.info(
        "%d samples x %d augmentation(s), latent shape %s",
        len(samples), samples[0].augs_per_sample, tuple(samples[0].latents.shape),
    )
    assert samples[0].augs_per_sample == cfg.measure.augs_per_sample

    logger.info("=== 3. CA measurement across timesteps ===")
    results = {}
    for timestep in cfg.measure.timesteps:
        results[timestep] = measure_concept_awareness(
            adapter,
            table,
            samples,
            timestep=timestep,
            axes=axes,
            accum_device=cfg.measure.accum_device,
            seed=cfg.measure.seed,
            split_half=cfg.measure.split_half,
        )
        summary = results[timestep].summary()
        logger.info(
            "  t=%-4d CA(style)=%.4f CA(view)=%.4f cos=%.4f  (%d contributions / %d samples)",
            timestep,
            summary["ca/style"]["mean"],
            summary["ca/viewpoint"]["mean"],
            summary[f"cos_of_means/{axes[0]}|{axes[1]}"]["mean"],
            summary["num_samples"],
            summary["num_items"],
        )
        # Within-axis direction agreement (||E[ghat]|| / CA). On the A100 this is
        # the first read on whether a c_Aug template set is pulling one way.
        logger.info(
            "         concentration %s",
            {
                axis: round(value, 4)
                for axis, value in summary["direction_concentration_mean"].items()
            },
        )

    def rank(measurements: dict) -> list:
        return rank_timesteps(
            measurements,
            table,
            axis_pair=(axes[0], axes[1]),
            conflict_quantiles=cfg.select.conflict_sim_quantiles,
            primary_quantile=cfg.select.conflict_primary_quantile,
            fraction=cfg.select.fraction,
            selection_multiplier=cfg.select.tscore_selection_multiplier,
            budget_basis=cfg.select.budget_basis,
            selection_axis=cfg.select.axis,
        )

    logger.info("=== 4. timestep scoring ===")
    scores = rank(results)
    for entry in scores:
        logger.info(
            "  t=%-4d exp_w_sim=%+.4f  sel_sim=%+.4f  S(t)=%.4f  conflicts=%d  (q sweep %s)",
            entry.timestep,
            entry.expanded_weighted_similarity,
            entry.selected_mean_similarity,
            entry.score,
            entry.conflict_count,
            {f"{q:g}": c for q, c in entry.conflict_counts.items()},
        )
    top = [entry.timestep for entry in scores[: cfg.select.top_timesteps]]
    logger.info("  best timesteps: %s", top)

    logger.info("=== 4b. split-half rank agreement ===")
    # measure.split_half gives two independently-accumulated halves for free; each
    # is ranked against its own pooled threshold, never the full run's.
    assert all(r.halves is not None for r in results.values()), "split_half produced no halves"
    half_scores = [
        rank({t: result.halves[side] for t, result in results.items()}) for side in (0, 1)
    ]
    for mode in ("expanded_weighted_similarity", "conflict_count", "score"):
        stats = timestep_rank_agreement(
            half_scores[0], half_scores[1], rank_by=mode, top_n=cfg.select.top_timesteps
        )
        logger.info(
            "  %-30s spearman=%+.3f  top%d overlap=%.3f  %s vs %s",
            mode,
            stats["spearman"],
            stats["top_n"],
            stats["top_overlap"],
            stats["top_a"],
            stats["top_b"],
        )

    logger.info("=== 5. projection selection (4 arms) ===")
    arms = {
        "A_single_t81": select_topk(
            axis_scores(results[81], cfg.select.axis),
            table,
            fraction=cfg.select.fraction,
            budget_basis=cfg.select.budget_basis,
            strategy="single@81",
            meta={"timesteps": [81]},
        ),
        "B_aggregate": select_topk(
            aggregate_across_timesteps(
                results, axis=cfg.select.axis, method="normalized_mean", timesteps=top
            ),
            table,
            fraction=cfg.select.fraction,
            budget_basis=cfg.select.budget_basis,
            strategy="aggregate",
            meta={"timesteps": top},
        ),
        "E_random": select_random(
            table, fraction=cfg.select.fraction, budget_basis=cfg.select.budget_basis, seed=0
        ),
        "E_all": select_all(table),
    }
    for name, selection in arms.items():
        logger.info(
            "  %-14s %2d units over %2d matrices  %6d params  kinds=%s",
            name,
            len(selection),
            len(selection.module_names()),
            selection.params_selected,
            selection.breakdown["kind"],
        )
    logger.info(
        "  Jaccard(A, B)=%.3f   Jaccard(A, random)=%.3f",
        arms["A_single_t81"].jaccard(arms["B_aggregate"]),
        arms["A_single_t81"].jaccard(arms["E_random"]),
    )

    if table.granularity == "head":
        logger.info("=== 5b. head selection -> LoRA targets (type A / type B) ===")
        # E_all selects every head of every matrix, so every adapter it implies
        # covers a whole matrix. A budgeted head selection generally does not, and
        # those matrices get a *sliced* adapter rather than a widened one.
        full_targets = resolve_target_modules(adapter.unet, arms["E_all"])
        assert full_targets == sorted(module_table.names), "type A expansion lost matrices"
        logger.info("  E_all -> %d matrices, all fully selected", len(full_targets))
        for name in ("A_single_t81", "B_aggregate"):
            plans = plan_sliced_lora(adapter.unet, arms[name])
            logger.info(
                "  %-14s -> %d adapter(s): typeA(out,col)=%d typeB(qkv,row)=%d, "
                "heads covered %s, %d fully selected",
                name,
                len(plans),
                sum(1 for p in plans if p.lora_type == "A"),
                sum(1 for p in plans if p.lora_type == "B"),
                [f"{len(p.head_indices)}/{p.num_heads}" for p in plans],
                sum(1 for p in plans if p.is_full),
            )

    logger.info("=== 6. LoRA training + CA drift (%s granularity) ===", table.granularity)
    # Trained at select.granularity: head by default, which means the paper's
    # sliced type A/B adapters. The module (peft) path stays reachable by config
    # because it is the granularity ablation's baseline.
    if table.granularity == "head":
        # Widened just enough that *both* LoRA types are exercised. Which kinds a
        # randomly initialised toy model's CA happens to favour is not something
        # the smoke test should depend on, but "to_out is column-sliced and q/k/v
        # are row-sliced" is exactly what has to keep working end to end.
        train_arm = _cover_both_lora_types(
            arms["A_single_t81"], table, axis_scores(results[81], cfg.select.axis)
        )
        plans = plan_sliced_lora(adapter.unet, train_arm)
        logger.info(
            "  head arm: %d head unit(s) over %d matrices (type A: %d, type B: %d), %d params",
            len(train_arm),
            len(plans),
            sum(1 for p in plans if p.lora_type == "A"),
            sum(1 for p in plans if p.lora_type == "B"),
            train_arm.params_selected,
        )
        assert {p.lora_type for p in plans} == {"A", "B"}, "smoke arm must exercise both types"
        assert any(not p.is_full for p in plans), "smoke arm must exercise a partial slice"
        budget_basis = cfg.select.budget_basis
    else:
        train_arm = select_topk(
            axis_scores(
                measure_concept_awareness(
                    adapter,
                    module_table,
                    samples,
                    timestep=81,
                    axes=axes,
                    accum_device=cfg.measure.accum_device,
                    seed=cfg.measure.seed,
                ),
                cfg.select.axis,
            ),
            module_table,
            fraction=cfg.select.fraction,
            budget_basis="attn_params",
            strategy="single@81",
            meta={"timesteps": [81]},
        )
        budget_basis = "attn_params"
        logger.info(
            "  module arm: %d matrices, %d params (%.1f%% of attention)",
            len(train_arm), train_arm.params_selected, 100 * train_arm.fraction_achieved,
        )

    workdir = Path(tempfile.mkdtemp(prefix="calora-smoke-"))
    try:
        # measure.drift_source=selfgen_current: the baseline is the base set, and
        # every snapshot regenerates x0 from the current (merged) weights.
        drift_samples, drift_provider, drift_x0 = build_drift_samples(cfg, adapter)
        assert drift_provider is not None and drift_x0["data_source"] == "selfgen"
        tracker = build_stability_tracker(
            adapter,
            drift_samples,
            axes=axes,
            timestep=81,
            selection_axis=cfg.select.axis,
            fraction=cfg.select.fraction,
            budget_basis=budget_basis,
            trained_selection=train_arm,
            granularity=table.granularity,
        )
        callback = make_stability_callback(
            adapter,
            drift_samples,
            tracker,
            axes=axes,
            every=cfg.train.stability_every,
            max_steps=cfg.train.max_steps,
            selection_axis=cfg.select.axis,
            fraction=cfg.select.fraction,
            budget_basis=budget_basis,
            sample_provider=drift_provider,
        )
        state = train_lora(
            adapter,
            dataset,
            train_arm,
            cfg.train,
            output_dir=workdir,
            callbacks=[callback],
            extra_metadata={"axis": cfg.select.axis},
        )
        logger.info("  final loss %.5f after %d steps", state.loss, state.step)
        # The checkpoint has to be able to state what it adapts on its own: the
        # reload below goes through the same path scripts/generate_images.py does.
        loaded_selection = load_lora(
            load_adapter(cfg.model).unet,
            workdir / "lora-final.safetensors",
            cfg.train.lora,
            expect_granularity=table.granularity,
            expect_axis=cfg.select.axis,
        )
        assert set(loaded_selection.names) == set(train_arm.names), (
            "the checkpoint does not describe the arm that was trained"
        )
        logger.info(
            "  checkpoint round trip: %d %s unit(s), axis=%s",
            len(loaded_selection),
            loaded_selection.granularity,
            loaded_selection.meta.get("axis"),
        )
        for snapshot in tracker.snapshots:
            logger.info(
                "  step %-4d jaccard=%.3f spearman=%.3f x0=%s (regenerated in %.2fs)",
                snapshot.step, snapshot.jaccard_vs_baseline, snapshot.spearman_vs_baseline,
                snapshot.x0_fingerprint, snapshot.generation_seconds,
            )
        assert tracker.snapshots, "drift tracking recorded no snapshot"
        assert all(s.generation_seconds > 0 for s in tracker.snapshots)
        assert torch.isfinite(torch.tensor(state.loss)), "training diverged"
        assert (workdir / "lora-final.safetensors").exists()
        curve = plot_loss_curve(state.history, workdir / "loss_curve.png")
        logger.info("  loss curve: %s", curve or "skipped (matplotlib not installed)")

        logger.info("=== 7. generation ===")
        # The LoRA is still attached to adapter.unet from step 6, so this samples
        # the trained arm rather than the base model. Prompt composition comes
        # from generate_images.py itself rather than being re-implemented here:
        # a smoke test that exercises a copy of the logic proves nothing about
        # the script the A100 will actually run.
        records = build_prompts(cfg, "both")
        generated_root = workdir / "generated"
        for record in records:
            image = adapter.generate(
                [record["prompt"]],
                num_inference_steps=cfg.eval.num_inference_steps,
                guidance_scale=cfg.eval.guidance_scale,
                generator=derive_generator(cfg.eval.seed, "gen", record["index"]),
            )[0]
            path = image_path(generated_root, record)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            record["path"] = str(path)
        logger.info(
            "  %d images (%s) at %dx%d via %s",
            len(records),
            {c: sum(r["condition"] == c for r in records) for c in
             dict.fromkeys(r["condition"] for r in records)},
            cfg.model.resolution, cfg.model.resolution, adapter.scheduler_name,
        )
        # Determinism is the requirement the ablation rests on: two arms are
        # compared image for image, so a repeat draw must reproduce the pixels.
        repeat = adapter.generate(
            [records[0]["prompt"]],
            num_inference_steps=cfg.eval.num_inference_steps,
            guidance_scale=cfg.eval.guidance_scale,
            generator=derive_generator(cfg.eval.seed, "gen", 0),
        )[0]
        assert list(repeat.getdata()) == list(
            Image.open(records[0]["path"]).convert("RGB").getdata()
        ), "generation is not reproducible from its seed"

        logger.info("=== 8. evaluation (CMMD / CLIP-Score) ===")
        reference_dir = workdir / "reference"
        reference_dir.mkdir(parents=True, exist_ok=True)
        for index in range(min(cfg.eval.reference_num_images, len(measure_dataset))):
            to_pil(measure_dataset[index]["image"]).save(reference_dir / f"{index:05d}.png")

        # PixelEmbedder, so no CLIP download: plumbing only, never a reportable number.
        embedder = PixelEmbedder()
        in_domain = [r for r in records if r["condition"] == "in_domain"]
        cmmd_result = cmmd(
            reference_dir, Path(in_domain[0]["path"]).parent, embedder=embedder
        )
        per_condition = {
            condition: clip_score(
                [r["path"] for r in records if r["condition"] == condition],
                [r["prompt"] for r in records if r["condition"] == condition],
                embedder=embedder,
            )
            for condition in dict.fromkeys(r["condition"] for r in records)
        }
        eval_path = workdir / "eval.json"
        dump_json(
            {
                "run": cfg.name,
                "arm": f"{table.granularity}@t81",
                "embedder": "pixel",
                "reportable": False,
                "num_images": len(records),
                "cmmd": cmmd_result,
                "clip_score": {"per_condition": per_condition},
            },
            eval_path,
        )
        payload = load_json(eval_path)
        logger.info(
            "  eval.json: cmmd=%.4f (%d ref / %d gen)  clip_score=%s  reportable=%s",
            payload["cmmd"]["cmmd"],
            payload["cmmd"]["num_reference"],
            payload["cmmd"]["num_generated"],
            {c: round(v["clip_score"], 3)
             for c, v in payload["clip_score"]["per_condition"].items()},
            payload["reportable"],
        )
        assert torch.isfinite(torch.tensor(payload["cmmd"]["cmmd"])), "CMMD is not finite"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        shutil.rmtree(selfgen_root, ignore_errors=True)

    logger.info("=== smoke test passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
