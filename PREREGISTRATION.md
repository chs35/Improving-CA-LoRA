# Pre-Registration Record

This document is a retrospective consolidation, compiled 2026-08-11, of analysis decisions that were fixed in configuration files, module constants, and a dated internal decision log before the relevant data were seen. It gathers in one place what was previously scattered across the report, the config comments, and that log. The repository's git history begins only at the public release, so the compilation itself cannot prove the timing of any of these decisions; it can only report the dates the internal log records and point at the artifacts that embody them. From this commit onward, any further change to the decisions below is tracked by git history.

**How to read an entry.** Each entry gives what was fixed (*Decision*), why (*Rationale*), and whether it survived (*Status*). Where a decision is embodied in a configuration value, the key is cited; unless another file is named, keys are in `configs/sdxl_cityscapes.yaml`, from which `configs/tsweep_sdxl.yaml` inherits. Dates are those of the internal decision log; where a decision predates that log's dated entries, the date is given as not recorded. "Before measurement" means before the first SDXL CA measurement run, which began 2026-07-29; "before training" means before the first of the eight reported training runs, which began 2026-07-31.

---

## A. What was measured, and on what

### 1. Unit of analysis: the attention-head slice

**Decision.** CA is aggregated and selected at the level of the theoretical attention head — a row slice of a fused Q/K/V projection, a column slice of OUT — giving U = 10,400 units in the SDXL UNet (`select.granularity: head`). The slice direction is defined in exactly one place, `HEAD_SLICE_DIMS` in `src/calora/models/projections.py`; measurement files carry a `unit_granularity` field and mixed-granularity loads are refused.

**Rationale.** The paper's selection unit is the head, not the fused `nn.Linear` module. An earlier implementation aggregated and selected at module granularity (560 units), which was ruled an implementation error rather than an acceptable deviation.

**Status.** **Changed on 2026-07-30.** Stage of the data pipeline seen at change time: the full 13-timestep CA measurement and the timestep ranking had been computed at module granularity, and one module-granularity training run (t = 81, style) had finished training and was in the image-generation stage; no CMMD or CLIP-Score number from any trained adapter existed. Everything reported — measurements, ranking, selections, all training runs — was re-derived at head granularity. The module-granularity artifacts were kept rather than deleted, and the config comment on `select.granularity` records that switching back reproduces them as a granularity ablation. For the record, the change did not manufacture the headline result: the module-granularity ranking's top-5 set was the same {41, 81, 1, 481, 601}, differing only in order.

### 2. Measurement protocol constants

**Decision.** Fixed before measurement, and enforced in code rather than by convention:

- 256 samples per axis per timestep (`measure.num_samples: 256`);
- 4 concept augmentations per sample per axis, drawn without replacement from the 5 templates of that axis (`measure.augs_per_sample: 4`), giving 1,024 concept-gradient contributions per axis per timestep;
- `measure.batch_size: 1`. This is an estimand invariant, not a performance setting: at batch > 1 the measured quantity silently becomes E[‖Σg_c‖/‖Σg_d‖]. The measurement path warns explicitly if a larger batch is configured;
- the measurement path forces `crop="center"` and `hflip=False` regardless of `data.crop` / `data.hflip` (four call sites, including `scripts/measure_ca.py` and the drift-gate script);
- noise draws paired across axes (both axes share the sample's noised latent and base prediction) and across timesteps (`measure.paired_across_timesteps: true`; the seed derivation does not take t as a key);
- fp32 throughout the measurement path (`model.dtype: float32`, `model.vae_dtype: float32`);
- loss reduction `mean` where the paper writes a sum, which is invariant for CA because the constant cancels in the ratio of norms, and protects floating-point range.

**Rationale.** Each constant protects the estimator or the comparison that the study rests on: pairing is what makes cross-axis and cross-timestep differences attributable to the axis or the timestep, batch 1 is what makes the measured quantity the paper's quantity, and the augmentation multiplicity is cheap because the forward pass and the denominator backward pass are shared within a sample (m = 4 costs about 3× m = 1 for 4× the concept-gradient sample).

**Status.** Unchanged. Dates: the batch-1, pairing and center-crop invariants come from the implementation and review phases (2026-07, exact dates not recorded); `augs_per_sample: 4` was fixed 2026-07-28; the sample count of 256 was carried over from the implementation phase and confirmed on 2026-07-28, when the paper was found not to state one; fp32 for the sweep was confirmed 2026-07-29 before the sweep started. The only precision exception anywhere in the study is the drift path (entry 15), which never touches the selection measurement.

### 3. c_Aug templates: the paper's sets, verbatim

**Decision.** `configs/prompt_augmentations.json` version 3 — the paper's Appendix B.2 sets, five templates per axis, quoted verbatim — replacing an earlier set of our own design (v2). The templates replace the whole fixed training prompt rather than being appended to a per-image caption, which couples this file to `data.caption_template`.

**Rationale.** Reproduction fidelity over a set we designed ourselves. The v2 set had deliberately excluded a first-person/eye-level viewpoint template, on the grounds that Cityscapes is entirely first-person, so such a template pulls the perturbed condition toward the real images and can cancel against the other slots inside the within-axis mean E[ĝ_viewpoint]. The paper's viewpoint set contains "eye-level view", so that tension was inherited knowingly: the paper is immune because it consumes only gradient norms, for which within-axis cancellation is harmless, whereas our direction metrics are exposed. The decision was to accept the paper's set and settle the question by measurement rather than by argument, using direction concentration (entry 10) as the check.

**Status.** Unchanged; fixed 2026-07-28, before any SDXL measurement. The recorded tension did not materialize — viewpoint concentration is 0.20-0.44 across the grid, 6-14× the random floor (report §5.1).

### 4. The 13-timestep measurement grid

**Decision.** Measure at [1, 41, 81, 121, 201, 301, 401, 481, 601, 701, 801, 901, 981] (`measure.timesteps` in `configs/tsweep_sdxl.yaml`).

**Rationale.** All four of the paper's candidates (1, 81, 201, 481) are included so the comparison is on the paper's own ground; the low-t region where those candidates cluster and where the paper's winner sits is refined at step 40 (41 between 1 and 81, 121 between 81 and 201); the rest of the schedule is covered coarsely at ~100-120 spacing so that a low-t advantage is tested rather than assumed, with 981 as a near-endpoint of the schedule; every t satisfies t ≡ 1 (mod 20), aligning the grid with the paper's candidates. Thirteen points is a cost decision: one CA measurement is far cheaper than one LoRA run, but the sweep still had to fit a single overnight measurement budget.

**Status.** Unchanged; the same grid was re-measured at head granularity after the change in entry 1. The grid was fixed before the first sweep started on 2026-07-29; its design rationale was written into the decision log on 2026-07-30 as a record supplement, and the date on which the grid itself was chosen is not separately recorded.

---

## B. The timestep criterion

### 5. Primary similarity source: `cos_of_means`

**Decision.** The per-unit cross-axis similarity is cos(E[ĝ_style], E[ĝ_viewpoint]) — the cosine of the axis-mean directions (`select.similarity_source: cos_of_means`). The alternative estimator `mean_of_cos` = E[cos(ĝ_style, ĝ_viewpoint)] is computed and logged on every scoring run as a weighting-invariant control; both are always reported.

**Rationale.** Two reasons for the primary choice: the implicit per-sample 1/‖g_d‖ weighting inside `cos_of_means` matches CA's own normalization, and averaging directions before taking the cosine matches how training itself averages gradients. The control is kept because `cos_of_means` can be timestep-dependent even when the underlying directions are not — a property of the estimator identified by a test during implementation and documented as such — so a reader needs the weighting-invariant reading to see whether a timestep effect is direction or weighting.

**Status.** Unchanged; fixed 2026-07-28, before measurement. Both estimators are reported in the results (report §5.1: the control agrees on the leading timesteps but ranks less stably, split-half 0.775 against the primary's 0.956).

### 6. Primary timestep metric: expanded weighted similarity (`ews`)

**Decision.** Rank timesteps by the CA-weighted mean cross-axis similarity over the top n·k units by CA, lower being better, with n = 5 (`select.tscore_selection_multiplier: 5`; k = 208, so 1,040 units). The auxiliary metric is `conflict_count`, the number of selected units whose similarity exceeds a threshold set as a quantile of the similarity distribution pooled across all timesteps, swept over {0.1, 0.2, 0.3, 0.4, 0.5} (`select.conflict_sim_quantiles`) with 0.3 as the primary quantile (`select.conflict_primary_quantile`).

**Rationale.** Reading similarity only over the top k units would read exactly at the selection boundary and inherit its bias; expanding to n·k softens the boundary, and CA weighting keeps the expansion from being dominated by the low-CA units it adds. `conflict_count` is the literal reading of the same hypothesis (count the units that are both high-CA and directionally aligned), kept as a differently-failing second view. The threshold quantile is pooled across timesteps rather than computed per timestep because a per-timestep quantile makes the count identical at every t and the metric vacuous.

**Status.** Unchanged as the primary metric; fixed 2026-07-28, before measurement, and reflected the same day in the code default for the ranking metric. `conflict_count`'s status changed as entry 7 describes.

### 7. Demotion rule for `conflict_count` — the rule fired

**Decision.** A pre-registered demotion rule: if the timestep ranking inverts across the quantile sweep, `conflict_count` loses primary-metric status. Checking this was designated the first thing to do after the sweep.

**Rationale.** A ranking that depends on where a free threshold parameter is placed cannot support a claim about which timestep is better.

**Status.** **The rule fired.** The t = 41 versus t = 81 order inverts between q = 0.1 (t = 81 ahead, 55 against 45) and q = 0.3 (t = 41 ahead, 120 against 95), at head granularity as it had at module granularity, so `conflict_count` was demoted to auxiliary. The split-half agreement corroborates independently (0.837 against `ews`'s 0.956). Nothing downstream changed, because `ews` had been fixed as primary before measurement and both metrics agree on the winning timestep. Recorded in report §3.2 and §5.1; the mechanism — a hard threshold applied to a small integer count — is analysed there.

### 8. The composite score S(t), demoted before measurement

**Decision.** An earlier composite score S(t) = Σ CA_norm · (1 − sim) was removed from candidacy as a primary metric. It is still computed and stored on every scoring run as a diagnostic (the `score` field of `timestep_scores.json`), and is not used for any decision.

**Rationale.** Decision-problem mismatch. The selection is binary — a unit is trained or it is not, and CA magnitude is discarded once the top-k is taken — while S(t) keeps CA as a continuous weight over all units. Only the device of CA-weighted averaging was carried forward, into `ews`.

**Status.** Unchanged; demoted 2026-07-28, before measurement.

### 9. Split-half agreement as the ranking-reliability instrument

**Decision.** Recompute the entire timestep ranking on even and odd halves of the samples and report the Spearman correlation and top-n overlap between the two half-rankings (`measure.split_half: true`). The split is by **sample**, not by contribution. A full bootstrap over direction vectors was rejected.

**Rationale.** The four augmentations of one sample share that sample's noised latent and base prediction, so their contributions are correlated; splitting by contribution would place correlated halves on both sides of the split and overstate agreement. The bootstrap alternative would require serializing per-contribution direction vectors, which the storage budget did not justify (`measure.save_direction_vectors: false`).

**Status.** Unchanged; proposed and approved 2026-07-28, before measurement. Reported in §5.1: `ews` split-half Spearman 0.956, with both halves producing the same top-3 set {41, 1, 81}.

### 10. Direction concentration as a validity precondition

**Decision.** Record ‖E[ĝ_axis]‖ / E[‖ĝ_axis‖] ∈ [0, 1] per axis per timestep (enabled by `measure.accumulate_direction: true`), and treat the direction-based criterion as valid only if this sits well above the random-direction floor of approximately 1/√n_contrib.

**Rationale.** The primary similarity is a cosine between axis-mean directions, which is meaningful only if an axis's templates push in a broadly consistent direction; if they cancel inside E[ĝ], the cosine compares two near-zero vectors. This failure mode is specific to our direction-based use — the paper consumes only norms — and it was live because of the eye-level template noted in entry 3. The statistic costs nothing extra to measure: E[‖ĝ‖] is CA itself, so it is derived from values already stored.

**Status.** Unchanged; added 2026-07-28, before measurement. The precondition passed: concentration is 0.20-0.65 across all timesteps and axes, 6-20× the floor of approximately 1/√1024 ≈ 0.031 (report §5.1).

---

## C. Selection and training

### 11. Head selection is pure CA top-k; the budget is a unit count

**Decision.** The trained set is the top-k of the per-axis CA ranking and nothing else. The direction signal is used **only** to choose the timestep and never to filter or re-rank units. The budget is 2% of units (`select.fraction: 0.02`, `select.budget_basis: auto`, which resolves to `unit_count` at head granularity), i.e. ⌊0.02 · 10,400⌋ = 208 units per selection.

**Rationale.** Pure CA, because the study's first-order goal was quantifying the timestep choice; excluding high-similarity units would deviate further from the paper than the study intends, and a unit with high CA on both axes — for example one carrying image detail that both concepts touch — can still be worth training. Unit count rather than parameter count, because the paper's budget is stated as a count ("top-k% of projection layers"), matching its Fig. 4 proportions; an earlier implementation used an attention-parameter budget, which was a second deviation from the paper.

**Status.** Unchanged. Dates: the pure-CA rule was fixed 2026-07-28; the unit-count budget basis was fixed 2026-07-30 when the paper's Appendix A.2 and Algorithms 2-4 were checked directly — both before any of the eight reported training runs. A consequence, disclosed in report §3.1 and §5.2, is that the effective parameter fraction is not constant across selections (2.09-2.37%), because unit widths differ between 640- and 1280-channel blocks.

### 12. Multi-timestep aggregation width: pre-registered 5, changed to 3

**Decision.** The multi-timestep selection normalizes CA within each timestep, averages the normalized scores over the top-N timesteps by `ews`, and takes the top 208 units of the aggregate (`select.aggregation: normalized_mean`, `select.normalization: sum`). The pre-registered width was N = 5. The released configs still carry `select.top_timesteps: 5` in both `configs/sdxl_cityscapes.yaml` and `configs/tsweep_sdxl.yaml`, each with a comment recording that the trained multi-t selection overrode it to 3 at selection time.

**Rationale for the change.** The pre-registered boundary turned out to be indefensible on the measured ranking: 5th place (t = 601, `ews` 0.539) and 6th place (t = 121, 0.540) are a tie, so including one and excluding the other is arbitrary, while the 3rd-to-4th boundary (0.382 → 0.500) is the largest gap in the ranking. A top-3 aggregate had been considered and rejected at module granularity on 2026-07-30 because it degenerated to the t = 81 selection (9 of 10 units identical); at head granularity that degeneracy is gone — the top-3 aggregate shares 81% with t = 81 and 82% with t = 41, against a 79% t = 81/t = 41 baseline — so aggregation genuinely moves the selection.

**Status.** **Changed on 2026-07-31.** Stage of the data pipeline seen at change time: the head-granularity timestep ranking had been computed, including `ews` values and split-half agreement — no training run had started at head granularity, no images had been generated, and no evaluation metric existed. Disclosed in report Table 1 row 5, §3.3 (footnote) and §5.2. The top-5 selections were generated and are retained for analysis (`results/sdxl_cityscapes/selection_B_top5_*.json`) but were never trained; the trained artifact records `timesteps: [41, 1, 81]`.

### 13. Training data scale: the full Cityscapes train split

**Decision.** T2I fine-tuning uses the full Cityscapes train split, 2,975 images (`data.split: train`), for 10,000 steps at batch 1 — roughly 3.4 epochs. Fixed before training, and identical for every run in the study.

**Rationale.** The paper's Table 12 does not state the size of the T2I training set, and the full split was the default reading.

**Status.** Unchanged, but its relation to the paper changed: on 2026-08-01, after the first CMMD numbers were in, an investigation into an unexpectedly strong CMMD established that the paper's headline setting is few-shot (~9 images, ~1,100 epochs). The deviation was therefore identified after the fact rather than chosen against a known alternative; nothing in the protocol was changed in response, because the reproduction criterion had already been restricted to the relative axis signature (entry 17), and all eight runs share the same data. The consequence — absolute CMMD values are not comparable with the paper's — is disclosed in report Table 1 row 1 and §4.2, together with the 0%-control calibration.

### 14. One training seed per run

**Decision.** All training runs use a single seed (`train.seed: 0`). The internal plan allowed for 3 seeds on the main arms if GPU time permitted.

**Rationale.** Compute budget; the 3-seed extension was conditional from the start.

**Status.** Unchanged, and the conditional extension was never triggered — the condition remained an open item in the decision log at the end of the study. Every trained number in the report comes from one seed, disclosed in Table 1 row 11 and §7.

---

## D. Drift tracking

### 15. bf16 autocast for drift snapshots, behind an equivalence gate

**Decision.** The drift snapshots may run under bf16 autocast only if a pre-registered equivalence gate passes: against an fp32 measurement over identical samples and seed, rank Spearman over all units ≥ 0.99 **and** top-2% Jaccard ≥ 0.8, both required. The thresholds were fixed as module constants `SPEARMAN_THRESHOLD = 0.99` and `JACCARD_THRESHOLD = 0.8` in `scripts/check_drift_autocast_gate.py` before the gate was run, with the comment stating that they are not tunable after the fact. If the gate passes, bf16 is applied to every run so that drift stays comparable across runs, and to both the snapshots and the step-0 baseline, since applying it to only one side would show up as fake drift. It is never applied to the selection measurement: `scripts/measure_ca.py` is untouched by this switch.

**Rationale.** Drift snapshots were about half of a training run's wall clock and bf16 nearly halves that, which is sound only if the measured CA is unchanged for the purpose it is put to — ranking units — hence a gate stated as two ranking statistics rather than as an error tolerance.

**Status.** Unchanged after fixing. Thresholds and gate script: 2026-07-30. Gate run on the base UNet and `measure.drift_autocast: bf16` enabled: 2026-07-31, before the reported training runs. The gate passed with Spearman 0.9988 and Jaccard 0.9349 at a 1.89× speedup (`results/sdxl_cityscapes/drift_autocast_gate.json`; the config comment on `measure.drift_autocast` cites this file under its original `runs/` path).

### 16. Drift re-measurement is observational only

**Decision.** CA is re-measured every 500 steps on a fixed 64-sample subset (`train.stability_every: 500`, `train.stability_num_samples: 64`), against a step-0 baseline re-measured with those same 64 samples rather than reused from the 256-sample selection scores. No selection, hyperparameter, or stopping decision in this study consumes the drift signal.

**Rationale.** Consuming drift would make the training trajectory depend on the quantity being observed, which is the study's third research question. Re-measuring the baseline at the snapshot's own sample count separates estimation noise from genuine drift and yields the per-run noise floor reported as the baseline ∩ trained column of Table 5.

**Status.** Unchanged. The step-0 re-measurement rule was fixed during the implementation review round (2026-07, exact date not recorded); every run ran the full 10,000 steps. Stated in report §3.4; consuming the drift signal is listed as future work in §6.

---

## E. Scope, evaluation, and controls

### 17. Reproduction success criterion: the axis contrast, not the absolute numbers

**Decision.** The reproduction claim is the paper's Fig. 6 axis-contrast signature — style-selected adapters achieve better domain fit (CMMD) while viewpoint-selected adapters better preserve weather-prompt controllability (CLIP-Score) — evaluated per selection criterion. Matching absolute values is not a criterion.

**Rationale.** The criterion was redefined from an earlier "CA top-2% beats full LoRA" formulation, which the metrics in scope cannot support in either direction: the paper's own tables show full LoRA winning the domain-fit metric, and the paper's case for CA selection runs through downstream mIoU, which is out of scope here.

**Status.** Unchanged after the redefinition on 2026-07-30. Stage of the data pipeline seen at change time: module-granularity measurement and ranking only — no CMMD or CLIP-Score number from any trained adapter existed. The criterion is what §5.3 reports against (4 of 4 selection pairs).

### 18. Evaluation protocol fixed before any generation

**Decision.** CMMD against 500 center-cropped Cityscapes val images (`eval.reference_split: val`, `eval.reference_num_images: 500`); CLIP ViT-L/14 as the backbone for both metrics (`eval.clip_model: openai/clip-vit-large-patch14`); 500 in-domain generations plus four weather conditions × 100 images (`eval.num_images`, `eval.weather_conditions`, `eval.images_per_condition`); 25 inference steps and guidance 5.0 from the paper's Table 12; deterministic per-image seeds. The reference set and the backbone are marked in the config itself as our decisions, because the paper specifies neither. Downstream segmentation (mIoU) was placed out of scope at the start of the project (date not recorded, implementation phase) and stayed out.

**Rationale.** The val split is exactly 500 images, matching the generated set size; ViT-L/14 is the backbone of the CMMD paper and the literature standard; mIoU would require the label-generator stages, which lie outside the study's research questions and would roughly double its compute.

**Status.** Unchanged; fixed 2026-07-28, before any image was generated.

### 19. Scope of the ablation controls: excluded, then re-included post hoc

**Decision.** The two no-CA controls at matched budget — full attention (all 10,400 units) and random 2% — were dropped from scope on 2026-07-30, to save GPU time and because the study was not following the paper's full ablation. They were later re-included and run under the identical protocol.

**Rationale for re-inclusion.** Negative controls proved necessary to discriminate what CA selection actually buys: without a random selection carrying no concept signal, nothing bounds the value of concept-aware placement. GPU time had also freed up once the main matrix finished.

**Status.** **Changed, and this is a post-hoc scope change — stated plainly as such.** The eight-run matrix finished on 2026-08-03; the controls were run from 2026-08-04, and the re-inclusion rationale was formally entered in the decision log on 2026-08-07. The decision to add them was therefore taken with the eight CA runs' CMMD and CLIP-Score results already known, and their outcome materially shaped the report's framing (a random 2% selection matches or beats most CA runs on CMMD, and beats the t = 41 style adapter on both metrics). They should be read as post-hoc additions, not as pre-registered comparisons. Disclosed in report Table 1 row 6 and §5.5.

---

## How to audit

Every entry above is backed by an artifact in this repository rather than by this document's own narration.

| Entry | Artifact |
| --- | --- |
| 1, 2, 5, 6, 9, 10, 11, 14, 16, 18 | `configs/sdxl_cityscapes.yaml` — the cited keys and their comments; the measurement invariants are additionally enforced in `scripts/measure_ca.py` and `src/calora/models/projections.py` |
| 3 | `configs/prompt_augmentations.json` — version 3, its `note` block records the replacement, the coupling to `data.caption_template`, and the eye-level tension verbatim |
| 4 | `configs/tsweep_sdxl.yaml` — `measure.timesteps` and its comment |
| 6, 7, 8 | `results/tsweep_sdxl_head/timestep_scores.json` — the ranking, the per-quantile `conflict_counts` that make the demotion rule's firing checkable (t = 41: 45 at q = 0.1, 120 at q = 0.3), the split-half agreement block, and the retained S(t) diagnostic under `score` |
| 12 | `results/sdxl_cityscapes/selection_B_top5_*.json` versus `selection_B_top3_*.json` — the `meta.timesteps` field distinguishes `[41, 1, 81, 481, 601]` from the trained `[41, 1, 81]`; `select.top_timesteps: 5` remains in both configs |
| 15 | `results/sdxl_cityscapes/drift_autocast_gate.json` — the two statistics, the thresholds they were tested against, and the pass flag; the thresholds themselves are module constants in `scripts/check_drift_autocast_gate.py` |
| 11, 19 | `results/sdxl_cityscapes/selection_*.json` — 208 units in every selection, including the random control, with per-selection parameter statistics |

The `results/` paths above are landing in the next commit of this series.
