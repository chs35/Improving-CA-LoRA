# Released evidence manifest

Every file in `results/` is a verbatim copy of an artifact produced by the experiment runs in
the parent project tree (`/workspace/Optimizing-CA-LoRA/runs/`). Nothing here was edited,
regenerated or reformatted after the fact: the `sha256` column below is the hash of both the
released copy and its upstream original, and file mtimes were preserved on copy (`cp -p`).

Some files were **renamed** on release so that each arm's artifacts sit together in one
directory (e.g. `runs/sdxl_cityscapes/train_A_t81_style.log` becomes
`results/sdxl_cityscapes/A_t81_style/train.log`). The *Original path* column always gives the
true upstream source, so any claim in the report can be traced back to the run that produced it.

Full-length hashes for every file live in [`sha256sums.txt`](sha256sums.txt); the table below
abbreviates them to the first 12 hex digits for readability. To verify a checkout:

```bash
cd results && sha256sum -c sha256sums.txt
```

`sha256sums.txt` and `MANIFEST.md` are themselves excluded from the checksum list -- they are
derived from the evidence files rather than part of it, and hashing them here would be circular.
The same applies to the three `sha256_*` manifests covering the Hugging Face artifacts
([`sha256_checkpoints.txt`](sha256_checkpoints.txt), [`sha256_measurements.txt`](sha256_measurements.txt),
[`sha256_generated_images.csv`](sha256_generated_images.csv)): they are checksum lists, not evidence,
and the files they describe live on the Hub rather than in this tree.


## Files in git


### `results/tsweep_sdxl_head/` -- Head-level timestep sweep -- the source of Table 2

13-timestep CA measurement sweep at head granularity. `timestep_scores.json` carries the
expanded weighted similarity (ews), conflict counts and split-half agreement; `measure_summary.json`
carries the per-timestep direction-concentration means.

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `config.yaml` | `runs/tsweep_sdxl_head/config.yaml` | 2,281 | `c607c8489926` |
| `measure_summary.json` | `runs/tsweep_sdxl_head/measure_summary.json` | 15,599 | `6a8c22d8fc76` |
| `run.log` | `runs/tsweep_sdxl_head/run.log` | 42,021 | `f2118d1d3189` |
| `timestep_scores.json` | `runs/tsweep_sdxl_head/timestep_scores.json` | 43,177 | `491cf0275e8a` |

4 files, 103,078 bytes (100.7 KiB).


### `results/sdxl_cityscapes/` -- SDXL / Cityscapes experiment -- shared inputs

Run config, the drift+autocast equivalence gate (Sec. 3.4) and the head-level unit selections
actually trained. The `*_B_top5_*` pair is retained as evidence of the pre-registered top-5
aggregation that preceded the disclosed change to top-3.

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `config.yaml` | `runs/sdxl_cityscapes/config.yaml` | 2,280 | `29c0079938f9` |
| `drift_autocast_gate.json` | `runs/sdxl_cityscapes/base/drift_autocast_gate.json` | 9,378 | `49879d036333` |
| `gate_base.log` | `runs/sdxl_cityscapes/gate_base.log` | 4,817 | `570cafbce9ff` |
| `selection_A_t81_style.json` | `runs/sdxl_cityscapes/selection_A_t81_style.json` | 20,295 | `a44b2e521b4a` |
| `selection_A_t81_viewpoint.json` | `runs/sdxl_cityscapes/selection_A_t81_viewpoint.json` | 20,419 | `a92f5497bb69` |
| `selection_B_top3_style.json` | `runs/sdxl_cityscapes/selection_B_top3_style.json` | 21,018 | `d8e92587b37b` |
| `selection_B_top3_viewpoint.json` | `runs/sdxl_cityscapes/selection_B_top3_viewpoint.json` | 21,065 | `4a8504f4a4e3` |
| `selection_B_top5_style.json` | `runs/sdxl_cityscapes/selection_B_top5_style.json` | 21,010 | `0d0be196f22e` |
| `selection_B_top5_viewpoint.json` | `runs/sdxl_cityscapes/selection_B_top5_viewpoint.json` | 21,048 | `a541e8e778b4` |
| `selection_C_t41_style.json` | `runs/sdxl_cityscapes/selection_C_t41_style.json` | 20,369 | `399c0697687b` |
| `selection_C_t41_viewpoint.json` | `runs/sdxl_cityscapes/selection_C_t41_viewpoint.json` | 20,382 | `e88c1607968b` |
| `selection_D_t1_style.json` | `runs/sdxl_cityscapes/selection_D_t1_style.json` | 20,372 | `7ab7c8b2a547` |
| `selection_D_t1_viewpoint.json` | `runs/sdxl_cityscapes/selection_D_t1_viewpoint.json` | 20,371 | `a3a9ddaee4b4` |
| `selection_E_all.json` | `runs/sdxl_cityscapes/selection_E_all.json` | 810,090 | `7c38f7c4e3bd` |
| `selection_E_random.json` | `runs/sdxl_cityscapes/selection_E_random.json` | 19,959 | `21059640a668` |

15 files, 1,052,873 bytes (1.00 MiB).


### `results/sdxl_cityscapes/A_t81_style/` -- Arm `A_t81_style`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/A_t81_style/eval.json` | 2,091 | `e5aa79715a05` |
| `eval.log` | `runs/sdxl_cityscapes/eval_A_t81_style.log` | 20,711 | `65340cb25480` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/A_t81_style/generated/manifest.json` | 530,814 | `fbc103589b4c` |
| `loss_curve.png` | `runs/sdxl_cityscapes/A_t81_style/loss_curve.png` | 115,545 | `267e0f8cb9fd` |
| `stability.json` | `runs/sdxl_cityscapes/A_t81_style/stability.json` | 549,549 | `776cdf71346c` |
| `train.log` | `runs/sdxl_cityscapes/train_A_t81_style.log` | 49,586 | `87ee4159eefb` |
| `train_summary.json` | `runs/sdxl_cityscapes/A_t81_style/train_summary.json` | 48,319 | `efc134664973` |

7 files, 1,316,615 bytes (1.26 MiB).


### `results/sdxl_cityscapes/A_t81_viewpoint/` -- Arm `A_t81_viewpoint`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/A_t81_viewpoint/eval.json` | 2,096 | `82eb27f54a54` |
| `eval.log` | `runs/sdxl_cityscapes/eval_A_t81_viewpoint.log` | 5,163 | `62572fc97862` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/A_t81_viewpoint/generated/manifest.json` | 534,422 | `578f882ee6ff` |
| `loss_curve.png` | `runs/sdxl_cityscapes/A_t81_viewpoint/loss_curve.png` | 115,988 | `2d14b3e194f8` |
| `stability.json` | `runs/sdxl_cityscapes/A_t81_viewpoint/stability.json` | 582,334 | `efd95e05a930` |
| `train.log` | `runs/sdxl_cityscapes/train_A_t81_viewpoint.log` | 50,528 | `8a2838599ec0` |
| `train_summary.json` | `runs/sdxl_cityscapes/A_t81_viewpoint/train_summary.json` | 48,459 | `82c09da6e108` |

7 files, 1,338,990 bytes (1.28 MiB).


### `results/sdxl_cityscapes/B_top3_style/` -- Arm `B_top3_style`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/B_top3_style/eval.json` | 2,093 | `96cd1e00cf2c` |
| `eval.log` | `runs/sdxl_cityscapes/eval_B_top3_style.log` | 5,161 | `6f3f6adfc5e6` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/B_top3_style/generated/manifest.json` | 531,716 | `bdd28eaa116c` |
| `loss_curve.png` | `runs/sdxl_cityscapes/B_top3_style/loss_curve.png` | 115,579 | `a9cc28b80d9d` |
| `stability.json` | `runs/sdxl_cityscapes/B_top3_style/stability.json` | 545,291 | `b1096b6fceb5` |
| `train.log` | `runs/sdxl_cityscapes/train_B_top3_style.log` | 49,274 | `78aa3751fac7` |
| `train_summary.json` | `runs/sdxl_cityscapes/B_top3_style/train_summary.json` | 49,060 | `07a570aa852c` |

7 files, 1,298,174 bytes (1.24 MiB).


### `results/sdxl_cityscapes/B_top3_viewpoint/` -- Arm `B_top3_viewpoint`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/B_top3_viewpoint/eval.json` | 2,096 | `5e1ea95ae9fc` |
| `eval.log` | `runs/sdxl_cityscapes/eval_B_top3_viewpoint.log` | 62,470 | `c5d35ea3046a` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/B_top3_viewpoint/generated/manifest.json` | 535,324 | `785654096b3b` |
| `loss_curve.png` | `runs/sdxl_cityscapes/B_top3_viewpoint/loss_curve.png` | 115,694 | `5f91a5a12e99` |
| `stability.json` | `runs/sdxl_cityscapes/B_top3_viewpoint/stability.json` | 564,536 | `a34b11f9c2cb` |
| `train.log` | `runs/sdxl_cityscapes/train_B_top3_viewpoint.log` | 51,090 | `5c3d14c27a1d` |
| `train_summary.json` | `runs/sdxl_cityscapes/B_top3_viewpoint/train_summary.json` | 49,117 | `7dd5920d7599` |

7 files, 1,380,327 bytes (1.32 MiB).


### `results/sdxl_cityscapes/C_t41_style/` -- Arm `C_t41_style`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/C_t41_style/eval.json` | 2,090 | `15fd1a27ff03` |
| `eval.log` | `runs/sdxl_cityscapes/eval_C_t41_style.log` | 31,286 | `5c619c291735` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/C_t41_style/generated/manifest.json` | 530,814 | `530917073a92` |
| `loss_curve.png` | `runs/sdxl_cityscapes/C_t41_style/loss_curve.png` | 115,563 | `2bf8d3fe1004` |
| `stability.json` | `runs/sdxl_cityscapes/C_t41_style/stability.json` | 556,710 | `295cd1822a78` |
| `train.log` | `runs/sdxl_cityscapes/train_C_t41_style.log` | 50,203 | `b26c70fe9a9f` |
| `train_summary.json` | `runs/sdxl_cityscapes/C_t41_style/train_summary.json` | 48,394 | `7beaba04fa68` |

7 files, 1,335,060 bytes (1.27 MiB).


### `results/sdxl_cityscapes/C_t41_viewpoint/` -- Arm `C_t41_viewpoint`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/C_t41_viewpoint/eval.json` | 2,096 | `81e609b164b7` |
| `eval.log` | `runs/sdxl_cityscapes/eval_C_t41_viewpoint.log` | 62,580 | `3fe928a079fe` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/C_t41_viewpoint/generated/manifest.json` | 534,422 | `f129509de337` |
| `loss_curve.png` | `runs/sdxl_cityscapes/C_t41_viewpoint/loss_curve.png` | 115,799 | `4ad4a0a5466d` |
| `stability.json` | `runs/sdxl_cityscapes/C_t41_viewpoint/stability.json` | 583,541 | `b4f308c2107a` |
| `train.log` | `runs/sdxl_cityscapes/train_C_t41_viewpoint.log` | 50,827 | `236d7e751924` |
| `train_summary.json` | `runs/sdxl_cityscapes/C_t41_viewpoint/train_summary.json` | 48,426 | `8076ef0f101f` |

7 files, 1,397,691 bytes (1.33 MiB).


### `results/sdxl_cityscapes/D_t1_style/` -- Arm `D_t1_style`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/D_t1_style/eval.json` | 2,089 | `3b950813a19a` |
| `eval.log` | `runs/sdxl_cityscapes/eval_D_t1_style.log` | 11,315 | `b5aa34779b7c` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/D_t1_style/generated/manifest.json` | 529,912 | `c7df00cc7184` |
| `loss_curve.png` | `runs/sdxl_cityscapes/D_t1_style/loss_curve.png` | 115,607 | `41bc8421a788` |
| `stability.json` | `runs/sdxl_cityscapes/D_t1_style/stability.json` | 556,883 | `224f9df8ee90` |
| `train.log` | `runs/sdxl_cityscapes/train_D_t1_style.log` | 49,706 | `8faca198b850` |
| `train_summary.json` | `runs/sdxl_cityscapes/D_t1_style/train_summary.json` | 48,403 | `c37e82e32977` |

7 files, 1,313,915 bytes (1.25 MiB).


### `results/sdxl_cityscapes/D_t1_viewpoint/` -- Arm `D_t1_viewpoint`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/D_t1_viewpoint/eval.json` | 2,096 | `08d01834913a` |
| `eval.log` | `runs/sdxl_cityscapes/eval_D_t1_viewpoint.log` | 10,077 | `008944457f5f` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/D_t1_viewpoint/generated/manifest.json` | 533,520 | `d45d5b37dd61` |
| `loss_curve.png` | `runs/sdxl_cityscapes/D_t1_viewpoint/loss_curve.png` | 115,863 | `c9d5fcb33fc2` |
| `stability.json` | `runs/sdxl_cityscapes/D_t1_viewpoint/stability.json` | 582,435 | `c6f46e19313f` |
| `train.log` | `runs/sdxl_cityscapes/train_D_t1_viewpoint.log` | 50,247 | `7aeb770b6965` |
| `train_summary.json` | `runs/sdxl_cityscapes/D_t1_viewpoint/train_summary.json` | 48,413 | `6dbc0c36b42c` |

7 files, 1,342,651 bytes (1.28 MiB).


### `results/sdxl_cityscapes/E_all/` -- Arm `E_all`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/E_all/eval.json` | 2,088 | `dc3e11854392` |
| `eval.log` | `runs/sdxl_cityscapes/eval_E_all.log` | 9,628 | `09c9efa491bd` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/E_all/generated/manifest.json` | 525,402 | `05a475e90276` |
| `loss_curve.png` | `runs/sdxl_cityscapes/E_all/loss_curve.png` | 115,431 | `9473e904ad3e` |
| `stability.json` | `runs/sdxl_cityscapes/E_all/stability.json` | 1,450,314 | `445a3518578e` |
| `train.log` | `runs/sdxl_cityscapes/train_E_all.log` | 50,450 | `d3e641c78ee6` |
| `train_summary.json` | `runs/sdxl_cityscapes/E_all/train_summary.json` | 878,864 | `41c4480899af` |

7 files, 3,032,177 bytes (2.89 MiB).


### `results/sdxl_cityscapes/E_random/` -- Arm `E_random`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/E_random/eval.json` | 2,091 | `7936fb2b4284` |
| `eval.log` | `runs/sdxl_cityscapes/eval_E_random.log` | 5,264 | `8a9ecc3fd63e` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/E_random/generated/manifest.json` | 528,108 | `21344fd5853b` |
| `loss_curve.png` | `runs/sdxl_cityscapes/E_random/loss_curve.png` | 115,459 | `3057c20a6d44` |
| `stability.json` | `runs/sdxl_cityscapes/E_random/stability.json` | 575,120 | `547fec546fe7` |
| `train.log` | `runs/sdxl_cityscapes/train_E_random.log` | 73,168 | `51dba8282429` |
| `train_summary.json` | `runs/sdxl_cityscapes/E_random/train_summary.json` | 47,982 | `a36687b27b67` |

7 files, 1,347,192 bytes (1.28 MiB).


### `results/sdxl_cityscapes/control_base/` -- Arm `control_base`

| File | Original path (parent project) | Size (bytes) | sha256 |
| --- | --- | ---: | --- |
| `eval.json` | `runs/sdxl_cityscapes/control_base/eval.json` | 2,092 | `35c8967620d6` |
| `eval.log` | `runs/sdxl_cityscapes/eval_control_base.log` | 12,586 | `9e8c647ad6f5` |
| `generation_manifest.json` | `runs/sdxl_cityscapes/control_base/generated/manifest.json` | 531,632 | `a6dfacda90d3` |

3 files, 546,310 bytes (533.5 KiB).


### `results/sdxl_cityscapes_fewshot10/` -- few-shot (10-image) experiment -- the source of Tables 3-5

Added 2026-11-04. Every file is a verbatim copy (`cp -p`) of the parent project's
`runs/sdxl_cityscapes_fewshot10/` (stage logs from its `logs/` directory, renamed to `train.log`,
`generate.log` and `eval.log` per arm; `generated/manifest.json` renamed to
`generation_manifest.json`) and of `runs/sdxl_cityscapes_fewshot10_gate/` for `tensor_core_gate/`.
The `selection_*.json` files are byte-identical to those in `results/sdxl_cityscapes/`: this
experiment re-used the earlier selections. Full-length hashes are in
[`sha256sums.txt`](sha256sums.txt) under the `sdxl_cityscapes_fewshot10/` prefix.

| Directory | Files | Size (bytes) | Contents |
| --- | ---: | ---: | --- |
| `results/sdxl_cityscapes_fewshot10/` | 13 | 1,001,758 | `campaign.log`, `config.yaml`, `selection_A_t81_style.json`, `selection_A_t81_viewpoint.json`, `selection_B_top3_style.json`, `selection_B_top3_viewpoint.json`, `selection_C_t41_style.json`, `selection_C_t41_viewpoint.json`, `selection_D_t1_style.json`, `selection_D_t1_viewpoint.json`, `selection_E_all.json`, `selection_E_random.json`, `training_subset.json` |
| `results/sdxl_cityscapes_fewshot10/A_t81_style/` | 8 | 2,434,595 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json`, `loss_curve.png`, `stability.json`, `train.log`, `train_summary.json` |
| `results/sdxl_cityscapes_fewshot10/A_t81_viewpoint/` | 8 | 2,353,110 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json`, `loss_curve.png`, `stability.json`, `train.log`, `train_summary.json` |
| `results/sdxl_cityscapes_fewshot10/B_top3_style/` | 8 | 2,357,459 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json`, `loss_curve.png`, `stability.json`, `train.log`, `train_summary.json` |
| `results/sdxl_cityscapes_fewshot10/B_top3_viewpoint/` | 8 | 2,370,623 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json`, `loss_curve.png`, `stability.json`, `train.log`, `train_summary.json` |
| `results/sdxl_cityscapes_fewshot10/C_t41_style/` | 8 | 2,396,867 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json`, `loss_curve.png`, `stability.json`, `train.log`, `train_summary.json` |
| `results/sdxl_cityscapes_fewshot10/C_t41_viewpoint/` | 8 | 2,420,658 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json`, `loss_curve.png`, `stability.json`, `train.log`, `train_summary.json` |
| `results/sdxl_cityscapes_fewshot10/D_t1_style/` | 8 | 2,371,264 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json`, `loss_curve.png`, `stability.json`, `train.log`, `train_summary.json` |
| `results/sdxl_cityscapes_fewshot10/D_t1_viewpoint/` | 8 | 2,368,301 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json`, `loss_curve.png`, `stability.json`, `train.log`, `train_summary.json` |
| `results/sdxl_cityscapes_fewshot10/E_all/` | 8 | 5,160,842 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json`, `loss_curve.png`, `stability.json`, `train.log`, `train_summary.json` |
| `results/sdxl_cityscapes_fewshot10/E_random/` | 8 | 2,409,503 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json`, `loss_curve.png`, `stability.json`, `train.log`, `train_summary.json` |
| `results/sdxl_cityscapes_fewshot10/control_base/` | 4 | 907,709 | `eval.json`, `eval.log`, `generate.log`, `generation_manifest.json` |
| `results/sdxl_cityscapes_fewshot10/tensor_core_gate/` | 3 | 57,337 | `drift_autocast_gate_viewpoint.json`, `gate.log`, `tensor_core_gate.json` |

100 files, 28,610,026 bytes (27.3 MiB).

## Large artifacts (Hugging Face)

The following artifacts back the same results but are far too large for git. They are published on
the Hugging Face Hub, in two public repositories:

- **Checkpoints (model repo):** <https://huggingface.co/chs35/improving-ca-lora-checkpoints>
- **Measurements + generated images (dataset repo):** <https://huggingface.co/datasets/chs35/improving-ca-lora-artifacts>

| Artifact | What it is | Files | Size | Location |
| --- | --- | ---: | ---: | --- |
| LoRA checkpoints | All 10 trained arms of the **few-shot experiment** (`results/sdxl_cityscapes_fewshot10/`; replaced 2026-11-04, the earlier full-data adapters remain at Hub revision `db005e2c124b6666c27b8a788cd5a1f6c1f3f080`): the final step-10000 adapter used to generate the images scored in Table 3, plus the intermediate checkpoints at steps 2000/4000/6000/8000 (50 `.safetensors` files in total) | 50 | 3,357,944,280 B (3.36 GB / 3.13 GiB) | [model repo](https://huggingface.co/chs35/improving-ca-lora-checkpoints), `<arm>/lora-{step002000,step004000,step006000,step008000,final}.safetensors` |
| Module-granularity pilot checkpoints | An undocumented earlier pilot of arm `A_t81` at *module* granularity, retained only as a granularity-ablation data point. No number in the report comes from it | 5 | 35,077,048 B (35.1 MB / 33.5 MiB) | [model repo](https://huggingface.co/chs35/improving-ca-lora-checkpoints), `pilot_module_granularity/A_t81/` |
| Head CA measurement tensors | The raw head-granularity CA measurements for the 13-timestep sweep, 13 timesteps x {full, half0, half1} = 39 `.safetensors` files. The `half0`/`half1` splits are what the split-half agreement in Table 2 is computed from | 39 | 190,967,712 B (191 MB / 182 MiB) | [dataset repo](https://huggingface.co/datasets/chs35/improving-ca-lora-artifacts), `measurements/tsweep_sdxl_head/` |
| Generated images | From the **earlier full-data experiment** (`results/sdxl_cityscapes/`), not from the current adapters: 11 arms x 900 PNG at 1024x1024 (500 in-domain + 4 weather conditions x 100), the exact images scored for CMMD and CLIPScore in Table 3, plus each arm's `manifest.json` | 9,911 (9,900 PNG + 11 `manifest.json`) | 12,037,555,772 B (12.04 GB / 11.21 GiB) | [dataset repo](https://huggingface.co/datasets/chs35/improving-ca-lora-artifacts), `generated/<arm>/` |

**15,621,544,812 bytes (15.62 GB / 14.55 GiB) across 10,005 files** on the Hub. Every file was
verified after upload against the manifests below: the Hub's own `lfs.sha256` for the 9,994 LFS
objects and the git blob id for the 11 `manifest.json` files, 10,005 / 10,005 matching, 0 mismatches.

Full-length sha256 for all of it, keyed by the same repo-relative paths the Hub uses:

| Manifest | Covers | Format |
| --- | --- | --- |
| [`sha256_checkpoints.txt`](sha256_checkpoints.txt) | all 55 checkpoint files (50 trained + 5 pilot) | `<sha256>  <path>` |
| [`sha256_measurements.txt`](sha256_measurements.txt) | the 39 CA measurement tensors | `<sha256>  <path>` |
| [`sha256_generated_images.csv`](sha256_generated_images.csv) | all 9,911 generated-image files | `path,bytes,sha256` |

To verify a download, e.g. for the checkpoints:

```bash
hf download chs35/improving-ca-lora-checkpoints --local-dir ckpt
cd ckpt && sha256sum -c ../results/sha256_checkpoints.txt
```

Each generated-image set is paired with the `generation_manifest.json` already released in this
tree (identical to the `manifest.json` uploaded alongside the images), which records the prompt,
condition and seed behind every individual PNG -- so per-image scores can be recomputed once the
images are downloaded.

### Not released: the Cityscapes reference crops

The CMMD reference statistics were computed from **500 centre crops of the Cityscapes `val`
split**. Those crops are *not* released, in git or on the Hub: the Cityscapes license forbids
redistribution of the imagery. Reproducing the CMMD column of Table 3 therefore requires obtaining
Cityscapes independently from <https://www.cityscapes-dataset.com/> and regenerating the crops with
the project's own exporter, `scripts/evaluate.py::export_reference`, which takes the first 500
images of the `val` split in index order at the model resolution of 1024x1024, centre-cropped and
un-mirrored -- the same framing the training data was read with, so that CMMD measures the gap in
content rather than a difference in cropping. The relevant knobs (`eval.reference_split`,
`eval.reference_num_images`, `eval.seed`) are recorded in the released `config.yaml`, and the
resulting protocol is echoed into every `eval.json`. Every other number in the report is
reproducible from the artifacts listed above without access to Cityscapes.

## Deliberate omissions

`runs/sdxl_cityscapes/timestep_scores.json` and `runs/sdxl_cityscapes/measure_summary.json` exist
upstream but were **deliberately not released**. They are leftovers from an early *module*-level
pilot and are superseded by the head-level sweep in `results/tsweep_sdxl_head/`, which is the actual
source of Table 2. Releasing both under similar names would invite a reader to quote the wrong file.
The small module-level selection files from that same pilot
(`selection_A_t81.json`, `selection_B_top3.json`, `selection_B_top5.json`, `selection_C_t41.json`,
`selection_D_t1.json`, `selection_D_t481.json`, `selection_D_t601.json`) are omitted for the same
reason -- the released `selection_*_{style,viewpoint}.json` files are the head-level ones that were
actually trained.


## Total released size

**192 files, 45,415,079 bytes (43.31 MiB) tracked in git** (92 files from the earlier experiment, 100 from the few-shot experiment added 2026-11-04).
Excluding the 15.62 GB of checkpoints, measurement tensors and generated images that are distributed
via Hugging Face rather than git.
