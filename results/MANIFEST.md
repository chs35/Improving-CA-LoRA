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


## Large artifacts (Hugging Face)

The following artifacts back the same results but are far too large for git. They will be uploaded
to a Hugging Face repository; the URLs are recorded here once upload completes.

| Artifact | What it is | Approx. size | Location |
| --- | --- | ---: | --- |
| LoRA checkpoints | All 10 trained arms: the final step-10000 adapter used to generate the images scored in Table 3, plus the intermediate checkpoints at steps 2000/4000/6000/8000 (50 `.safetensors` files in total) | 3.36 GB (3.13 GiB) | (URL recorded after upload) |
| Head CA measurement tensors | The raw head-granularity CA measurements for the 13-timestep sweep, 13 timesteps x {full, half0, half1} = 39 `.safetensors` files. The `half0`/`half1` splits are what the split-half agreement in Table 2 is computed from | 191 MB (182 MiB) | (URL recorded after upload) |
| Generated images | 11 arms x 900 PNG at 1024x1024 (500 in-domain + 4 weather conditions x 100), the exact images scored for CMMD and CLIPScore in Table 3 | 12.0 GB (11.2 GiB) | (URL recorded after upload) |

Each generated-image set is paired with the `generation_manifest.json` already released in this
tree, which records the prompt, condition and seed behind every individual PNG -- so per-image
scores can be recomputed once the images are downloaded.

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

**92 files, 16,805,053 bytes (16.03 MiB) tracked in git.**
Excluding the ~15.5 GB of checkpoints, measurement tensors and generated images that are distributed
via Hugging Face rather than git.
