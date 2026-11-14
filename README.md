# Improving CA-LoRA

**English** | [한국어](README.ko.md)

A reimplementation of CA-LoRA on SDXL + Cityscapes, and a quantitative rule for choosing the timestep its head selection is measured at.

CA-LoRA (Concept-Aware LoRA, CVPR 2026, [arXiv:2503.22172](https://arxiv.org/abs/2503.22172)) fine-tunes only the attention heads that are sensitive to one concept (style or viewpoint), so the model fits a target domain while keeping its ability to follow prompts the training data never covered. That sensitivity (CA) is measured at a single diffusion timestep, and the selected heads change a lot with it; the paper picked t = 81 by eye and gave no quantitative rule.

This repository reimplements CA-LoRA from the paper alone (no official code has been released) and proposes such a rule, **ews**: the CA-weighted cosine similarity between the style and viewpoint gradients on the high-CA heads, with the lowest-ews timestep chosen.

ews picks t = 201. Against the paper's t = 81, with adapters trained on the same 10 Cityscapes images (one seed), the viewpoint adapter kept more weather-prompt controllability but fit the domain worse (higher CMMD), and the style adapter got worse on both metrics. Most of the CMMD gaps are no larger than the gap between two nearly identical selections, and a random selection of the same size reaches a lower CMMD than every CA-based one. In short, the ews-chosen timestep was not uniformly better than t = 81.

The full report is in Korean: **[REPORT.md](REPORT.md)**.

## Run

```bash
pip install -e ".[diffusion,dev]"
python -m pytest tests/ -q                # CPU only, toy model, no downloads

# Final experiment (one A100 80GB). Set data.root in configs/sdxl_cityscapes.yaml first.
CA=configs/ca_selfgen.yaml; FT=configs/sdxl_cityscapes_selfgen_fewshot10.yaml
python scripts/generate_selfgen.py   --config $CA     # 256 images sampled from base SDXL
python scripts/measure_ca.py         --config $CA --set measure.precision=bf16
python scripts/score_timesteps.py    --config $CA --timesteps 1 41 81 121 201 301 401 481 601 701 801 901 981
python scripts/select_projections.py --config $CA --strategy single --timesteps 201 --set select.axis=style --out selection_C_t201_style.json
python scripts/train_lora.py         --config $FT --run-name C_t201_style --selection runs/ca_selfgen/selection_C_t201_style.json
python scripts/generate_images.py    --config $FT --run-name C_t201_style --mode both --lora runs/sdxl_cityscapes_selfgen_fewshot10/C_t201_style/lora-final.safetensors
python scripts/evaluate.py           --config $FT --run-name C_t201_style
```

Cityscapes is not included and must be downloaded separately from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/). The last four commands are repeated for each adapter; `scripts/ops/` holds the scripts that actually ran the experiment on the server.

## Layout

```
configs/      experiment configs (ca_selfgen.yaml: CA measurement, sdxl_cityscapes_selfgen_fewshot10.yaml: training)
scripts/      one script per pipeline stage; scripts/ops/ = server-side runners of the final experiment
src/calora/   CA measurement and head selection, sliced LoRA, training, evaluation (CMMD, CLIP-Score)
tests/        CPU unit tests on a toy UNet
results/      raw outputs behind the report's tables
REPORT.md     research report (Korean)
```

`results/ca_selfgen/` has the CA measurement summary, the ews scores (`timestep_scores.json`) and the selections; `results/sdxl_cityscapes_selfgen_fewshot10/` has the six CA-selected adapters. The three baselines in the report (random heads, all heads, base SDXL) do not depend on CA; they come from an earlier run on the same 10 images and are in `results/sdxl_cityscapes_fewshot10/`.

## Related repositories

**Improving-CA-LoRA** (choosing the CA timestep with a quantitative criterion) → [CA-LoRA-TSM](https://github.com/chs35/CA-LoRA-TSM) (not relying on one timestep) → [Hierarchical-CFG](https://github.com/chs35/Hierarchical-CFG) (restoring controllability at sampling time).

## License

[MIT](LICENSE). SDXL weights and Cityscapes images are not included and keep their own licenses.
