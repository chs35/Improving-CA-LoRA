# Improving CA-LoRA

[English](README.md) | **한국어**

CA-LoRA를 SDXL + Cityscapes에서 재구현하고, head 선택을 재는 timestep을 정량 기준으로 고르는 방법을 제안한다.

CA-LoRA(Concept-Aware LoRA, CVPR 2026, [arXiv:2503.22172](https://arxiv.org/abs/2503.22172))는 한 개념(style 또는 viewpoint)에 민감한 attention head만 학습해, 모델을 목표 도메인에 맞추면서 학습 데이터에 없는 프롬프트를 따르는 능력은 남기려는 방법이다. 이 민감도(CA)는 확산 timestep 하나에서 재는데, 어떤 head가 선택되는지는 그 timestep에 따라 크게 달라진다. 논문은 t = 81을 눈으로 골랐고 정량 기준은 제시하지 않았다.

이 레포는 공식 코드가 공개되지 않은 CA-LoRA를 논문만 보고 재구현하고, 그 기준으로 **ews**를 제안한다. ews는 CA가 높은 head들에서 style과 viewpoint의 gradient 사이 코사인 유사도를 CA로 가중 평균한 값이고, ews가 가장 낮은 timestep을 고른다.

ews는 t = 201을 고른다. 같은 Cityscapes 10장으로 학습해(seed 하나) 논문의 t = 81과 비교하면, viewpoint 어댑터는 날씨 프롬프트 제어성을 더 지켰지만 도메인 적합도는 나빠졌고(CMMD 상승), style 어댑터는 두 지표 모두 나빠졌다. CMMD 차이는 대부분 거의 같은 두 선택 사이의 차이보다 크지 않고, 같은 개수를 무작위로 고른 선택이 CA로 고른 모든 선택보다 CMMD가 낮다. 요약하면 ews로 고른 timestep이 t = 81보다 일관되게 낫지는 않았다.

자세한 내용은 **[REPORT.md](REPORT.md)** 에 있다.

## 실행

```bash
pip install -e ".[diffusion,dev]"
python -m pytest tests/ -q                # CPU만 사용, toy 모델, 다운로드 없음

# 최종 실험 (A100 80GB 1대). 먼저 configs/sdxl_cityscapes.yaml의 data.root를 설정한다.
CA=configs/ca_selfgen.yaml; FT=configs/sdxl_cityscapes_selfgen_fewshot10.yaml
python scripts/generate_selfgen.py   --config $CA     # base SDXL로 이미지 256장 생성
python scripts/measure_ca.py         --config $CA --set measure.precision=bf16
python scripts/score_timesteps.py    --config $CA --timesteps 1 41 81 121 201 301 401 481 601 701 801 901 981
python scripts/select_projections.py --config $CA --strategy single --timesteps 201 --set select.axis=style --out selection_C_t201_style.json
python scripts/train_lora.py         --config $FT --run-name C_t201_style --selection runs/ca_selfgen/selection_C_t201_style.json
python scripts/generate_images.py    --config $FT --run-name C_t201_style --mode both --lora runs/sdxl_cityscapes_selfgen_fewshot10/C_t201_style/lora-final.safetensors
python scripts/evaluate.py           --config $FT --run-name C_t201_style
```

Cityscapes는 포함하지 않으며 [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)에서 별도로 받아야 한다. 마지막 네 명령은 어댑터마다 반복한다. 서버에서 실험을 실제로 돌린 스크립트는 `scripts/ops/`에 있다.

## 폴더 구조

```
configs/      실험 설정 (ca_selfgen.yaml: CA 측정, sdxl_cityscapes_selfgen_fewshot10.yaml: 학습)
scripts/      파이프라인 단계별 스크립트. scripts/ops/ = 최종 실험을 돌린 서버용 실행 스크립트
src/calora/   CA 측정과 head 선택, sliced LoRA, 학습, 평가 (CMMD, CLIP-Score)
tests/        toy UNet으로 도는 CPU 단위 테스트
results/      보고서 표의 원본 결과
REPORT.md     연구 보고서
```

`results/ca_selfgen/`에는 CA 측정 요약, ews 점수(`timestep_scores.json`), 선택이 있고, `results/sdxl_cityscapes_selfgen_fewshot10/`에는 CA로 고른 어댑터 여섯 개의 결과가 있다. 보고서의 기준 셋(무작위 head, 전체 head, base SDXL)은 CA와 무관하다. 같은 10장으로 학습한 이전 실행의 결과이고 `results/sdxl_cityscapes_fewshot10/`에 있다.

## 관련 레포

**Improving-CA-LoRA**(CA 측정 timestep을 정량 기준으로 고르기) → [CA-LoRA-TSM](https://github.com/chs35/CA-LoRA-TSM)(timestep 하나에 의존하지 않기) → [Hierarchical-CFG](https://github.com/chs35/Hierarchical-CFG)(샘플링 단계에서 제어성 되찾기).

## 라이선스

[MIT](LICENSE). SDXL 가중치와 Cityscapes 이미지는 포함하지 않으며 각자의 라이선스를 따른다.
