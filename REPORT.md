# Improving CA-LoRA
정희승 (Heeseung Chung), 2026년 11월.

## 1. 연구 배경

Concept-Aware LoRA(CA-LoRA)는 text-to-image 확산 모델을 목표 도메인(Cityscapes)에 맞추면서, 학습 데이터에 없는 장면(예: 악천후)을 프롬프트로 만들어 내는 사전학습 모델의 능력은 남기려는 방법이다. 모델 전체를 학습하지 않고, 도메인에 맞추려는 개념(style 또는 viewpoint)에 민감한 attention head만 LoRA로 학습한다.

head $h$가 한 개념에 얼마나 민감한지는 concept awareness(CA)로 잰다.

$$CA_t(h) = E\left[\frac{||\nabla_{\theta_h} L_{Concept}||}{||\nabla_{\theta_h} L_{Diff}||}\right]$$

$L_{Concept}$는 프롬프트에서 그 개념의 표현만 바꿨을 때 모델의 노이즈 예측이 달라지는 크기이고, $L_{Diff}$는 표준 diffusion loss다. $\theta_h$는 head $h$의 가중치다. SDXL의 attention에는 이런 head 단위가 10,400개 있고, CA-LoRA는 CA 상위 2%인 208개만 학습한다. 두 gradient는 확산 timestep $t$ 하나에서 계산하므로, 어떤 head가 선택되는지는 $t$가 정한다.

선택은 $t$에 민감하다. Table 1은 13개 $t$에서 CA를 재고, 각 $t$의 선택이 논문이 쓴 $t = 81$의 선택과 몇 개나 겹치는지 센 것이다. style 개념에서 $t = 81$이 고른 208개 가운데 $t = 1$에서도 선택되는 head는 62개, $t = 481$에서는 79개이고, $t$가 601 이상이면 32개 이하다. viewpoint 개념에서는 각각 133개, 117개, 87개 이하다.

두 개념이 서로 다른 head를 고르는 정도도 $t$에 따라 변한다. 같은 $t$에서 style과 viewpoint가 함께 고르는 head는 $t = 201$에서 7개이고, $t = 1$에서는 95개, $t = 981$에서는 97개다. 뒤의 두 경우에는 선택의 절반 가까이가 두 개념에 공통이다.

| $t$ | 1 | 41 | 81 | 121 | 201 | 301 | 401 | 481 | 601 | 701 | 801 | 901 | 981 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| style: $t = 81$ 선택과 겹침 | 62 | 191 | 208 | 197 | 178 | 155 | 129 | 79 | 32 | 17 | 13 | 12 | 14 |
| viewpoint: $t = 81$ 선택과 겹침 | 133 | 188 | 208 | 189 | 174 | 156 | 136 | 117 | 87 | 42 | 28 | 27 | 25 |
| 같은 $t$의 style과 viewpoint 겹침 | 95 | 22 | 17 | 11 | 7 | 13 | 27 | 47 | 90 | 96 | 61 | 57 | 97 |

Table 1. $t$에 따른 선택의 변화. 값은 선택된 head 208개 가운데 겹치는 개수다. CA는 논문의 절차대로 base SDXL이 직접 생성한 이미지 256장에서 쟀다.

논문은 후보 {1, 81, 201, 481}의 CA 분포를 눈으로 비교해 $t = 81$을 골랐고, $t$를 고르는 정량 기준은 제시하지 않았다. 이 보고서는 그 기준을 제안하고, 그 기준으로 고른 $t$가 학습 결과를 어떻게 바꾸는지 확인한다.

## 2. 방법론

제안하는 기준 ews(expanded weighted similarity)는 CA가 높은 head들에서 style과 viewpoint의 gradient가 얼마나 같은 방향을 가리키는지를 잰 값이다. ews가 가장 낮은 $t$를 고른다.

CA는 gradient norm의 비율이라 gradient의 방향을 버린다. 어떤 head에서 style의 gradient와 viewpoint의 gradient가 같은 방향이면, 그 head는 CA가 높더라도 두 개념을 구분하지 못한다. 이런 head를 style에 민감하다는 이유로 골라 학습하면 viewpoint도 함께 움직인다. ews는 CA가 높은 head들이 두 개념을 서로 다른 방향으로 움직이는 $t$가 좋은 $t$라는 가정에 선다.

계산은 다음과 같다. 개념 $a$의 gradient를 CA와 같은 방식으로 정규화하고,

$$\hat{g}_a(h) = \frac{\nabla_{\theta_h} L^{(a)}_{Concept}}{||\nabla_{\theta_h} L_{Diff}||}, \quad a \in \{style, viewpoint\}$$

head마다 두 개념의 평균 gradient 사이 코사인 유사도 $s_t(h) = \cos(E[\hat{g}_{style}(h)],\ E[\hat{g}_{viewpoint}(h)])$를 구한다. ews는 이 유사도를 CA로 가중 평균한 값이다.

$$ews(t) = \sum_{h \in S_t} w_t(h)\, s_t(h), \qquad w_t(h) = \frac{CA_t(h)}{\sum_{h' \in S_t} CA_t(h')}$$

$S_t$는 $t$에서 style CA가 가장 높은 head 1,040개이고, 가중치의 $CA_t$도 style의 CA다. 실제로 학습하는 208개의 5배로 잡은 것은, 208번째 근처의 순위가 조금 바뀌어도 값이 흔들리지 않게 하기 위해서다. 두 목적의 gradient 사이 코사인을 충돌 신호로 쓰는 것은 다중 과제 학습에서 쓰이던 방법이고, ews는 이를 $t$를 고르는 데 쓴다.

ews가 낮은 $t$ 여러 개를 함께 쓰는 선택도 만들었다. ews가 가장 낮은 세 $t$에서 CA를 각각 합이 1이 되도록 정규화해 평균하고, 그 평균이 가장 높은 208개를 고른다. 아래에서는 이를 집계 선택이라 부른다.

## 3. 결과

비교 대상은 논문이 쓴 $t = 81$의 선택이다. ews로 만든 선택은 ews가 가장 낮은 $t$ 하나의 선택과, 가장 낮은 세 $t$의 집계 선택 둘이다. 세 선택을 각각 style의 CA와 viewpoint의 CA로 만들어 어댑터 여섯 개를 학습했다. 아래에서 style 어댑터는 style의 CA로 고른 head를 학습한 LoRA를, viewpoint 어댑터는 viewpoint의 CA로 고른 head를 학습한 LoRA를 가리킨다. 모든 어댑터는 Cityscapes train에서 무작위로 뽑은 같은 10장으로 학습했다.

지표는 둘이다. CMMD는 생성 이미지 500장과 Cityscapes val 500장 사이의 CLIP 임베딩 분포 거리이고, 낮을수록 도메인에 잘 맞는다. 날씨 점수는 프롬프트에 foggy, night-time, rainy, snowy를 넣어 100장씩 생성한 이미지의 CLIP-Score 평균이고, 높을수록 프롬프트의 날씨가 이미지에 남아 있다.

ews는 $t = 201$을 고른다. 13개 $t$ 가운데 ews는 $t = 201$에서 0.178로 가장 낮고, 그다음이 $t = 301$(0.215), $t = 121$(0.233)이다. 논문의 $t = 81$은 0.258로 네 번째다. 논문의 나머지 후보인 $t = 481$은 0.287, $t = 1$은 0.621이고, 가장 높은 $t = 981$은 0.870이다. 집계 선택은 [201, 301, 121]을 쓴다.

| 학습한 head | 개념 | CMMD ↓ | 날씨 점수 ↑ |
|---|---|---|---|
| $t = 81$ (논문) | style | 1.421 | 21.43 |
| $t = 201$ (ews 최저) | style | 1.584 | 19.13 |
| 집계 [201, 301, 121] | style | 1.562 | 18.43 |
| $t = 81$ (논문) | viewpoint | 1.439 | 24.95 |
| $t = 201$ (ews 최저) | viewpoint | 1.503 | 26.60 |
| 집계 [201, 301, 121] | viewpoint | 1.703 | 27.51 |
| 무작위 208개 | 없음 | 0.748 | 24.03 |
| 전체 10,400개 | 없음 | 1.437 | 16.61 |
| 학습 없음 (base SDXL) | 없음 | 4.075 | 31.68 |

Table 2. 학습 결과. 위 여섯 행은 CA로 고른 208개 head를 학습한 어댑터이고, 아래 세 행은 CA를 쓰지 않는 기준이다. 모든 행은 학습 seed 하나의 결과다. 본문의 차이는 반올림하기 전 값으로 계산했다.

viewpoint 어댑터는 ews로 고른 $t$에서 날씨 점수가 올랐고 CMMD는 나빠졌다. 날씨 점수는 $t = 81$의 24.95에서 $t = 201$의 26.60으로 1.65 올랐고, 집계 선택에서는 27.51로 2.55 올랐다. CMMD는 $t = 81$의 1.439에서 $t = 201$의 1.503으로 0.06, 집계 선택의 1.703으로 0.26 높아졌다.

style 어댑터는 ews로 고른 $t$에서 두 지표 모두 나빠졌다. CMMD는 $t = 81$의 1.421에서 $t = 201$의 1.584, 집계 선택의 1.562로 높아졌다. 날씨 점수는 21.43에서 각각 19.13, 18.43으로 2.30, 3.01 내려갔다.

ews로 고른 $t$에서는 두 개념의 어댑터가 날씨 프롬프트에 더 다르게 반응한다. viewpoint 어댑터와 style 어댑터의 날씨 점수 차이는 $t = 81$에서 3.52이고, $t = 201$에서 7.47, 집계 선택에서 9.08이다. 이 차이는 viewpoint 어댑터의 상승과 style 어댑터의 하락이 합쳐진 것이다.

학습 결과도 $t$에 민감하다. $t = 81$과 $t = 201$의 선택은 208개 가운데 style에서 178개, viewpoint에서 174개가 같다. head의 84-86%가 같은데도 날씨 점수는 style 어댑터에서 2.30, viewpoint 어댑터에서 1.65 다르다.

CMMD의 차이는 seed 하나로는 선택의 효과라고 말하기 어렵다. $t = 201$의 선택과 집계 선택은 style에서 203개, viewpoint에서 196개가 같아 거의 같은 head를 학습한다. 이 두 선택 사이에서도 CMMD는 style 어댑터에서 0.02, viewpoint 어댑터에서 0.20 다르고, 날씨 점수는 0.71, 0.90 다르다. $t = 81$과 ews로 고른 $t$ 사이의 CMMD 차이는 집계 선택의 viewpoint 어댑터(0.26)를 빼면 0.06-0.16으로 이 폭 안에 있다. 날씨 점수의 차이는 같은 개념끼리 비교하면 style 어댑터에서 이 폭의 3.2-4.2배, viewpoint 어댑터에서 1.8-2.8배다.

CA를 쓰지 않는 기준과 비교하면 다음과 같다. 학습하지 않은 base SDXL은 날씨 점수가 31.68로 가장 높고 CMMD가 4.075로 가장 나쁘다. 어떤 어댑터든 도메인에 맞추는 대가로 날씨 점수를 잃는다. head 208개를 무작위로 골라 학습한 어댑터는 CMMD가 0.748로, CA로 고른 여섯 어댑터 모두보다 낮다. 무작위 어댑터의 날씨 점수 24.03은 style 어댑터 셋(18.43-21.43)보다 높고 viewpoint 어댑터 셋(24.95-27.51)보다 낮다. head 전체를 학습한 어댑터는 날씨 점수가 16.61로 가장 낮다. viewpoint 어댑터가 무작위 선택보다 날씨 점수를 더 지키는 폭은 $t = 81$에서 0.93이고, $t = 201$에서 2.58, 집계 선택에서 3.48이다.

## 참고문헌

- M. Park et al. "Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation." CVPR 2026. arXiv:2503.22172. (CA-LoRA)
- D. Podell et al. "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." ICLR 2024. arXiv:2307.01952.
- M. Cordts et al. "The Cityscapes Dataset for Semantic Urban Scene Understanding." CVPR 2016. arXiv:1604.01685.
- E. J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685.
- T. Yu et al. "Gradient Surgery for Multi-Task Learning." NeurIPS 2020. arXiv:2001.06782. (gradient 코사인을 충돌 신호로 쓰는 방법)
- S. Jayasumana et al. "Rethinking FID: Towards a Better Evaluation Metric for Image Generation." CVPR 2024. arXiv:2401.09603. (CMMD)
- J. Hessel et al. "CLIPScore: A Reference-free Evaluation Metric for Image Captioning." EMNLP 2021. arXiv:2104.08718.
