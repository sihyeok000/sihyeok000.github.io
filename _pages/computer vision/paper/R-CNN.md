---
title: "R-CNN : Rich feature hierarchies for accurate object detection and semantic segmentation"
tags:
    - computer vision
    - deep learning
    - CNN
    - object detection

date: "2024-05-12"
thumbnail: "/assets/img/thumbnail/rcnn.jpg"
bookmark: true
---

# R-CNN: 정확한 객체 탐지 및 분할을 위한 풍부한 특징 계층 구조

Object Detection 분야에 혁신을 가져온 R-CNN(Regions with CNN features) 논문을 핵심 위주로 정리하고, 그 원리를 알기 쉽게 설명합니다.

---

## Abstract (초록)

> 이 논문은 기존의 Object Detection 성능을 크게 뛰어넘는, 간단하면서도 확장 가능한 새로운 알고리즘을 제안합니다. 이전까지 PASCAL VOC 데이터셋에서 최고 성능은 low-level 이미지 특징과 high-level 컨텍스트를 결합한 복잡한 앙상블 모델이었습니다.

R-CNN은 두 가지 핵심 아이디어를 결합하여 mAP(mean Average Precision)를 30% 이상 향상시켰습니다.

1.  **Regions with CNNs:** 객체의 위치를 찾고 분할하기 위해, 먼저 이미지에서 객체가 있을 법한 영역(Region Proposal)을 찾고, 이 영역들에 CNN(Convolutional Neural Network)을 적용합니다.
2.  **Pre-training & Fine-tuning:** 라벨링된 데이터가 부족한 문제를 해결하기 위해, 대규모 데이터셋(ILSVRC)으로 미리 학습(Supervised Pre-training)시킨 후, 특정 도메인의 데이터셋(PASCAL)에 맞게 미세 조정(Fine-tuning)하여 성능을 극대화합니다.

이 접근법은 당시 최고의 모델이었던 OverFeat을 큰 차이로 능가하며 Object Detection의 새로운 패러다임을 열었습니다.

---

## 1. Introduction (서론)

과거의 시각 인식(Visual Recognition) 기술은 주로 `SIFT`, `HOG`와 같은 특징 추출 방식에 의존했습니다. 이는 인간의 시각 시스템으로 치면 초기 단계인 V1 피질의 기능과 유사합니다. 하지만 실제 사물 인식은 그보다 더 복잡한 과정을 거치므로, 더 나은 특징을 학습할 필요가 있었습니다.

CNN은 이러한 문제의 해결책으로 떠올랐지만, 1990년대 이후 SVM의 등장으로 주류에서 밀려났습니다. 그러나 2012년, AlexNet이 ILSVRC에서 압도적인 성능을 보여주면서 CNN은 다시 주목받기 시작했습니다.

이 논문은 CNN을 Object Detection에 효과적으로 적용하기 위한 두 가지 핵심 질문에 집중합니다.

* **Deep Network를 사용하여 객체의 위치를 어떻게 특정할 것인가? (Localization)**
* **양이 적은 데이터만으로 어떻게 대용량 모델을 훈련시킬 것인가? (Training)**

---

## 2. R-CNN을 이용한 객체 탐지 (Object detection with R-CNN)

R-CNN 시스템은 세 가지 주요 모듈로 구성됩니다.

![R-CNN 모델 구조](https://imgur.com/GkhUs94)

1.  **Region Proposal (영역 제안):** 먼저, 이미지에서 객체가 있을 만한 위치를 약 2000개 정도 찾아냅니다. 이 논문에서는 **Selective Search** 알고리즘을 사용합니다. 이 단계에서는 객체의 종류(class)는 고려하지 않고, 오직 "여기에 무언가 있을 것 같다"는 후보 영역만 빠르게 추출합니다.

2.  **Feature Extraction (특징 추출):** 제안된 2000개의 영역들을 모두 동일한 크기(227x227 픽셀)로 변형(Warping)시킨 후, 미리 학습된 대규모 CNN 모델에 입력하여 각 영역으로부터 4096차원의 고정 길이 특징 벡터(feature vector)를 추출합니다.

3.  **Classification (분류):** 마지막으로, 추출된 특징 벡터를 사용하여 각 영역이 어떤 객체 클래스에 속하는지, 또는 배경인지를 **선형 SVM(Support Vector Machines)** 분류기를 통해 판별합니다.

### 훈련 과정의 핵심: Pre-training과 Fine-tuning

R-CNN의 높은 성능은 효과적인 훈련 전략 덕분입니다.

-   **Supervised Pre-training:** 먼저, 대규모 이미지 분류 데이터셋인 `ILSVRC 2012`를 사용하여 CNN 모델을 충분히 학습시킵니다. 이 과정을 통해 CNN은 이미지의 기본적인 특징(선, 질감, 색상 등)을 풍부하게 학습하게 됩니다.

-   **Domain-specific Fine-tuning:** 그 후, 우리가 실제로 사용하려는 `PASCAL VOC` 데이터셋에 맞게 CNN 모델을 미세 조정합니다. 이 과정에서는 마지막 분류 레이어를 기존 1000개에서 `(객체 클래스 수 + 배경 1개)`로 교체하고, 더 낮은 학습률(learning rate)로 모델을 추가 학습시켜 특정 도메인에 대한 성능을 끌어올립니다.

> **💡 IoU (Intersection-over-Union)란?**
> Fine-tuning과 SVM 훈련 시, 어떤 영역이 '정답'이고 어떤 영역이 '오답'인지 판단하는 기준이 필요합니다. 이때 IoU가 사용됩니다. IoU는 **실제 정답 영역(Ground-truth box)과 모델이 제안한 영역(Proposed region)이 얼마나 겹치는지를 나타내는 지표**입니다.
> ![IoU 계산법](https://imgur.com/1IwDnR3)

> R-CNN에서는 이 IoU 값이 특정 임계값(예: 0.5) 이상이면 '정답'(Positive), 그보다 훨씬 낮으면 '오답'(Negative)으로 간주하여 모델을 학습시킵니다.

### Bounding-box Regression

R-CNN은 Selective Search가 제안한 영역의 위치가 완벽하지 않을 수 있다는 점을 보완하기 위해, **Bounding-box regression**을 추가로 사용합니다. 이는 CNN을 통해 얻은 특징을 바탕으로, 제안된 영역의 위치와 크기를 실제 객체에 더 가깝게 미세 조정하는 선형 회귀 모델입니다. 이 과정을 통해 탐지 정확도를 3~4%p 추가로 향상시켰습니다.

---

## 3. 결론 (Conclusion)

R-CNN은 두 가지 핵심 아이디어를 통해 Object Detection 분야에 큰 발전을 이루었습니다.

1.  **고전적인 컴퓨터 비전 기법(Region Proposals)과 딥러닝(CNN)의 성공적인 결합**을 보여주었습니다.
2.  대규모 데이터셋으로 **사전 학습(Pre-training) 후, 특정 도메인에 맞게 미세 조정(Fine-tuning)**하는 패러다임이 데이터가 부족한 환경에서도 매우 효과적임을 입증했습니다.

이러한 접근법은 이후 등장하는 Fast R-CNN, Faster R-CNN 등 더 발전된 모델들의 기반이 되었습니다.