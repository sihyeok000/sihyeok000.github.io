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

Object Detection 분야에 혁신을 가져온 R-CNN(Regions with CNN features) 논문의 핵심을 위주로 정리하고, 그 원리를 설명한다.

---

## R-CNN, Object Detection의 게임 체인저

R-CNN은 두 가지 핵심 아이디어를 결합하여 당시의 성능 지표였던 mAP(mean Average Precision)를 30% 이상 끌어올리는 기염을 토했습니다.

1.  **Regions with CNNs:** 객체의 위치를 찾고 분할하기 위해, 먼저 이미지에서 객체가 있을 법한 영역(Region Proposal)을 찾고, 이 영역들에 CNN(Convolutional Neural Network)을 적용한다.
2.  **Pre-training & Fine-tuning:** 라벨링된 데이터가 부족한 문제를 해결하기 위해, 대규모 데이터셋(ILSVRC)으로 미리 학습(Supervised Pre-training)시킨 후, 특정 도메인의 데이터셋(PASCAL)에 맞게 미세 조정(Fine-tuning)하여 성능을 극대화한다.

이 접근법은 당시 최고의 모델이었던 OverFeat을 큰 차이로 능가하며 Object Detection의 새로운 패러다임을 열었습니다.

---

## 왜 R-CNN이 필요했을까?

과거의 시각 인식(Visual Recognition) 기술은 주로 `SIFT`, `HOG`와 같은 고정된 특징 추출 방식에 의존했습니다. 이는 이미지의 밝기, 형태 등 단순한 정보를 기반으로 했기에 복잡한 실제 환경의 객체를 인식하는 데 한계가 있었습니다.

CNN은 이러한 문제의 해결책으로 떠올랐지만, 1990년대 이후 한동안 주류에서 밀려나 있었습니다. 하지만 2012년, AlexNet이 이미지 분류 대회인 ILSVRC에서 압도적인 성능을 보여주면서 딥러닝과 CNN은 화려하게 부활했습니다.

R-CNN은 바로 이 강력한 CNN을 '이미지 분류'가 아닌 '객체 탐지'에 어떻게 효과적으로 적용할 수 있을지에 대한 고민에서 출발했습니다.

* **Deep Network를 사용하여 객체의 위치를 어떻게 특정할 것인가? (Localization)**
* **양이 적은 데이터만으로 어떻게 대용량 모델을 훈련시킬 것인가? (Training)**

---

## R-CNN은 어떻게 동작하는가?

R-CNN 시스템은 세 가지 주요 모듈로 구성된다.

![R-CNN 모델 구조](/assets/img/r_cnn_architecture.png)
> 이미지 출처: Rich feature hierarchies for accurate object detection and semantic segmentation (Girshick et al., 2014)

1.  **Region Proposal (영역 제안):** 먼저, 이미지에서 객체가 있을 만한 위치를 약 2000개 정도 찾아낸다. 이 논문에서는 **Selective Search** 알고리즘을 사용한다. 이 단계에서는 객체의 종류(class)는 고려하지 않고, 오직 "여기에 무언가 있을 것 같다"는 후보 영역만 빠르게 추출한다.

2.  **Feature Extraction (특징 추출):** 제안된 2000개의 영역들을 모두 동일한 크기(227x227 픽셀)로 변형(Warping)시킨 후, 미리 학습된 대규모 CNN 모델에 입력하여 각 영역으로부터 4096차원의 고정 길이 특징 벡터(feature vector)를 추출한다.

3.  **Classification (분류):** 마지막으로, 추출된 특징 벡터를 사용하여 각 영역이 어떤 객체 클래스에 속하는지, 또는 배경인지를 **선형 SVM(Support Vector Machines)** 분류기를 통해 판별한다.

### 훈련 과정의 핵심: Pre-training과 Fine-tuning

R-CNN의 높은 성능은 효과적인 훈련 전략 덕분이다.

-   **Supervised Pre-training:** 먼저, 대규모 이미지 분류 데이터셋인 `ILSVRC 2012`를 사용하여 CNN 모델을 충분히 학습시킨다. 이 과정을 통해 CNN은 이미지의 기본적인 특징(선, 질감, 색상 등)을 풍부하게 학습하게 된다.

-   **Domain-specific Fine-tuning:** 그 후, 우리가 실제로 사용하려는 `PASCAL VOC` 데이터셋에 맞게 CNN 모델을 미세 조정한다. 이 과정에서는 마지막 분류 레이어를 기존 1000개에서 `(객체 클래스 수 + 배경 1개)`로 교체하고, 더 낮은 학습률(learning rate)로 모델을 추가 학습시켜 특정 도메인에 대한 성능을 끌어올린다.

> **💡 IoU (Intersection-over-Union)란?**
> Fine-tuning과 SVM 훈련 시, 어떤 영역이 '정답'이고 어떤 영역이 '오답'인지 판단하는 기준이 필요하다. 이때 IoU가 사용된다. IoU는 **실제 정답 영역(Ground-truth box)과 모델이 제안한 영역(Proposed region)이 얼마나 겹치는지를 나타내는 지표**이다.
> `IoU = Overlapping Region / Combined Region`
> R-CNN에서는 이 IoU 값이 특정 임계값(예: 0.5) 이상이면 '정답'(Positive), 그보다 훨씬 낮으면 '오답'(Negative)으로 간주하여 모델을 학습시킨다.

### Bounding-box Regression

R-CNN은 Selective Search가 제안한 영역의 위치가 완벽하지 않을 수 있다는 점을 보완하고자, **Bounding-box regression**을 추가로 사용한다. 이는 CNN을 통해 얻은 특징을 바탕으로, 제안된 영역의 위치와 크기를 실제 객체에 더 가깝게 미세 조정하는 선형 회귀 모델이다. 이 과정을 통해 탐지 정확도를 3~4%p 추가로 향상시켰다.

---

## R-CNN의 의의와 영향

R-CNN은 다음과 같은 중요한 기여를 통해 Object Detection 분야에 큰 발전을 이루었습니다.

1.  **고전적인 컴퓨터 비전 기법(Region Proposals)과 딥러닝(CNN)의 성공적인 결합**을 보여주었다.
2.  대규모 데이터셋으로 **사전 학습(Pre-training) 후, 특정 도메인에 맞게 미세 조정(Fine-tuning)**하는 패러다임이 데이터가 부족한 환경에서도 매우 효과적임을 입증했다.

비록 처리 속도가 느리다는 단점이 있었지만, R-CNN이 제시한 접근법은 이후 등장하는 Fast R-CNN, Faster R-CNN 등 더 빠르고 정확한 모델들의 탄탄한 기반이 되었습니다.