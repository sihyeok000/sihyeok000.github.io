---
title: "R-CNN : Rich feature hierarchies for accurate object detection and semantic segmentation"
tags:
    - Computer vision
    - Deep learning
    - CNN
    - R-CNN
    - Object deection
date: "2024-05-12"
thumbnail: "/assets/img/thumbnail/rcnn.jpg"
bookmark: true
---

> 이 글은 [R-CNN : Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524) 논문을 참고하여 핵심 내용을 정리한 것입니다.

---

## R-CNN : Object Detection의 게임 체인저
R-CNN은 기존 Object detection 방식의 패러다임을 전환시킨 2-stage object detection 모델이다. 당시 object detection의 벤치마크 PASCAL VOC dataset에서 다른 모델들이 성능 향상의 정체기를 겪고 있었고, 주된 방식은 여러 low-level image features에 high-level context를 결합하는 복잡한 앙상블 시스템이었다.

R-CNN은 두 가지 핵심 아이디어를 결합하여 VOC2012에서 이전 최고기록보다 mAP(mean Average Precision)를 30% 이상 향상시킨 53.3%의 mAP를 달성했다.

1.  **Regions with CNNs:** 객체의 위치를 찾고 분할하기 위해, 먼저 이미지에서 객체가 있을 법한 영역(Region Proposal)을 찾고, 이 영역들에 CNN(Convolutional Neural Network)을 적용한다.
2.  **Pre-training & Fine-tuning:** 라벨링된 데이터가 부족한 문제를 해결하기 위해, 대규모 데이터셋(ILSVRC)으로 미리 학습(Supervised Pre-training)시킨 후, 특정 도메인의 데이터셋(PASCAL)에 맞게 미세 조정(Fine-tuning)하여 성능을 극대화한다.

이를 Region Proposals + CNN => R-CNN 이라고 이름붙였다.

---

## R-CNN 등장 배경
이전의 Visual Recognition 기술은 주로 전통적인 컴퓨터 비전 방법론인 `SIFT`, `HOG`와 같은 고정된 특징 추출 방식에 의존했다. 이는 이미지의 밝기, 형태 등 단순한 정보를 기반으로 했기에 복잡한 실제 환경의 객체를 인식하는 데 한계가 있다.

> **💡 SIFT(Scale-Invariant Feature Transform)**
> 이미지에서 scale, rotation 등의 변화에 invariant(robust)한 특징점을 추출하는 컴퓨터 비전 알고리즘. DoG(Difference of Gaussian)을 계산하여 극값을 찾은 뒤(주로 edge, corner) 특징점 주변의 패턴을 모아 벡터로 description하는 방식으로 특징을 추출한다.
> ![SIFT](/assets/img/sift.png)
이미지 출처 : Boosting of factorial correspondence analysis for image retrieval

  
> **💡 HOG(Histogram of Oriented Gradients)**
> 이미지를 작은 구역들로 나누어 경계선의 방향 분포를 히스토그램으로 만들어 이미지 형태를 표현하는 방법. 흑백으로 변환 후 픽셀마다 x, y 방향으로의 밝기차이를 계산하고 이미지를 일정 크기의 셀들로 나누어 그래디언트에 대한 히스토그램을 만든다.

실제로 영장류의 시각을 통한 recognition이 몇 단계 downstream으로 일어난다는 것은 이미지에서 더 좋은 특징을 잡기 위해 몇 단계를 나눌 수 있다는 것을 암시하고, 이는 생물학적 영감을 받아 개발된 Neocognitron 모델과도 연관이 있다. 이 영향을 받은 CNN(Convolution Neural Network)의 경우 1990년대에 많이 사용되었으나 SVM의 등장으로 사용이 줄어들었다. 그러나 2012년 CNN을 사용한 `AlexNet`이 2012년 ImageNet 대회(ILSVRC)에서 압도적으로 1위를 하면서 컴퓨터 비전 영역에서 딥러닝, 특히 CNN의 가능성을 증명하였다.

R-CNN은 바로 이 강력한 CNN을 '이미지 분류'가 아닌 '객체 탐지'에 어떻게 효과적으로 적용할 수 있을지에 대한 고민에서 출발했다.

---

## 두 가지 문제
Object Detection에서 CNN을 적용하고, 기존 PASCAL VOC에서의 획기적인 성능 향상을 위해 저자들은 두 가지 문제에 집중하였다.
* **1. Deep Network를 사용하여 객체의 위치를 어떻게 특정할 것인가? (Localization)**
* **2. 양이 적은 데이터만으로 어떻게 대용량 모델을 훈련시킬 것인가? (Training)**

##### **1. Localization**

AlexNet과 같은 Image classification task와는 달리 Object detection의 경우 이미지 내에서 많은 Object를 Localizing해야 한다. 이를 회귀 문제로 치환하거나, Sliding-window approach를 채택할 수 있다(비슷한 시기 `OverFeat`의 경우 이를 채택). 회귀 문제로 치환하는 경우에는 성능이 낮다는 연구결과가 있었다. 그리고 Sliding-window 방식은 convolutional layers를 깊게 쌓을 시 큰 receptive field와 stride가 발생하는 문제가 있다.

대신 이러한 문제를 **"Recognition using regions"** 패러다임을 통해 해결한다. 테스트 시 input image에 대해서 약 2,000개의 category-independent region proposals를 생성하고 CNN을 사용하여 각 proposals에 대한 fixed-length feature vector를 추출한다. 이를 category-specific한 선형 SVM 분류기를를 이용해서 class를 예측하는 방식을 사용한다.

##### **2. Training**

두 번째 문제는 라벨링된 training data가 부족하여 대규모 CNN을 훈련하기에 충분치 않다는 것이다. 이 문제의 기존 해결책은 **unsupervised** pre-training을 한 후 supervised fine-tuning을 적용하는 것이다. 이 논문에서는 대규모 dataset ILSVRC에 대한 **supervised** pre-training + 소규모 dataset PASCAL에 대해 **fine-tuning** 하는 것이 CNN을 학습시키는 데 있어 효과적인 패러다임이라는 것을 보여준다.


---


## R-CNN은 어떻게 동작하는가?

R-CNN 시스템은 세 가지 주요 모듈로 구성된다.

![R-CNN 모델 구조](/assets/img/r_cnn_architecture.png)
> 이미지 출처: Rich feature hierarchies for accurate object detection and semantic segmentation (Girshick et al., 2014)

1.  **Region Proposal (영역 제안):** 먼저, 이미지에서 객체가 있을 만한 위치를 약 2000개 정도 찾아낸다. 이 논문에서는 **Selective Search** 알고리즘을 사용한다. 이 단계에서는 객체의 종류(class)는 고려하지 않고, 오직 "여기에 무언가 있을 것 같다"는 후보 영역만 빠르게 추출한다.

2.  **Feature Extraction (특징 추출):** 제안된 2000개의 영역들을 모두 동일한 크기(227x227 픽셀)로 변형(Warping)시킨 후, 미리 학습된 대규모 CNN 모델에 입력하여 각 영역으로부터 4096차원의 고정 길이 특징 벡터(feature vector)를 추출한다.

3.  **Classification (분류):** 마지막으로, 추출된 특징 벡터를 사용하여 각 영역이 어떤 객체 클래스에 속하는지, 또는 배경인지를 **선형 SVM(Support Vector Machines)** 분류기를 통해 판별한다.

### 훈련 전략

R-CNN의 훈련 전략은 다음과 같다. 특히 Supervised Pre-training과 Domain-specific fine-tuning이 이 논문에서 주요하게 강조되는 부분이다.

-   **Supervised Pre-training:** 먼저, 대규모 이미지 분류 데이터셋인 `ILSVRC 2012`를 사용하여 CNN 모델을 충분히 학습시킨다. 이는 Bounding box없이 Image - Object class의 쌍으로 이루어진 데이터셋으로, 이 과정을 통해 CNN은 이미지의 기본적인 특징(선, 질감, 색상 등)을 풍부하게 학습하게 된다.

-   **Domain-specific Fine-tuning:** 그 후, 우리가 실제로 사용하려는 `PASCAL VOC` 데이터셋에 맞게 CNN 모델을 fine-tuning한다. 이 과정에서는 마지막 분류 레이어를 기존 1000개에서 (객체 클래스 수 + 배경 1개)로 교체하고, 더 낮은 학습률(learning rate)로 모델을 추가 학습시켜 특정 도메인에 대한 성능을 끌어올린다.

- **Object category classifiers** 자동차를 검출하기 위한 binary classifier를 예시로 생각해보자. 이를 tight하게 감싸고 있는 region의 경우 positive example에 해당할거고, 자동차와 관련이 없는 background에 해당하는 region은 negative example이 되어야 한다. 그런데 자동차와 부분적으로 겹치고있는 것은 어떻게 설정해야 하는가? 이 논문에서는 **IoU**의 임계값을 설정하여 그 임계값 이상일 시 positive example로 보았다. 임계값은 여러 값들에 대해 Grid search를 진행하여 0.3으로 설정하였다.

-  **Hard negative mining**
앞서 이전 단계에서 추출된 features를 class로 분류하기 위해 선형 SVM을 이용한다고 했다. 이를 학습시키는 전략으로 Hard negative mining을 채택했다. hard negative는 실제로는 negative example인데 positive라고 잘못 예측하기 쉬운 데이터를 의미한다. 이를 학습데이터로 사용하기 위해 모으는 것이 Hard negative mining이다. Object detection에서는 특히 훈련 데이터에 Negative patch가 많은 불균형이 존재하므로 이러한 작업이 성능 향상에 미치는 영향이 크다.

> **💡 IoU (Intersection-over-Union)란?**
> Fine-tuning과 SVM 훈련 시, 어떤 영역이 '정답'이고 어떤 영역이 '오답'인지 판단하는 기준이 필요하다. 이때 IoU가 사용된다. IoU는 **실제 정답 영역(Ground-truth box)과 모델이 제안한 영역(Proposed region)이 얼마나 겹치는지를 나타내는 지표**이다.
> `IoU = Overlapping Region / Combined Region`

**Bounding-box Regression**
모델 학습 이후 error analysis를 기반으로 localization error를 줄이기 위한 방법으로  **Bounding-box regression**을 추가로 사용한다. 이는 R-CNN이전에 존재한 object detection model인 DPM(Deformable Part Model)에서 영감을 받았다. CNN을 통과해 얻은 features를 바탕으로, 제안된 영역의 위치와 크기를 실제 객체에 더 가깝게 미세 조정하는 선형 회귀 모델이다. 이 과정을 통해 mAP를 3~4%p 추가로 향상시켰다.

---

## R-CNN의 의의와 영향

R-CNN은 당시 정체되었던 object detection model의 성능을 획기적으로 끌어올리는 간단하면서도 확장가능한 알고리즘이다. 이러한 성능을 달성하기 위해서 Bottom-up region proposals에 high-capacity CNN을 적용하는것과 부족한 훈련데이터 존재시 풍부한 보조 데이터를 통해 네트워크를 pre-training한 이후 domain-specific한 해당 소규모 데이터로 fine-tuning하는 방법론이 적용되었다. 즉 R-CNN의 의의와 영향은 다음과 같다.

1.  **고전적인 컴퓨터 비전 기법(Region Proposals)과 딥러닝(CNN)의 성공적인 결합**을 보여주었다.
2.  대규모 데이터셋으로 **사전 학습(Pre-training) 후, 특정 도메인에 맞게 미세 조정(Fine-tuning)**하는 패러다임이 데이터가 부족한 환경에서도 매우 효과적임을 입증했다.

비록 처리 속도가 느리다는 단점이 있었지만(여러 단계를 통과해야 하는 모델 구조), 이후 등장하는 `SPP-Net`, `YOLO`, `Fast R-CNN` 그리고 `Faster R-CNN` 등 더 빠르고 정확한 모델들의 기반이 되었다. CNN feature의 강력함을 AlexNet 이후 한 번 더 증명하여 Object detection 분야에서 feature engineering이 거의 사라지고, deep feature가 표준이 되었다.