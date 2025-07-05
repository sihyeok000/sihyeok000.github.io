---
title: "OverFeat: Integrated Recognition, Localization and Detection
using Convolutional Networks"
tags:
- computer vision
- deep learning
- CNN
- object detection
- OverFeat
date: "2024-05-17"
thumbnail: "/assets/img/thumbnail/overfeat.png"
bookmark: true
---

> 이 글은 [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/abs/1312.6229) 논문을 참고하여 핵심 내용을 정리한 것입니다.

# OverFeat: 하나의 CNN으로 Classification, Localization 및 Detection

AlexNet이 이미지 분류의 새 시대를 연 직후, 딥러닝의 다음 과제는 '이미지 안에 무엇이 있는가'를 넘어 '그것이 어디에 있는가'를 알아내는 객체 탐지(Object Detection)로 확장되었다. 비슷한 시기 R-CNN은 Region proposal과 classification 이 나누어진 2-stage detector로 개발되었으나  OverFeat는 **하나의 통합된 CNN 프레임워크**를 통해 이미지 분류(Classification), 객체 위치 특정(Localization), 그리고 객체 탐지(Detection)를 동시에 수행하는 방식을 제시하며 이 분야의 발전에 중요한 기틀을 마련한 논문이다.

---
# OverFeat의 등장 배경

CNN이 이미지 분류에서 압도적인 성능을 보이자, 연구자들은 자연스럽게 이 강력한 도구를 객체 탐지에 적용하고자 했다.  가장 직관적인 방법은 이미지의 모든 가능한 위치와 크기에 대해 잘라낸 영역(Window)마다 분류용 CNN을 실행하는 **슬라이딩 윈도우(Sliding Window)** 방식이었다. 

하지만 이 방식에는 두 가지 명백한 한계가 있었다.

* **부정확한 위치:** 슬라이딩 윈도우가 실제 객체의 경계와 정확히 일치할 확률은 매우 낮다. 창문 안에 객체의 일부(예: 개의 머리)만 포함되더라도 분류는 성공할 수 있지만, 객체의 정확한 위치를 찾는 데는 실패하게 된다. 
* **많은 계산량:** 이미지 전체를 조밀하게 훑으려면 수많은 윈도우에 대해 반복적으로 CNN을 실행해야 하므로 비효율적이다.


OverFeat는 바로 이 문제들을 해결하기 위해, 단순히 분류기를 반복 실행하는 것을 넘어 **하나의 네트워크가 위치 정보까지 함께 학습하고 추론**하는 통합적인 접근법을 제안했다.  특히, 배경(Background) 샘플에 대한 별도의 학습 없이 오직 객체(Positive Classes)에만 집중함으로써 네트워크가 더 높은 정확도를 달성할 수 있도록 설계했다. 

---

## OverFeat, 통합된 비전 시스템

OverFeat는 [2013년 ILSVRC 대회](https://www.image-net.org/challenges/LSVRC/2013/results.php#loc)에서 Localization 부문 우승을 차지하며 그 성능을 입증했다. 이 논문의 요점은 이미지에서 object를 동시에 classify, localize 및 detect하도록 단일 CNN을 훈련하면 모든 task에서의 정확도를 높일 수 있음을 보여주는 것이다. 또한 예측된 bounding boxes를 누적하는 방식의 localization 및 detection을 위한 새로운 방식을 소개한다. 이 방식을 통해 background samples에 대한 학습 없이 positive classes에만 집중할 수 있도록 한다.

이 모델의 핵심 아이디어는 다음과 같이 요약할 수 있다.

1.  **통합 프레임워크(Integrated Framework):** 분류, 위치 특정, 탐지라는 각기 다른 작업을 별개의 모델이 아닌, **특징을 공유하는 단일 CNN**으로 학습하고 해결한다. 
2.  **효율적인 다중 스케일 슬라이딩 윈도우(Multi-Scale Sliding Window):** 다양한 크기의 이미지에 대해 Sliding window 방식을 효율적으로 구현하였다. 이는 여러 크기의 이미지를 각각 네트워크에 input하는 대신 더 큰 이미지에 convolution을 한 번만 적용하여 중복된 계산을 피하는 것이다.
3.  **Bounidng box regression:** Object의 경계를 예측하도록 학습하는 deep learning apporach를 도입하였다. classifier 뿐만 아니라 object의 정확한 위치와 크기를 예측하는 regression 모델을 별도로 훈련시킨다.
4.  **예측 누적 및 병합(Prediction Accumulation & Merging):** 탐지된 bounding boxes를 suppress하는 방식 대힌 **accumulate, merge**하여 예측의 신뢰도를 높이는 방식을 제안한다. 이 방식을 통해 background samples에 대한 별도의 훈련 없이 positive classes에만 집중할 수 있게 된다.


## OverFeat의 작동 방식

OverFeat 시스템은 분류, 위치 특정, 탐지 작업을 단계적이면서도 유기적으로 해결한다. 모든 작업은 기본적으로 동일한 CNN 구조에서 학습된 특징을 공유한다.


### 1. 다중 스케일 분류 (Multi-Scale Classification)

OverFeat의 첫 번째 혁신은 **고정된 크기의 입력 이미지를 사용하지 않는다는 점**이다.

* **FCN(Fully Convolutional) 방식의 추론:** 학습 시에는 fully connected layer를 사용하지만, 추론(test) 시에는 이 layer들을 1x1 컨볼루션 레이어처럼 취급한다.  이 덕분에 네트워크는 다양한 크기의 이미지를 입력받아, 입력 크기에 비례하는 공간적 출력(Spatial Output Map)을 생성할 수 있다. 
* **효율성:** 여러 스케일의 이미지를 각각 CNN에 넣는 대신, 이미지를 한 번만 통과시켜 얻은 특징 맵(feature map) 위에서 슬라이딩 윈도우를 적용하는 것과 같은 효과를 낸다.  이는 계산적으로 매우 효율적이다.

결과적으로, 각기 다른 스케일로 이미지를 입력하여 얻은 분류 결과를 종합하여 최종 클래스를 결정함으로써 분류 정확도를 높인다. 

### 2. 회귀를 통한 위치 예측 (Localization via Regression)

분류만으로는 객체의 정확한 위치를 알 수 없다. OverFeat는 이 문제를 해결하기 위해 **바운딩 박스 회귀 모델**을 도입했다.

* **Regressor 학습:** 분류기를 학습시킨 후, 네트워크의 5번째 컨볼루션 레이어 위에 새로운 회귀 헤드(regressor head)를 추가하여 별도로 학습시킨다.
* **좌표 예측:** 이 회귀 모델의 역할은 특정 위치의 특징 맵을 보고, 그곳에 객체가 있다면 실제 경계 상자의 **정확한 좌표($x_{min}, y_{min}, x_{max}, y_{max}$)를 예측**하는 것이다. 즉, 슬라이딩 윈도우의 부정확함을 보정해주는 역할을 한다.
* **Regressor 학습:** 분류기를 학습시킨 후, 네트워크의 5번째 컨볼루션 레이어 위에 새로운 회귀 헤드(regressor head)를 추가하여 별도로 학습시킨다. 
* **좌표 예측:** 이 회귀 모델의 역할은 특정 위치의 특징 맵을 보고, 그곳에 객체가 있다면 실제 경계 상자의 **정확한 좌표($x_{min}, y_{min}, x_{max}, y_{max}$)를 예측**하는 것이다.  즉, 슬라이딩 윈도우의 부정확함을 보정해주는 역할을 한다.

### 3. 증거 누적으로 객체 탐지 (Detection by Accumulating Evidence)

마지막 탐지 단계에서는 위 두 과정에서 얻은 수많은 예측들을 종합하여 최종 결과를 도출한다.

- **예측 생성:** 네트워크는 여러 스케일에서 수천 개의 잠재적 바운딩 박스와 각 박스에 대한 클래스 신뢰도 점수를 생성한다. 
2.  **예측 병합 (Merge Predictions):** 여기서 OverFeat의 독특한 접근법이 드러난다. 가장 높은 점수의 예측 하나만 남기는 기존 방식과 달리, OverFeat는 **서로 가깝게 예측된 동일 클래스의 바운딩 박스들을 합친다**.
3.  **신뢰도 향상:** 여러 예측 박스가 합쳐지면서 흩어져 있던 약한 증거들이 모여 하나의 강력하고 신뢰도 높은 탐지 결과를 만들어낸다. 

![OverFeat 결과 예시](/assets/img/OverFeat_example.png)
> 이미지 출처: Integrated Recognition, Localization and Detection using Convolutional Networks
> **위(Localization):** 5개의 가장 유력한 예측을 통해 늑대의 위치를 정확히 찾아낸다.
> **아래(Detection):** 사람, 램프, 마이크 등 작고 여러 개인 객체들도 성공적으로 탐지한다. 이는 탐지 데이터셋의 난이도가 더 높다는 것을 보여준다. 

---

## OverFeat의 의의와 영향

OverFeat는 R-CNN과 거의 동시에 등장하여 객체 탐지 분야에 큰 영향을 미쳤다. R-CNN이 'Region Proposal + CNN'이라는 2단계 접근법의 가능성을 열었다면, OverFeat는 다음과 같은 중요한 기여를 했다.

1.  **통합 프레임워크의 제시:** 단일 CNN을 공유하여 분류, 위치 특정, 탐지를 모두 수행하는 아이디어는 이후 YOLO, SSD와 같은 **1-stage detector**의 개념적 토대가 되었다. 
2.  **슬라이딩 윈도우의 재해석:** 컨볼루션 특징 맵 위에서 슬라이딩 윈도우를 효율적으로 수행하는 방식은 이후 많은 모델에서 채택되었다. 
3.  **바운딩 박스 회귀의 표준화:** CNN 특징을 이용해 경계 상자의 위치를 직접 회귀하는 아이디어는 R-CNN을 포함한 거의 모든 현대 객체 탐지 모델의 핵심 구성 요소로 자리 잡았다. 

비록 속도와 정확도 면에서 후속 모델들에게 자리를 내주었지만, OverFeat가 제시한 통합적이고 효율적인 접근 방식은 딥러닝 기반 객체 탐지 기술이 나아갈 방향을 제시한 선구적인 연구로 평가받는다.