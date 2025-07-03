---
title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
tags:
- computer vision
- deep learning
- CNN
- object detection
- R-CNN
- RPN
date: "2024-05-24"
thumbnail: "/assets/img/thumbnail/fasterrcnn.png"
bookmark: true
---

> 이 글은 [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) 논문을 참고하여 핵심 내용을 정리한 것입니니다.

# Faster R-CNN: 영역 제안(Region Proposal) 혁신신

SPPnet, Fast R-CNN과 같은 모델의 등장으로 객체 탐지(Object Detection) 네트워크의 속도는 비약적으로 빨라졌다. 하지만 이들의 혁신에도 불구하고, 전체 탐지 시스템은 여전히 실시간과는 거리가 있었다. 문제는 탐지 네트워크 외부에 존재했다. 객체가 있을 만한 위치를 추천해주는 **'영역 제안(Region Proposal)'** 단계가 심각한 병목 현상을 일으키고 있었기 때문이다.

Faster R-CNN은 바로 이 문제를 정면으로 돌파한 논문이다. 탐지 네트워크와 **완전히 통합된 Region Proposal Network(RPN)**를 제안함으로써, 영역 제안에 드는 비용을 거의 없애고 진정한 의미의 실시간 고성능 객체 탐지 시스템의 시대를 열었다.

---

## 왜 Faster R-CNN이 필요했을까?

Fast R-CNN까지의 탐지 모델들은 두 단계로 구성되었다.

1.  **영역 제안:** Selective Search와 같은 알고리즘을 사용해 객체가 있을 법한 후보 영역 수천 개를 이미지에서 추출한다.
2.  **객체 탐지:** 각 후보 영역에 대해 CNN 기반 탐지 네트워크를 실행하여 클래스를 분류하고 위치를 보정한다.

문제는 1단계였다. Selective Search와 같은 알고리즘은 CPU 기반으로 작동하여 이미지 한 장당 2초라는 느린 속도를 보였다. 반면, 2단계의 탐지 네트워크는 GPU를 활용하여 훨씬 빠른 속도로 작동했다. 아무리 탐지 네트워크를 최적화해도, 영역 제안 단계의 느린 속도 때문에 전체 시스템의 발목이 잡혀있던 것이다.

Faster R-CNN은 이 비효율적인 이중 구조를 타파하고자 했다. "영역 제안 역시 딥러닝 네트워크로 처리할 수 있지 않을까? 더 나아가, 탐지 네트워크와 연산을 공유할 수는 없을까?"라는 아이디어에서 출발하여, **RPN이라는 혁신적인 구조를 제안**했다.

---

## Faster R-CNN은 어떻게 동작하는가?

Faster R-CNN의 핵심은 **RPN(Region Proposal Network)**과 이것이 탐지 네트워크(Fast R-CNN)와 **합성곱 계층을 공유**하는 구조에 있다.

![Faster R-CNN Architecture Diagram](/assets/img/faster_rcnn_architecture.png)
> RPN은 백본 네트워크(예: VGG-16)의 특징 맵을 입력받아 후보 영역을 생성하고, 이 후보 영역과 특징 맵은 Fast R-CNN 탐지 헤드로 전달되어 최종 탐지를 수행한다. 모든 과정이 하나의 통합된 네트워크 안에서 이루어진다.

### 1. Region Proposal Network (RPN)
![Region Proposal Network (RPN)](/assets/img/rpn.png)
RPN은 그 자체가 하나의 작은 완전 합성곱 네트워크(FCN)이다. 그 역할은 백본 네트워크가 추출한 깊은 특징 맵(feature map)을 보고, 객체가 있을 만한 영역과 그 점수를 빠르게 제안하는 것이다.

* **슬라이딩 윈도우와 앵커 박스:** RPN은 특징 맵 위를 작은 윈도우로 슬라이딩하며 각 위치를 훑는다. 각 위치에서 바로 영역을 제안하는 것이 아니라, 미리 정의된 다양한 크기와 종횡비를 가진 **k개의 앵커 박스(Anchor Box)**를 기준으로 삼는다. 논문에서는 3가지 크기와 3가지 종횡비를 조합해 총 9개의 앵커를 사용했다.
* **두 개의 출력:** RPN은 각 앵커 박스에 대해 두 가지를 예측한다.
    1.  **객체 점수 (Objectness Score):** 이 앵커 안에 객체가 있는지 없는지에 대한 확률.
    2.  **상자 회귀 (Box Regression):** 앵커 박스의 위치를 실제 객체 위치에 더 가깝게 보정하기 위한 좌표 조정값.

이 구조의 가장 큰 장점은 **변환 불변성(Translation-Invariant)**이다. 이미지에서 객체가 이동해도 동일한 앵커와 함수를 통해 제안을 예측할 수 있어, 모델이 훨씬 효율적이고 강건해진다.

### 2. 훈련 방식: 4단계 교대 최적화

RPN과 Fast R-CNN 탐지 네트워크가 안정적으로 특징 공유 계층을 학습하도록 하기 위해, 논문은 독특한 **4단계 교대 최적화(Alternating Optimization)** 방식을 사용한다.

1.  **1단계 (RPN 훈련):** ImageNet으로 사전 학습된 모델을 가져와 RPN을 먼저 훈련시킨다.
2.  **2단계 (탐지기 훈련):** 1단계에서 생성된 영역 제안들을 이용해 별도의 Fast R-CNN 탐지 네트워크를 훈련한다.
3.  **3단계 (RPN 미세 조정):** 2단계에서 훈련된 탐지기의 가중치로 RPN을 다시 초기화한다. 이때 공유하는 합성곱 계층은 고정하고, RPN 고유의 계층만 미세 조정한다. 이제 두 네트워크는 계층을 공유하기 시작한다.
4.  **4단계 (탐지기 미세 조정):** 마지막으로, 공유 계층을 고정한 채 Fast R-CNN의 완전 연결 계층만 다시 미세 조정한다.

이 과정을 통해 두 네트워크는 매끄럽게 하나의 통합된 네트워크로 완성된다.

---

## Faster R-CNN의 의의와 영향

Faster R-CNN이 가져온 결과는 놀라웠다.

* **획기적인 속도:** RPN은 특징 공유 덕분에 영역 제안을 이미지당 **10ms**라는 거의 공짜에 가까운 비용으로 수행했다. 그 결과 VGG-16과 같은 무거운 모델을 사용했음에도 전체 시스템이 **초당 5프레임(5fps)**으로 작동하여 실시간 탐지의 가능성을 열었다.
* **높은 정확도:** 단순히 빠르기만 한 것이 아니었다. 학습을 통해 생성된 RPN의 제안은 기존 Selective Search보다 품질이 높아, PASCAL VOC 2007 데이터셋에서 **73.2%의 mAP**를 달성하며 당시 최고 수준의 정확도를 기록했다.

![Faster R-CNN Detection Example](/assets/img/thumbnail/fasterrcnn.png)
> 이미지 출처: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks 논문. 다양한 크기와 종횡비의 객체들을 효과적으로 탐지하는 모습.

Faster R-CNN은 영역 제안 단계를 딥러닝 파이프라인 안으로 완전히 끌어들여, **최초로 종단간(end-to-end) 학습이 가능한 통합된 실시간 객체 탐지 프레임워크**를 구축했다. 이 RPN의 개념은 이후 YOLO, SSD와 같은 1-stage detector에도 큰 영향을 미쳤으며, 현대 객체 탐지 기술의 표준을 세운 기념비적인 연구로 평가받는다.