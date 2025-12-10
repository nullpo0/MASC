# MASC : Multi-Attribute Semantic Consistency for Deepfake Image Detection

## 프로젝트 개요

이 프로젝트는 **단일 얼굴 이미지**에 대해,  
여러 고수준(face attribute, identity, vision-language 임베딩 등)의 정보를 활용하여  
**semantic self-consistency**를 측정하고 이를 바탕으로 deepfake 여부를 판별하는 방법을 구현한다.

기존 딥페이크 탐지 연구들은 주로
- **저수준 아티팩트 / 주파수 특성**에 의존하거나 
- 하나의 모델(CNN, ViT, CLIP 등)을 end-to-end로 학습해 real/fake를 분류하는 방식이 많다.   

또한 CLIP과 같은 Vision-Language Model을 deepfake detection에 활용하는 연구나   
이미지 내 패치 간 **feature self-consistency**를 활용하는 연구는 존재하지만,   
여러 개의 사전학습 모델에서 나온 **고수준 속성·의미들의 일관성/불일치 자체를 명시적으로 feature로 사용하는 시도는 거의 없다.  

본 프로젝트는 이러한 간극에 주목하여, **여러 독립적인 pre-trained 모델의 “발언”이 서로 얼마나 모순되는지**를 수치화한 후  
이를 최종 분류기의 입력으로 사용하는, 비교적 새로운 형태의 high-level deepfake detector를 제안한다.

---

## 방법 요약

1. **전처리**
   - 얼굴 검출 및 정렬 수행
   - 전체 얼굴 + 상단(눈/이마), 중앙(코/볼), 하단(입/턱) 등의 부분 영역을 crop

2. **다중 고수준 모델 적용 (모두 frozen)**
   - Face attribute classifier: 나이, 성별, 표정 등 예측
   - Face recognition 모델(예: ArcFace): identity embedding 추출
   - Vision-Language 모델(예: CLIP 이미지 인코더): global / local 임베딩 추출 및 텍스트 프롬프트(“real photo of a person”, “AI-generated face”)와의 유사도 계산   

3. **Semantic Consistency / Inconsistency Feature 구성**
   - 전체 얼굴 vs 부분 영역 간 **속성 분포 차이** (나이/성별/표정 KL divergence 등)
   - 패치 간 **CLIP 임베딩 self-consistency** (1 − cosine similarity)
   - “real” / “fake” 관련 텍스트 프롬프트와의 CLIP 유사도 차이
   - identity embedding이 암시하는 인구통계(예: age/gender cluster)와 attribute classifier 출력 간의 불일치 정도

   위 값들을 모두 모아 하나의 **semantic (in)consistency feature vector**를 구성한다.  
   각 feature는 정규화/스케일링 후 사용한다.

4. **최종 분류기 학습**
   - 입력: semantic (in)consistency feature vector
   - 모델: 작은 MLP 또는 Logistic Regression / XGBoost
   - 출력: real vs fake 확률
   - 학습/평가는 FaceForensics++ 등 공개 deepfake 이미지 데이터셋을 활용하고,  
     cross-dataset 실험을 통해 **일반화 성능과 robustness**를 검증한다.   

---

## 핵심 아이디어 & 기대 효과

- **High-level cue 활용**  
  얼굴의 나이, 성별, 표정, identity, 텍스트 의미 등 고수준 정보를 활용하므로,  
  단순 픽셀 노이즈·압축·해상도 변화에 덜 민감한 robust한 특성을 기대할 수 있다.

- **Semantic self-consistency 기반 탐지**  
  기존 self-consistency 기반 방법이 주로 CNN feature level의 일관성에 초점을 둔 반면,   
  본 프로젝트는 사람이 이해할 수 있는 **속성/의미 수준**에서의 일관성/불일치를 명시적으로 수치화한다.

- **Multi-model, Multi-patch 관점**  
  하나의 네트워크를 end-to-end로 학습하는 대신,  
  여러 사전학습 모델과 여러 얼굴 영역의 출력을 조합해 “서로 말이 안 맞는 정도”를 학습하므로,  
  새로운 생성 모델/조작 방식에 대해서도 **일반화 가능성**이 높을 것으로 기대된다.

---

## 목표

- deepfake 이미지에 대해 **semantic consistency feature 기반** binary classifier 구현
- 여러 baseline(CNN/ViT 기반, CLIP 단일 임베딩 기반 등)과 비교하여
  - in-dataset 성능
  - cross-dataset generalization
  - 이미지 열화(JPEG 압축, blur 등)에 대한 robustness
  를 정량적으로 평가
- 최종적으로, **“고수준 의미 일관성”이 딥페이크 탐지에 유효한 단서가 될 수 있다**는 것을 실험적으로 보이는 것을 목표로 한다.
