# MASC : Multi-Attribute Semantic Consistency for Deepfake Image Detection

## 프로젝트 개요

이 프로젝트는 **단일 얼굴 이미지**에 대해,  
여러 고수준(face attribute, identity, vision-language 임베딩 등)의 정보를 활용하여  
**semantic self-consistency**를 측정하고 이를 바탕으로 deepfake 여부를 판별하는 방법을 구현한다.

---

## 방법 요약

1. **전처리**
   - 얼굴 검출 및 정렬 수행
   - 전체 얼굴 + 상단(눈/이마), 중앙(코/볼), 하단(입/턱) 등의 부분 영역을 crop

2. **다중 고수준 모델 적용 (모두 frozen)**
   - Face attribute classifier: 나이, 성별, 표정 등 예측
   - Face recognition 모델(예: ArcFace): identity embedding 추출
   - Vision-Language 모델(예: CLIP 이미지 인코더): global / local 임베딩 추출  

3. **Semantic Consistency / Inconsistency Feature 구성**
   - 전체 얼굴 vs 부분 영역 간 **속성 분포 차이** (나이/성별/표정 KL divergence 등)
   - 전체 얼굴 vs 부분 영역 간 **identity embedding self-consistency** (cosine similarity)
   - 전체 얼굴 vs 부분 영역 간 **CLIP 임베딩 self-consistency** (cosine similarity)

   위 값들을 모두 모아 하나의 **semantic consistency feature vector**를 구성한다.  
   각 feature는 정규화/스케일링 후 사용한다.

4. **최종 분류기 학습**
   - 입력: semantic consistency feature vector
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

---

## Getting Started
1. git clone
```
git clone https://github.com/nullpo0/MASC.git
cd MASC
```
2. 가상환경 구축
- Anaconda 환경 기준
```
conda create -n MASC python=3.11
conda activate MASC
pip install py-feat
pip install facenet-pytorch
pip uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/openai/CLIP.git
```
3. preprocessing
```
python preprocessing_pipeline.py --data_root <dataset_path>

# if device == cpu:
    python preprocessing_pipeline.py --data_root <dataset_path> --batch-size 4 --device cpu 
```
4. training
```
python train.py
```
5. predict
```
python predict.py
```
---
## pretrained model
- facenet, clip, pyfeat은 preprocessing_pipeline.py 실행 시 자동으로 다운로드(꽤 오래 걸림)

- fairface
https://drive.google.com/file/d/11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB/view?usp=sharing

---
## 주의사항
- 원활하게 작동하려면 CUDA GPU에서 하는것을 추천함. cpu로도 되긴 하지만 오래걸림.