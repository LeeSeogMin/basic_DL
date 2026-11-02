# 9주차 계획서 — 이미지 인식과 CNN: 컴퓨터가 이미지를 보는 방식

## 1. 장의 역할

9주차는 학생이 처음으로 이미지를 데이터로 다루는 장이다. "이미지는 숫자다"라는 말을 문장으로 듣는 데서 끝내지 않고, 8x8 손글씨 한 장의 픽셀값 64개를 표로 찍어 눈으로 확인한다.

그다음 합성곱을 numpy로 직접 계산한다. 3x3 창 하나에서 곱하고 더하는 과정을 로그에 그대로 찍어, 학생이 손으로 따라갈 수 있게 한다. 필터를 바꾸면 도드라지는 무늬가 달라지는 것까지 확인한다.

CNN 전체 구조는 개념 수준으로만 다룬다. 이 수업에서 CNN을 직접 학습시키지 않는다. 합성곱이 무엇을 하는지 손으로 확인하는 것이 이 장의 목표다. 분류기는 scikit-learn의 MLPClassifier로 만들고, 8x8을 한 줄로 편 64개 숫자를 그대로 넣는다. 이 방식이 어디까지 되고 어디서 틀리는지를 실패 갤러리로 본다.

`torch`, `tensorflow`를 쓰지 않는다. 합성곱은 `convolve2d()`에 직접 구현한다.

대상은 학부 2학년이다. 1차시 이론 강의, 2차시 실습으로 나눠 진행한다.

## 2. 학습 목표

1. 이미지 한 장이 컴퓨터에게 어떤 숫자 덩어리인지 설명한다.
2. 픽셀과 채널을 구분하고, 흑백과 컬러가 숫자 판 몇 장인지 말한다.
3. 이미지를 한 줄로 펴서 넣을 때 무엇이 사라지는지 실험 결과로 설명한다.
4. 3x3 필터로 합성곱을 손으로 한 칸 계산하고, 필터가 어떤 창에 큰 값을 내는지 말한다.
5. pooling이 크기를 줄이면서 무엇을 남기고 무엇을 남기지 못하는지 설명한다.
6. CNN의 층 순서를 개념 수준에서 그린다.

## 3. 절과 실습의 배치 (2026-11-02 개정)

7장과 같이 이론 절 바로 뒤에 그 개념을 확인하는 실습을 둔다. 문제를 먼저 측정하고 해법을 뒤에 둔다.

| 순서 | 제목 | 핵심 내용 | 그림 |
|---|---|---|---|
| 1 | 이미지는 숫자다 | 8x8 손글씨 한 장 = 숫자 64개, 0~16 | `digit_pixels.png` |
| 2 | 픽셀과 채널 | 흑백 판 한 장, 컬러 세 장, 숫자 개수 표 | `fig9-1-pixel-and-channel.png` |
| 3 | 한 줄로 펴서 넣으면 무엇이 사라지는가 | 영상 링크(한빛미디어 [딥러닝 프로젝트] 1강, hSUOpeUKqrM), 위아래 이웃이 8칸 떨어진다, 예측 1(순서를 섞어도 같다)·예측 2(밀면 무너진다)·가중치 폭증 | `flatten_neighbors.png` |
| 실습 9.1 | 64개 숫자로 분류하고 한계를 측정한다 | 분류·실패 갤러리, 순서 섞기 실험, 밀기 실험 | `shift_test.png`, `failure_gallery.png` |
| 4 | 합성곱과 필터 | 창 하나 계산, 닮은 창에 큰 값(세로선 필터 = 오른쪽 열 합 - 왼쪽 열 합), 창 옮기기와 6x6, 같은 필터 공유 | `fig9-2-convolution-step.png`, `filter_similarity.png`, `sliding_window.png` |
| 5 | pooling | 2x2 최대 pooling, 덩어리 안 이동은 지우고 경계를 넘는 이동은 못 지운다 | `fig9-3-pooling.png` |
| 실습 9.2 | 필터와 pooling을 직접 계산한다 | 필터 네 개, 1칸 민 이미지의 합성곱·pooling 비교 | `filter_gallery.png`, `pooling_shift.png` |
| 6 | CNN의 전체 모양 | 층 순서, 판 크기와 장수, 필터를 학습으로 찾는다, 3절 문제와 대응 표 | `fig9-4-cnn-structure.png` |
| 7 | 직접 움직여 보는 웹 자료 | 외부 시각 자료 4건 | 링크 표 |

6절에는 "이 수업에서 CNN을 직접 학습시키지 않는다"를 명시한다. 학습한 CNN의 이동 내성은 측정하지 않았다고 적는다.

7장 개정과 같이 「책임 있는 AI 체크」 절을 본문에서 뺐다.

## 4. 실습 설계

실습은 두 개다. 각각 노트북에서 몇 초 안에 끝난다. GPU가 필요 없다.

데이터는 두 실습 모두 `sklearn.datasets.load_digits()`다. 합성 데이터가 아니라 실제 손글씨를 스캔해 만든 공개 데이터이고, 내려받지 않고 scikit-learn 안에서 바로 쓴다.

### 실습 9.1 — 64개 숫자로 분류하고 한계를 측정한다

- 파일: `practice/chapter9/code/9-1-flatten-classifier.py` (2026-11-02 이전 이름 `9-2-digit-classifier.py`)
- 데이터: `load_digits()` 1797장, 8x8을 한 줄로 편 64개 숫자
- 모델: `MLPClassifier` 은닉층 하나. CNN이 아니다
- 학생 조작 지점: `SAMPLE_INDEX`, `HIDDEN_SIZE`, `MAX_ITER`, `TEST_RATIO`, `GALLERY_SIZE`
- 하는 일: 픽셀 표, 평탄화 그림, 분류(기준선 비교·`hidden_layer_sizes` 설명과 학습한 가중치 모양 출력·혼동행렬·실패 갤러리), 실험 1(64칸 순서를 섞어 새로 학습), 실험 2(학습한 모델에 오른쪽·왼쪽·아래 1칸, 오른쪽 2칸 민 테스트 이미지를 넣음)
- `max_iter`를 작게 잡아 나오는 `ConvergenceWarning`을 숨기지 않고 로그에 남긴다
- 산출물: `results/digit_pixels.png`, `results/flatten_neighbors.png`, `results/confusion_matrix.png`, `results/failure_gallery.png`, `results/shift_test.png`, `results/misclassified_digits.csv`

### 실습 9.2 — 필터와 pooling을 직접 계산한다

- 파일: `practice/chapter9/code/9-2-filter-and-pooling.py` (2026-11-02 이전 이름 `9-1-image-is-numbers.py`)
- 데이터: `load_digits()` 중 한 장(기본 17번)
- 학생 조작 지점: `SAMPLE_INDEX`, `MY_FILTER`, `SHIFT_PIXELS`
- 하는 일: 창 하나 손계산, 세로선 필터에 가장 크게·작게·0 근처로 반응한 창, 창 옮기기 그림, 필터 네 개 비교, 오른쪽으로 민 이미지의 합성곱·2x2 최대 pooling 비교
- 합성곱은 `convolve2d()`, pooling은 `max_pool2d()`에 직접 구현한다
- 산출물: `results/filter_similarity.png`, `results/sliding_window.png`, `results/filter_gallery.png`, `results/pooling_shift.png`

### 개념 그림 생성

- 파일: `practice/chapter9/code/9-0-concept-figures.py`
- 산출물: `results/fig9-1-pixel-and-channel.png`, `results/fig9-2-convolution-step.png`, `results/fig9-3-pooling.png`, `results/fig9-4-cnn-structure.png`
- 한글 폰트 도우미: `practice/chapter9/code/_krfont.py` (폰트가 없으면 영어 라벨로 자동 전환)
- 이 네 장의 숫자는 설명을 위해 손으로 고른 값이다. 실험 결과가 아니라는 것을 코드와 로그에 적는다

## 5. 실행 증거

```bash
python scripts/run_and_capture.py 9
python scripts/run_and_capture.py 9 --verify
```

본문의 정확도, 표본 수, 픽셀값, 필터 결과 범위, 혼동행렬 값, 밀기·순서 섞기 실험 값은 `practice/chapter9/results/*.log`에서만 인용한다.

## 6. 외부 시각 자료(열어서 확인함)

| 자료 | 확인한 내용 |
|---|---|
| Setosa, Image Kernels | 얼굴 사진 위에서 커널을 고르거나 직접 숫자를 넣으면 결과가 바로 바뀐다. sharpen, blur, emboss, sobel, outline, identity, custom 일곱 가지 |
| Google ML 실습(한국어), 컨볼루셔널 신경망 | 5x5 위를 3x3이 훑는 애니메이션, stride 2의 2x2 최대 풀링 애니메이션, CNN 전체 구조 그림 |
| vdumoulin, conv_arithmetic | padding·stride 조합별 합성곱 애니메이션 15개 이상, MIT 라이선스 |
| scikit-learn, Recognizing hand-written digits | 우리 실습과 같은 8x8 1797장 데이터를 SVM으로 분류하고 혼동행렬을 그린다 |

빼기로 한 것

- Google Teachable Machine — 본문이 자바스크립트로 그려져 WebFetch로 내용을 확인하지 못했다
- CNN Explainer(poloclub) — `poloclub.gatech.edu` 주소는 404, `poloclub.github.io` 주소는 본문이 읽히지 않았다

주의: Google ML 실습 페이지에는 "2025년 12월 15일에 삭제될 예정"이라는 지원 중단 공지가 붙어 있다. 확인 시점에는 열렸다. 본문 표에 이 사실을 적는다.

## 7. 제출물

수업 중 실습 실시 여부만 확인한다. 보고서를 쓰지 않는다.

1. `results/shift_test.png`
2. `results/filter_gallery.png`
