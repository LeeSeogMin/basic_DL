# 4주차 계획서 — 머신러닝 첫걸음: 규칙을 쓰지 않고 분류하기

## 1. 장의 역할

4주차는 딥러닝으로 들어가기 전, "모델이 데이터에서 배운다"는 생각을 학생이 처음으로 손으로 확인하는 장이다. 사람이 규칙을 직접 쓰는 방식과 모델이 판단 경계를 찾는 방식의 차이를 2차원 점 분류로 비교한다.

수학적 엄밀함보다 직관을 우선한다. 다만 용어는 정확하게 쓴다. 모든 개념을 학생이 눈으로 볼 수 있는 점, 색, 선, 표, 오류 사례와 연결한다.

대상은 학부 2학년이다. 1차시 이론 강의, 2차시 실습으로 나눠 진행하고, 각 차시는 75분이다.

## 2. 학습 목표

1. 규칙 기반 방식과 머신러닝 방식의 차이를 설명한다.
2. 분류와 회귀를 구체적 예로 구분한다.
3. 훈련 데이터와 테스트 데이터를 나누는 이유를 설명한다.
4. 기준선 모델을 만들고 새 모델과 비교한다.
5. 정확도만 보고 모델을 평가하면 위험한 이유를 말한다.
6. 혼동행렬을 보고 어느 방향으로 틀렸는지 해석한다.

## 3. 1차시 이론 절 구성

| 절 | 제목 | 핵심 내용 | 시각 자료 |
|---|---|---|---|
| 1 | 규칙을 직접 쓰면 어디서 막히는가 | 조건문의 한계, 예시로 배우는 방식 | 그림 4.1 흐름 비교 |
| 2 | 분류와 회귀 | 답이 이름인가 숫자인가, 질문이 문제 유형을 정한다 | 표 4.2 |
| 3 | 훈련 데이터와 테스트 데이터 | 연습 문제와 시험 문제, stratify | 그림 4.2 분할 도식 |
| 4 | 기준선 | 비교 상대 없는 정확도는 해석 불가 | — |
| 5 | 정확도 하나만 보면 속는다 | 불균형 데이터, 스팸 5% 예시 | — |
| 6 | 혼동행렬 | 놓침과 헛짚음, 두 오류의 무게 차이 | 그림 4.3 칸 이름표 |
| 7 | 직접 움직여 보는 웹 자료 | 외부 시각 자료 4건 | 링크 표 |

## 4. 2차시 실습 설계

실습은 두 개다. 각각 노트북에서 몇 초 안에 끝난다. GPU가 필요 없다.

### 실습 4.1 — 내가 그은 선 vs 모델이 찾은 선

- 파일: `practice/chapter4/code/4-1-simple-classifier.py`
- 데이터: 합성 학생 데이터 160명(공부 시간, 수면 시간 → 시험 통과 여부)
- 학생 조작 지점: 파일 상단의 `MY_SLOPE`, `MY_INTERCEPT`
- 비교 대상 셋: 기준선 / 내가 그은 선 / 로지스틱 회귀가 찾은 선
- 성취 지점: 모델이 찾은 선을 기울기·절편 식으로 꺼내 보고, 자기 선과 겨룬다
- 산출물: `results/decision_boundary.png`, `results/misclassified_examples.csv`

### 실습 4.2 — 정확도 95% 모델의 정체

- 파일: `practice/chapter4/code/4-2-accuracy-trap.py`
- 데이터: 합성 메일 1000통, 스팸 50통(5%)
- 비교 대상 셋: 무조건 정상 / 기본 로지스틱 회귀 / `class_weight="balanced"`
- 교훈 지점: 정확도가 가장 높은 모델이 스팸의 3분의 2를 놓친다
- 산출물: `results/confusion_matrix.png`, `results/accuracy_trap_table.csv`

### 개념 그림 생성

- 파일: `practice/chapter4/code/4-0-concept-figures.py`
- 산출물: `results/fig4-1-rule-vs-learning.png`, `results/fig4-2-train-test-split.png`, `results/fig4-3-confusion-guide.png`
- 한글 폰트 도우미: `practice/chapter4/code/_krfont.py` (폰트가 없으면 영어 라벨로 자동 전환)

## 5. 실행 증거

```bash
python scripts/run_and_capture.py 4
python scripts/run_and_capture.py 4 --verify
```

본문의 정확도, 표본 수, 혼동행렬 값은 `practice/chapter4/results/*.log`에서만 인용한다.

## 6. 외부 시각 자료(열어서 확인함)

| 자료 | 확인한 내용 |
|---|---|
| MLU-Explain, Train/Test/Validation Sets | 훈련 데이터의 점을 드래그하면 경계선이 갱신되는 인터랙션 |
| Google ML 단기집중과정(한국어), 데이터 세트 분할 | 2분할·3분할 비교 그림 두 장 |
| Google ML 단기집중과정(한국어), 정확도·정밀도·재현율 | 혼동행렬 기반 정확도 정의, 1% 클래스에서 99% 정확도 예시 |
| scikit-learn, Classifier comparison | 분류기별 결정 경계 격자 그림, BSD-3-Clause |

## 7. 제출물

수업 중 실습 실시 여부만 확인한다. 보고서를 쓰지 않는다.

1. `results/decision_boundary.png`
2. `results/confusion_matrix.png`
