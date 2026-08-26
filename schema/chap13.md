# 13주차 계획서 — 나만의 AI 앱 만들기: 데모와 사용자 경험

## 1. 장의 역할

13주차는 지금까지 만든 모델을 남이 쓸 수 있는 형태로 감싸는 장이다. 새 알고리즘을 배우지 않는다. 4장부터 12장까지 만든 분류기를 그대로 두고, 그 앞뒤에 무엇을 붙여야 다른 사람이 결과를 이해하고 믿을지를 다룬다.

기술 난도보다 판단 난도가 높은 장이다. 확률을 보여줄지, 자신 없을 때 답을 미룰지, 무엇을 입력받지 않을지는 코드가 아니라 사람이 정한다. 그 판단을 숫자로 확인할 수 있게 실습 13.2에서 맞바꿈을 직접 센다.

대상은 학부 2학년이다. 1차시 이론 강의, 2차시 실습으로 나눠 진행한다.

## 2. 학습 목표

1. 모델을 실행하는 것과 남이 쓸 수 있는 앱을 만드는 것의 차이를 다섯 단계로 설명한다.
2. 앱의 입력 범위를 정하고, 범위를 벗어난 입력을 거절하는 코드를 쓴다.
3. 예측 하나에 확률과 판단 근거와 한계 안내를 함께 붙인다.
4. 확신 기준선을 정해 자신 없는 예측을 "판단 보류"로 돌린다.
5. 기준선을 올릴 때 틀린 답과 보류 건수가 맞바뀌는 것을 표와 그림으로 읽는다.
6. 남이 같은 결과를 얻을 수 있게 실행 조건을 적는다.
7. 개인정보를 요구하지 않는 입력 구조를 설계한다.

## 3. 1차시 이론 절 구성

| 절 | 제목 | 핵심 내용 | 시각 자료 |
|---|---|---|---|
| 1 | 모델과 앱은 무엇이 다른가 | 예측 한 줄 대 다섯 단계, gradio·streamlit 소개(설치는 하지 않음) | 그림 13.1 다섯 단계 |
| 2 | 입력과 출력을 정하는 법 | 입력 범위 표, 거절 문구, 출력에 넣을 네 가지 | 표 13.1, 표 13.2 |
| 3 | 사용자가 오해하지 않게 하는 문구 | 단정하는 화면과 확률·근거·한계를 적은 화면 | 그림 13.2 문구 비교 |
| 4 | 자신 없을 때 말하는 법 | 확신 = max(p, 1-p), 기준선, 판단 보류, 4장과의 연결 | — |
| 5 | 남이 실행할 수 있게 만들기 | 폴더·패키지 목록·seed·실행 명령 네 가지 | 그림 13.3 재현 가능한 실행 |
| 6 | 앱을 만들 때 피할 것 | 개인정보를 요구하는 구조, 대체 설계 | 표 13.3 |
| 7 | 직접 움직여 보는 웹 자료 | 외부 시각 자료 4건 | 링크 표 |

## 4. 2차시 실습 설계

실습은 두 개다. 각각 노트북에서 몇 초 안에 끝난다. GPU가 필요 없다.
**gradio·streamlit을 설치하지 않는다.** 설치 여부에 수업이 좌우되면 안 되므로 데모는 콘솔로 만들고 화면을 그림으로 남긴다.

### 실습 13.1 — 분류기를 데모 앱으로 감싸기

- 파일: `practice/chapter13/code/13-1-demo-app.py`
- 데이터: 합성 메일 1200통(스팸 480통, 40%), 특징 세 개 — 링크 개수, 대문자 비율, 느낌표 개수
- 학생 조작 지점: 파일 상단의 `MY_INPUTS`(입력 목록), `MODEL_LIMITS`(한계 안내문)
- 앱이 하는 일 다섯 가지: 입력 받기 → 입력 검사 → 예측 → 확률·근거 보이기 → 한계 알리기
- 성취 지점: 자기가 넣은 입력이 카드 그림으로 나오고, 잘못된 입력은 앱이 거절한다
- 산출물: `results/demo_app_cards.png`, `results/demo_app_results.csv`

### 실습 13.2 — 자신 없을 때 답을 미루는 앱

- 파일: `practice/chapter13/code/13-2-honest-output.py`
- 데이터: 실습 13.1과 같은 합성 메일 데이터, 같은 모델
- 학생 조작 지점: `CONFIDENCE_THRESHOLD`(확신 기준선), `THRESHOLD_SWEEP`(비교할 기준선 목록)
- 교훈 지점: 기준선 0.50 → 0.95 로 올리면 틀린 답이 46건에서 3건으로 줄지만, 답하지 못한 건수가 0건에서 274건으로 는다
- 4장과의 연결: 4장은 무엇을 놓쳤는지 세는 데서 멈췄고, 여기서는 놓칠 것 같으면 답하지 않는 선택지를 추가한다
- 산출물: `results/confidence_tradeoff.png`, `results/threshold_table.csv`

### 개념 그림 생성

- 파일: `practice/chapter13/code/13-0-concept-figures.py`
- 산출물: `results/fig13-1-model-vs-app.png`, `results/fig13-2-output-honesty.png`, `results/fig13-3-reproducible-run.png`
- 한글 폰트 도우미: `practice/chapter13/code/_krfont.py` (폰트가 없으면 영어 라벨로 자동 전환)
- 확신 기준선의 맞바꿈은 개념 도식으로 그리지 않는다. 지어낸 곡선 대신 실습 13.2가 실제로 센 값으로 그린다

## 5. 실행 증거

```bash
python scripts/run_and_capture.py 13
python scripts/run_and_capture.py 13 --verify
```

본문의 확률, 건수, 정확도, 기준선 표는 `practice/chapter13/results/*.log`에서만 인용한다.

## 6. 외부 시각 자료(열어서 확인함)

| 자료 | 확인한 내용 |
|---|---|
| Gradio Quickstart | `pip install --upgrade gradio` 설치 명령, `gr.Interface` 예제, localhost:7860 에서 열리는 화면 설명 |
| Streamlit, Installation | `pip install streamlit`, `streamlit hello` 두 명령, Anaconda·Codespaces 설치 경로 |
| Google PAIR, Explainability + Trust | 확신도 표시 방식 네 가지(범주형·N-best·백분율·불확실성 시각화), 표시 전 사용자 테스트 권고 |
| scikit-learn, Probability calibration curves | 예측 확률이 실제 빈도와 얼마나 맞는지 재는 곡선, 보정 전후 비교 그림 |

확인에 실패해 뺀 자료: Google Model Cards(`modelcards.withgoogle.com/about`은 `deepmind.google/models/model-cards`로 넘어가고, 모델 카드가 담는 항목이 페이지에 적혀 있지 않다).

## 7. 제출물

수업 중 실습 실시 여부만 확인한다. 보고서를 쓰지 않는다.

1. `results/demo_app_cards.png`
2. `results/confidence_tradeoff.png`

## 8. 책임 있는 AI 체크

`docs/ch13.md`의 `### 책임 있는 AI 체크` 절에 제출 전 점검 표를 둔다. 남이 쓰는 화면을 만드는 주다. 화면에 무엇이 적혀 있는지가 점검 대상이다.

점검 항목: 한계 안내, 과장 금지, 신뢰도 표시, 학습 범위, 사용 제한, 개인정보
