# 14주차 계획서 — 기말 프로젝트 제작 워크숍: 작지만 실행되는 결과물

## 1. 장의 역할

14주차는 새 개념을 가르치는 장이 아니다. 4주차부터 배운 것을 학생이 자기 프로젝트로 굴리게 하는 장이다. 1차시에서 프로젝트를 어떻게 짜는지 정리하고, 2차시에서 뼈대 코드를 자기 데이터로 바꿔 끼운 다음 제출 전 점검까지 마친다.

강의계획서의 기말 프로젝트 필수 요건 7가지와 "피해야 할 결과물" 목록을 이 장의 근거로 삼는다. 모델을 크게 만드는 방향이 아니라, 문제·데이터·모델·평가·한계 다섯 칸을 모두 채우는 방향으로 이끈다.

대상은 학부 2학년이다. 1차시 이론 강의, 2차시 실습으로 나눠 진행한다.

## 2. 학습 목표

1. 자기 프로젝트의 문제를 한 문장으로 쓴다.
2. 데이터를 어디서 구할지 정하고, 표본 수와 특징 수를 확인한다.
3. 프로젝트 크기를 어디까지 줄여야 2주 안에 끝나는지 판단한다.
4. 평가 방법과 기준선을 모델을 만들기 전에 정한다.
5. AI 도구를 어디에 썼고 무엇을 직접 검증했는지 기록한다.
6. 개인정보·출처·편향·보안·과도한 의존 다섯 가지 위험 중 자기 프로젝트에 해당하는 것을 짚는다.

## 3. 1차시 이론 절 구성

| 절 | 제목 | 핵심 내용 | 시각 자료 |
|---|---|---|---|
| 1 | 무엇을 만들 것인가 | 문제를 한 문장으로, 입력과 출력을 못 쓰면 아직 문제가 아니다 | 그림 14.1 다섯 칸 뼈대 |
| 2 | 데이터를 어디서 구하는가 | 만들기·모으기·공개 데이터 세 갈래, 출처와 이용 조건 확인 | 표 14.2 |
| 3 | 얼마나 작게 시작해야 하는가 | 한 번 실행되는 최소 결과물을 먼저 만든다, 줄이는 세 가지 손잡이 | — |
| 4 | 평가를 미리 정한다 | 기준선을 먼저 계산, 지표를 나중에 고르면 결과에 맞춰 고르게 된다 | — |
| 5 | AI 활용 과정을 남기는 법 | 도구·프롬프트·도와준 부분·검증한 부분·이해 못 한 부분 다섯 줄 | — |
| 6 | 위험 점검 | 개인정보·출처·편향·보안·과도한 의존, 되돌릴 수 있는 것과 없는 것 | 그림 14.2 갈림길, 그림 14.3 점검 여섯 가지 |
| 7 | 직접 움직여 보는 웹 자료 | 외부 자료 4건 | 링크 표 |

## 4. 2차시 실습 설계

실습은 두 개다. 각각 노트북에서 몇 초 안에 끝난다. GPU가 필요 없다.

### 실습 14.1 — 프로젝트 뼈대 한 파일

- 파일: `practice/chapter14/code/14-1-project-skeleton.py`
- 데이터: 합성 리뷰 데이터 300줄(긍정어수·부정어수·느낌표수·글자수 → 평점 좋음/나쁨)
- 학생 조작 지점: 파일 상단의 `DATA_SOURCE`, `CSV_PATH`, `LABEL_COLUMN`, `N_SAMPLES`, `TEST_RATIO`, `HIDDEN_UNITS`
- 구조: 데이터 불러오기 → 훈련/테스트 나누기 → 기준선 → 모델 학습 → 정확도와 혼동행렬 → 틀린 사례 저장
- 자기 데이터로 바꾸는 지점: `load_data()` 함수 하나. `DATA_SOURCE = "csv"` 로 두면 자기 CSV를 읽는다
- 성취 지점: 4주차 기준선, 5주차 은닉층, 7주차 훈련/테스트 비교가 한 파일에서 한꺼번에 나온다
- 산출물: `results/project_result.png`, `results/project_wrong_examples.csv`, `results/project_input_example.csv`

### 실습 14.2 — 제출 전 자동 점검

- 파일: `practice/chapter14/code/14-2-submission-check.py`
- 점검 대상: `practice/chapter14/example_project`(통과 예시)와 `practice/chapter14/example_project_rough`(문제를 일부러 넣은 예시)
- 학생 조작 지점: 파일 상단의 `PROJECT_DIR`, `COMPARE_DIR`
- 점검 여섯 항목: 실행되는 .py / 결과 파일 / 입력 데이터 / 비밀키 / 실명·연락처·학번 / 사용 방법 문서
- 안전 규칙: 파일을 읽기만 한다. 고치거나 지우지 않는다. 파이썬 파일도 실행하지 않고 `ast.parse` 로 문법만 본다
- 교훈 지점: 두 폴더가 6/6과 1/6으로 갈린다. 미통과 항목은 파일 이름과 줄 번호까지 나온다
- 산출물: `results/submission_check.png`, `results/submission_check_table.csv`

### 개념 그림 생성

- 파일: `practice/chapter14/code/14-0-concept-figures.py`
- 산출물: `results/fig14-1-project-skeleton.png`, `results/fig14-2-good-vs-avoid.png`, `results/fig14-3-submission-check.png`
- 한글 폰트 도우미: `practice/chapter14/code/_krfont.py` (폰트가 없으면 영어 라벨로 자동 전환)

### 예시 프로젝트 폴더

실습 14.2가 읽는 교재용 폴더다. 두 폴더 모두 사람이 지어낸 내용이며 실제 학생 자료가 아니다.

| 폴더 | 넣어 둔 것 |
|---|---|
| `example_project/` | `README.md`, `run.py`, `data/input_example.csv`, `results/run.log`, `results/scores.csv` |
| `example_project_rough/` | `app.py`(가짜 키 두 줄), `half_written.py`(문법 오류), `participants.csv`(이름·학번·연락처 열) |

`example_project_rough/half_written.py` 는 문법 오류를 일부러 남긴 교재용 파일이므로 고치지 않는다.

## 5. 실행 증거

```bash
python scripts/run_and_capture.py 14
python scripts/run_and_capture.py 14 --verify
```

본문의 정확도, 표본 수, 혼동행렬 값, 점검 통과 수는 `practice/chapter14/results/*.log`에서만 인용한다.

## 6. 외부 시각 자료(열어서 확인함)

| 자료 | 확인한 내용 |
|---|---|
| scikit-learn, Toy datasets | 내려받기 없이 쓰는 데이터 6개와 각각의 표본 수·특징 수 |
| UCI Machine Learning Repository | 데이터셋 689개, 분류·회귀·군집과 표본 수·특징 수로 거르기 |
| 공공데이터포털 | 파일데이터·오픈API·표준데이터 세 형식, 데이터 신청과 분쟁조정 절차 |
| scikit-learn, Common pitfalls and recommended practices | 전처리 불일치·데이터 누출·`random_state` 고정을 잘못된 코드와 올바른 코드로 비교 |

## 7. 제출물

수업 중 실습 실시 여부만 확인한다. 보고서를 쓰지 않는다.

1. `results/project_result.png`
2. `results/submission_check.png`

## 8. 책임 있는 AI 체크

`docs/ch14.md`의 `### 책임 있는 AI 체크` 절에 제출 전 점검 표를 둔다. 기말 프로젝트에 그대로 쓰는 표다. 제출 전에 한 줄씩 짚는다.

점검 항목: 성능 과장, 실패 기록, 과적합 확인, AI 기록, 개인정보, 보안, 한계 안내
