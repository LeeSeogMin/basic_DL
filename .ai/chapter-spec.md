# 장별 강의자료 작성 규격

이 문서는 `docs/ch{N}.md` 강의자료를 만드는 모든 작업자가 따르는 공통 규격이다.
기준 사례는 **4장**이다. 형식·문체·분량이 헷갈리면 4장을 그대로 따른다.

---

## 0. 먼저 읽을 것

1. `docs/ch4.md` — 형식·문체·분량의 유일한 기준. 전부 읽는다.
2. `practice/chapter4/code/` 의 `_krfont.py`, `4-0-concept-figures.py`, `4-1-simple-classifier.py`, `4-2-accuracy-trap.py` — 코드 스타일 기준.
3. `contents.md` 의 해당 주차 절 — 내용 근거.
4. `강의계획서.md` 의 주차별 계획 표 — 그 주차의 이론·실습·제출물.
5. `AGENTS.md`, `harness.md` — 저장소 절대 규칙.
6. `schema/chap4.md` — 장 계획서 형식.

---

## 1. 만들어야 하는 산출물

| 경로 | 내용 |
|---|---|
| `schema/chap{N}.md` | 장 계획서. `schema/chap4.md` 형식을 따른다 |
| `practice/chapter{N}/code/_krfont.py` | `practice/chapter4/code/_krfont.py`를 그대로 복사 |
| `practice/chapter{N}/code/{N}-0-concept-figures.py` | 이론용 개념 그림 2~3장 생성 |
| `practice/chapter{N}/code/{N}-1-*.py` | 실습 1 |
| `practice/chapter{N}/code/{N}-2-*.py` | 실습 2 |
| `practice/chapter{N}/results/` | 실행 로그·증거 JSON·그림 (직접 만들지 않고 아래 게이트로 생성) |
| `docs/ch{N}.md` | 강의자료 본문 |

---

## 2. 절대 규칙

- 실행하지 않은 코드 결과를 본문에 쓰지 않는다. 본문의 모든 수치·출력·표는 `practice/chapter{N}/results/*.log` 에서만 인용한다.
- 코드를 다 쓰면 실행 증거를 만든다.

```bash
python scripts/run_and_capture.py {N}
python scripts/run_and_capture.py {N} --verify
```

  `--verify` 가 **문제 0건**이어야 한다. 실패하면 코드를 고쳐 다시 실행한다.

- 생성한 PNG를 **Read 도구로 열어 눈으로 확인한다.** 글자 깨짐·겹침·잘림·범례가 데이터를 가리는 문제가 있으면 코드를 고쳐 다시 실행한다.
- 합성 데이터를 쓰면 코드 주석과 본문 양쪽에 "합성 데이터"라고 밝힌다.
- CPU에서 1분 안에 끝나야 한다. GPU 코드를 쓰지 않는다. 난수 seed를 고정한다.
- 쓸 수 있는 패키지: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`. 이미 설치돼 있다.
  `torch`, `tensorflow`, 그 밖의 새 패키지를 설치하거나 쓰지 않는다.
- 요청받은 장 밖의 파일을 고치지 않는다. 다른 장의 `practice/chapter*/`, `docs/ch*.md`, `schema/chap*.md` 를 건드리지 않는다.
- `강의계획서.md`, `contents.md`, `AGENTS.md`, `harness.md` 를 고치지 않는다. 고칠 필요를 발견하면 보고만 한다.

---

## 3. `docs/ch{N}.md` 구조

4장과 똑같이 만든다. 이론과 실습을 한 파일에 담는다.

```
# 제{N}장 제목 — 부제

---

## [전반부] 이론 강의
---

### 학습 목표
(불릿 4~5개)

(앞 장과 이어지는 도입 문단 2~3줄)

---

### 1. 절 제목
#### 1-1. 하위절          <- 필요할 때만
...
### 2. 절 제목
...
(이론 절은 5~7개)

### {마지막 번호}. 직접 움직여 보는 웹 자료
(외부 시각 자료 표)

---

## [후반부] 실습
---

(실습 두 개를 소개하는 2~3줄)

#### 실행 환경

### 실습 {N}.1: 제목
#### 실습 목표
#### 데이터
#### 실행 명령
#### 핵심 코드
#### 실행 결과
#### 결과 읽기
#### 해 보기

### 실습 {N}.2: 제목
(같은 구성)

---

### 책임 있는 AI 체크
(`**표 {N}.{k}** 제출 전 점검` 표. 그 장에서 실제로 다룬 위험만 5~6줄)

---

### 과제 (LMS 제출)
#### 제출물
#### 평가 기준
#### 마감

---

### 핵심 정리
(개념 | 한 줄 정리 표)

(다음 장 예고 한 줄)
```

---

## 4. 작성 원칙

- 대상은 **학부 2학년**이다. 최대한 쉽게 쓴다. 수식을 앞세우지 않고 그림과 실행 결과로 설명한다. 어려운 용어는 처음 나올 때 한 줄로 푼다.
- 실습의 목표는 **성취감**이다. 학생이 값 하나를 바꾸면 결과가 눈에 띄게 달라지도록 만든다. 코드 맨 위에 학생이 바꿀 상수를 모아 두고 주석으로 표시한다.
- **시간 표시를 하지 않는다.** "10분", "50분", "전반부 40분" 같은 표현을 쓰지 않는다.
- 과제는 가볍게 낸다. 수업 중 실습을 실행했는지만 확인한다. 실습을 실행하면 만들어지는 **결과 파일 2개**를 LMS에 올리는 것으로 끝낸다. 보고서·분석문·성찰문을 요구하지 않는다.
- 그림은 이렇게 넣는다.

```markdown
![짧은 설명](../practice/chapter{N}/results/파일명.png)

**그림 {N}.1** 그림이 무엇을 보여주는지 한 줄
```

- **`### 책임 있는 AI 체크`를 반드시 넣는다.** 실습 뒤, 과제 앞에 놓는다. 그 장에서 학생이 실제로 만난 위험만 적는다.
  다른 장의 항목을 베끼지 않는다. 로그에 없는 수치를 점검 항목에 쓰지 않는다.
- 표 위에는 `**표 {N}.1** 설명` 문단을 단다.
- 실행 로그를 인용한 뒤에는 `_출처: practice/chapter{N}/results/파일명.log_` 를 붙인다.
- `○ **소제목**` 뒤에 `  - 항목` 들여쓰기 불릿을 쓴다.
- **`○` 로 시작하는 줄이 연달아 나오면 사이에 빈 줄을 넣는다.** 안 넣으면 한 문단으로 합쳐져 PDF에서 붙어 나온다.

---

## 5. 한국어 문체

| 쓰지 않음 | 쓴다 |
|---|---|
| 이 방법은 정확도 향상을 제공한다 (무생물 주어) | 이 방법을 쓰면 정확도가 올라간다 |
| ~되어진다 / ~로 여겨진다 (피동 남용) | ~한다 / ~로 본다 |
| 높은 신뢰도를 가진다 | 신뢰도가 높다 |
| 성능에 있어서의 개선 / ~에 대한 | 성능이 좋아진 정도 |
| 분석을 수행한다 (대동사) | 분석한다 |
| 핵심 요인 중 하나이다 | 핵심 요인이다 |
| 모델이 확신한다 (의인화) | 예측 분산이 작다 |
| ~해 보자 / 짚어 두자 / 함정이 하나 있다 (구어체) | 다음 순서로 확인한다 / 여기에 조건이 하나 붙는다 |
| 훌륭하다 / 깔끔하다 / 완벽하다 (평가어) | 오차가 0이다 / 구조가 단순하다 |
| 크게 개선 | 0.71 → 0.78 |

- 한 문장에 한 가지만 담는다. 60자를 넘으면 끊는다.
- 결론 문장에는 누가 · 무엇을 한다가 들어간다. 추상명사로 문장을 끝내지 않는다.
- 검증하지 않은 수치·사례를 지어내지 않는다. 확인이 안 되면 추상적으로 남긴다.

---

## 6. 외부 웹 자료

마지막 이론 절에 시각 자료 표를 넣는다. 열은 `자료 | 무엇을 볼 수 있는가 | 주소` 세 개다.

- **WebFetch로 실제 열어 내용을 확인한 자료만 넣는다.** 확인하지 못했거나 본문이 안 읽히는 페이지는 표에서 뺀다. 페이지에 없는 내용을 있다고 쓰지 않는다.
- 3~4개만 고른다. 이 장 주제와 직접 맞는 것만 넣는다.
- 후보(모두 무료)
  - MLU-Explain — `https://mlu-explain.github.io/` 아래에 neural-networks, logistic-regression, linear-regression, roc-auc, cross-validation, train-test-validation, precision-recall, random-forest, decision-tree, bias-variance, double-descent, reinforcement-learning, equality-of-odds 가 있다
  - Google 머신러닝 단기집중과정 한국어판 — `https://developers.google.com/machine-learning/crash-course/...?hl=ko`
  - scikit-learn 공식 예제 — `https://scikit-learn.org/stable/auto_examples/...`
  - TensorFlow Playground — `https://playground.tensorflow.org/`
  - Google Teachable Machine — `https://teachablemachine.withgoogle.com/`
- 외부 그림을 강의자료에 직접 걸지 않는다. 주소만 표에 적는다. 그림은 우리가 직접 만든 것만 넣는다.

---

## 7. 그림 만들기

`practice/chapter4/code/4-0-concept-figures.py` 를 본보기로 삼는다.

- `_krfont.py` 를 `import _krfont` 로 불러 `ko = _krfont.setup()` 을 호출한다. 한글 폰트가 없는 환경에서는 영어 라벨로 자동 전환된다.
- 모든 라벨에 `_krfont.label("한글", "english", ko)` 를 쓴다.
- 색은 4장과 맞춘다: 파랑 `#4C78A8`, 주황 `#F58518`, 초록 `#54A24B`, 빨강 `#E45756`, 회색 `#79706E`.
- 범례가 데이터를 가리면 축 아래로 뺀다: `ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)`
- 개념 그림 파일명은 `fig{N}-1-이름.png` 형식으로 짓는다.

---

## 8. 완료 보고에 담을 것

1. 만든 파일 목록
2. `run_and_capture.py {N} --verify` 결과 (문제 0건인지)
3. 실습 두 개의 핵심 실행 수치 (로그에서 인용)
4. 웹 자료 중 확인에 실패해서 뺀 것이 있으면 그 목록
5. 규격을 못 지킨 곳이 있으면 무엇을 왜 못 지켰는지
