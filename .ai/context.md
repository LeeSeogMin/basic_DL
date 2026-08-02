# Project Context

## Project Goal

딥러닝 기초 강의자료 구성 프로젝트. 기존 `docs/` 자료를 강의 목표, 핵심 개념, 실습 코드, 실행 결과, 과제/프로젝트 자료로 정리한다.

## Current State

- 2026-07-05: `geoAI`의 실행 증거 하네스 구조를 참조해 이 프로젝트용 Tier 1 harness를 도입했다.
- 2026-07-05: 루트 `.venv`를 생성하고 `practice/requirements.txt` 공통 의존성을 설치했다. import smoke test 통과: Python 3.14.6, NumPy 2.5.1, Pandas 3.0.3, scikit-learn 1.9.0, SciPy 1.18.0.
- 2026-07-05: 새 강의계획 정본 `contents.md`를 작성했다. 방향은 학부 2학년 대상 딥러닝 기초 + 매주 흥미로운 실습 + VS Code/GitHub/Copilot/Antigravity 등 AI 코딩 도구 활용 + AI 결과 검증 역량이다. `docs/curri.md`는 과거 계획서로만 보관한다.
- 2026-07-05: `docs/curri.md`를 `contents.md` 기반 학생 공지/운영용 강의계획서로 갱신했다. 교과목 정보, 수업 개요, 평가, AI 활용 윤리, 15주 계획, 중간/기말 프로젝트 기준을 포함한다.
- 2026-07-05: 강의 목적을 "이미 AI를 일상적으로 쓰는 학생들이 원리를 이해하고, 더 전문적으로 활용하며, 폐해를 줄이는 책임 있는 사용자가 되게 하는 것"으로 강화했다. `contents.md`와 `docs/curri.md`에 책임 있는 AI 사용 축(환각, 편향, 개인정보, 저작권, 보안, 과도한 의존성)을 반영했다.
- 2026-07-05: 2주차 시범 장을 작성했다. `schema/chap2.md`, `docs/ch2.md`, `practice/chapter2/code/2-1-hello-ai.py`를 만들고 `run_and_capture.py 2`로 실행 증거를 생성했다. `docs/ch2.md`는 직관적·구체적 설명, 학술적 평서문, 책임 있는 AI 사용 체크를 표준으로 삼는다.
- 2026-07-05: 4주차 `docs/ch4.md` 작성 완료. `schema/chap4.md`, `practice/chapter4/code/4-1-simple-classifier.py` 생성, `run_and_capture.py 4` 실행 증거 생성. 실행값: 전체 160, 훈련 112, 테스트 48, 기준선 정확도 0.521, 로지스틱 회귀 테스트 정확도 0.917, 혼동행렬 `[[24, 1], [3, 20]]`, 틀린 사례 CSV 4건. 그림: `decision_boundary.png`, `confusion_matrix.png`.
- 현재 원천 자료는 `docs/curri.md`와 `docs/ch1.md`-`docs/ch14.md`이다.
- 새 작업 구조는 `schema/`, `content/`, `practice/`, `lecture/`, `scripts/`, `.ai/`, `.claude/`를 사용한다.

## Important Constraints

- 실행 결과는 `scripts/run_and_capture.py`로 생성한 로그에서만 인용한다.
- 합성 데이터는 허용되지만 반드시 합성 데이터라고 표시한다.
- 무거운 딥러닝 프레임워크는 공통 의존성에 강제하지 않는다.
- 기존 `docs/` 원고는 요청 범위 안에서만 점진적으로 정리한다.

## Recent Decisions

- 2026-07-05: `geoAI`의 GeoAI 정체성/정책장 게이트는 이 프로젝트에 맞지 않아 제외했다.
- 2026-07-05: `run_and_capture.py`는 장 폴더 `chapterN`과 `chapterNN`을 모두 지원하도록 구성했다.
- 2026-07-05: 일반 검증은 `scripts/harness.sh`, 실습 실행 증거는 `scripts/run_and_capture.py`로 분리했다.
- 2026-07-05: 현재 저장소는 git 저장소가 아니므로 커밋/상태 추적은 아직 없다.
- 2026-07-05: 장별 원고 작성 우선순위는 `docs/ch2.md` 환경설정 문서부터 시작한다. 현재 `docs/ch8.md`가 없어 중간 미니 프로젝트 문서로 새로 생성해야 한다.
- 2026-07-05: 11주차는 "생성형 AI의 원리와 폐해", 12주차는 "AI 코딩 에이전트와 책임 있는 자동화"로 강화한다.
- 2026-07-05: 다음 원고 후보는 `docs/ch1.md` 정리 또는 `docs/ch3.md` Python과 데이터 장 작성이다. 2주차와 4주차 문체와 구조를 기준으로 삼는다.
