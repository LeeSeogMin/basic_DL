# AGENTS.md

이 저장소에서 작업할 때는 이 파일을 우선 따른다. 장기 규칙은 `CODEX.md`, 실행 증거와 단계 게이트는 `harness.md`를 함께 확인한다.

## 1. 프로젝트 성격

- 과목: 딥러닝 기초
- 목적: 15주 강의자료, 실습 코드, 과제/프로젝트 자료를 일관된 구조로 구성한다.
- 현재 원천 자료: `docs/ch{N}.md`, `docs/curri.md`
- 기본 산출물:
  - `docs/`: 장별 강의 원고와 수업 자료의 단일 원천
  - `lecture/`: 학부 수업용 재구성 자료, 활동, 슬라이드 초안
  - `practice/chapter{N}/`: 실행 가능한 실습 코드, 데이터, 결과 로그
  - `schema/`: 장별 구성 계획
  - `content/`: 리서치, 초안, 그래픽, 리뷰 결과

## 2. 절대 규칙

- 실행하지 않은 코드 결과를 본문에 쓰지 않는다.
- 임의 숫자, 가짜 출력, "예시 결과"를 실제 실행 결과처럼 쓰지 않는다.
- 참고문헌, 논문, URL, DOI는 확인한 것만 쓴다.
- 플랫폼 종속 경로를 코드에 하드코딩하지 않는다. Python 코드는 `pathlib.Path`를 우선 사용한다.
- 요청 범위 밖의 리팩토링, 파일명 변경, 원고 대수정은 하지 않는다. 발견 사항은 보고한다.

## 3. Harness 필수 규칙

- 실습 코드를 작성하거나 수정하면 `scripts/run_and_capture.py`로 실행 로그와 증거 JSON을 남긴다.
- 본문에 들어가는 수치, 모델 성능, 표의 실행 결과는 `practice/chapter{N}/results/*.log`에서만 인용한다.
- 코드 수정 후 기존 결과를 재사용하려면 `python3 scripts/run_and_capture.py N --verify`로 소스 해시 일치를 확인한다.
- 일반 검증은 `./scripts/harness.sh`를 실행한다. 성공 시 `HARNESS_PASS`가 출력되어야 한다.

## 4. 장별 작업 흐름

장 작성 또는 큰 개정은 아래 순서로 진행한다.

1. `docs/curri.md`와 해당 `docs/ch{N}.md`를 확인한다.
2. `schema/chap{N}.md`에 장 목표, 절 구성, 실습 계획을 정리한다.
3. 필요한 경우 `content/research/`에 최신 자료와 참고문헌을 정리한다.
4. 실습 코드는 `practice/chapter{N}/code/`에 둔다.
5. `scripts/run_and_capture.py`로 실행 증거를 만든다.
6. 결과를 근거로 `docs/ch{N}.md` 또는 `lecture/chapter{N}.md`를 수정한다.
7. `./scripts/harness.sh`로 최소 검증을 수행한다.

## 5. 강의자료 작성 기준

- 첫 화면은 학습자가 바로 수업 흐름을 파악할 수 있어야 한다.
- 각 장은 **Part A(강의, 1교시 90분)**와 **Part B(실습, 2교시 90분)** 두 파트로 구성한다.
  - **Part A — 강의**: 강의 목표, 핵심 개념, 직관적 설명, 최소 실행 예제, 핵심 정리
  - **Part B — 실습**: 실습 목표, 실습 절차, AI 활용 가이드, 결과 해석, 책임 있는 AI 체크, 제출물
- 본문에는 핵심 코드만 짧게 넣고 전체 코드는 `practice/`에 둔다.
- 수업 운영 메모와 교재형 설명이 섞이면 `lecture/`와 `docs/`로 역할을 분리한다.

## 6. 딥러닝 실습 기준

- CPU에서도 수업 시간 안에 실행 가능한 예제를 우선한다.
- 긴 학습은 기본값을 작게 둔다. 예: 작은 데이터셋, 적은 epoch, 고정 seed.
- 무거운 프레임워크는 필요한 장에서만 사용한다. 공통 의존성은 `practice/requirements.txt`를 기준으로 한다.
- 랜덤성을 쓰는 코드는 seed를 고정하고, 실행 로그에 주요 설정을 출력한다.
- GPU가 있으면 사용할 수 있지만, CPU fallback이 있어야 한다.

## 7. 작업 메모리

작업 시작 시 필요한 경우 `.ai/context.md`, `.ai/todo.md`, `.ai/gotchas.md`를 확인한다. 작업 중 새로 발견한 반복 함정은 `.ai/gotchas.md` 또는 `.claude/lessons-learned.md`에 남긴다.
