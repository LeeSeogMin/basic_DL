# 딥러닝 기초 강의자료 하네스

## 1. 목적

이 하네스는 강의자료 작성 지시를 산문 규칙에만 맡기지 않고, 산출물과 실행 증거로 강제하기 위한 장치이다.

이 프로젝트의 주요 위험은 세 가지이다.

1. 실행하지 않은 딥러닝 코드 결과를 본문에 넣는 위험
2. 장별 자료가 강의 목표, 실습 코드, 과제, 결과 해석으로 연결되지 않는 위험
3. 기존 원고가 슬라이드 메모, 교재 설명, 실습 지시문이 뒤섞인 상태로 누적되는 위험

따라서 하네스는 다음을 강제한다.

1. 실행 결과는 실제 로그에서만 인용한다.
2. 장 작업은 계획, 실습, 실행 증거, 원고 반영 순서로 진행한다.
3. 강의자료 산출물은 `docs/`, `lecture/`, `practice/`의 역할을 분리한다.

현재 수준은 Tier 1이다. 완전한 자동 품질 평가 시스템은 아니지만, 실행 증거 게이트와 산출물 게이트로 반복 오류를 줄인다.

## 2. 기본 원칙

- 실행 결과는 `practice/chapter{N}/results/*.log`에서만 인용한다.
- 실행 로그와 함께 소스 해시, 출력 해시, 실행 시각, 플랫폼, 종료코드를 `*.evidence.json`에 남긴다.
- 코드 수정 후에는 `--verify`로 결과가 낡았는지 확인한다.
- 반복 오류는 `.claude/lessons-learned.md` 또는 `.ai/gotchas.md`에 기록한다.

## 3. 장별 워크플로우

| 단계 | 이름 | 필수 산출물 | 역할 |
|---|---|---|---|
| 1 | Planning | `schema/chap{N}.md` | 장 목표, 절 구성, 실습 설계 |
| 2 | Source Review | `docs/curri.md`, `docs/ch{N}.md` 확인 | 기존 자료와 수업계획 정렬 |
| 3 | Research | 필요 시 `content/research/ch{N}-*.md` | 교재, 논문, 공식 문서 확인 |
| 4 | Practice | `practice/chapter{N}/code/` | 실행 가능한 실습 코드 작성 |
| 5 | Evidence | `practice/chapter{N}/results/*.log`, `*.evidence.json` | 실행 결과 증거화 |
| 6 | Lecture Material | `docs/ch{N}.md` 또는 `lecture/chapter{N}.md` | 결과 기반 강의자료 반영 |
| 7 | Verification | `HARNESS_PASS` | 일반 검증 통과 |

## 4. 실행 증거 게이트

실행 증거 게이트는 `scripts/run_and_capture.py`가 담당한다.

```bash
python3 scripts/run_and_capture.py 5
python3 scripts/run_and_capture.py 5 --file 5-1-mlp-basics.py
python3 scripts/run_and_capture.py 5 --verify
```

생성 산출물은 다음과 같다.

| 산출물 | 위치 | 의미 |
|---|---|---|
| 실행 로그 | `practice/chapter{N}/results/{파일}.log` | stdout, stderr, 실행 명령 |
| 증거 JSON | `practice/chapter{N}/results/{파일}.evidence.json` | 소스 해시, 출력 해시, 종료코드, 플랫폼 |
| 결과 파일 | `practice/chapter{N}/results/*.csv`, `*.png` 등 | 표, 그림, 분석 결과의 근거 |

본문의 정확도, 손실값, 표본 수, 실행 시간, 출력 예시는 이 로그에서만 가져온다. 로그가 없으면 결과를 본문에 쓰지 않는다.

## 5. 일반 검증 게이트

일반 검증은 `scripts/harness.sh`를 사용한다.

```bash
./scripts/harness.sh
./scripts/verify.sh
```

`harness.sh`는 프로젝트 타입을 감지해 가능한 검증을 실행한다.

- Python: `ruff`가 있으면 lint, `tests/` 또는 `test_*.py`가 있으면 pytest
- Node.js: `package.json`이 있으면 install, lint, test, build 스크립트
- Makefile: `verify` target이 있으면 실행

성공 시 `HARNESS_PASS`, 빠른 검증 성공 시 `VERIFY_PASS`를 출력한다.

## 6. 강의자료 산출물 게이트

`docs/ch{N}.md` 또는 `lecture/chapter{N}.md`는 다음 요소를 갖추는 것을 권장한다.

```text
# N강 또는 N장. 제목

## 강의 목표
## 핵심 개념
## 직관적 설명
## 최소 실행 예제
## 실행 결과와 해석
## 체크 질문
## 실습 또는 과제
## 핵심 정리
```

기존 `docs/ch{N}.md`가 이 구조와 다르면 한 번에 전면 개편하지 않는다. 장 작업을 할 때 해당 범위 안에서 점진적으로 정리한다.

## 7. 딥러닝 실습 지뢰 목록

- GPU 전용 코드는 금지한다. CUDA, MPS, CPU 순서의 fallback을 둔다.
- 긴 학습을 기본값으로 두지 않는다. 수업용 코드는 작은 데이터셋과 적은 epoch를 기본값으로 한다.
- 랜덤 seed를 고정한다.
- 데이터 다운로드가 필요한 코드는 실패 시 원인을 명확히 출력한다. 조용히 더미 데이터로 대체하지 않는다.
- 합성 데이터는 허용되지만, 본문에서 반드시 합성 데이터라고 표시한다.
- 라이브러리 버전 차이로 출력이 바뀔 수 있으므로 로그에 주요 버전을 출력하는 것을 권장한다.

## 8. 실패 시 행동 규칙

하네스 검증이 실패하면 다음 원칙을 따른다.

1. 누락된 산출물을 명시하고 다음 단계로 진행하지 않는다.
2. 실행 로그가 없으면 본문 수치를 작성하지 않는다.
3. 증거 JSON의 소스 해시가 현재 코드와 다르면 코드를 다시 실행한다.
4. 코드가 실패하면 실패 로그를 남기고 원인을 수정한다.
5. 요청 범위 밖의 문제는 수정하지 말고 보고한다.
