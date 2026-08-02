# CODEX.md

딥러닝 기초 강의자료 구성 프로젝트의 Codex 작업 지침이다. 구체적 실행 게이트는 `harness.md`, 저장소 운영 규칙은 `AGENTS.md`를 우선한다.

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r practice/requirements.txt

./scripts/harness.sh
```

특정 장의 실습 코드를 실행 증거와 함께 남기려면 다음을 사용한다.

```bash
python3 scripts/run_and_capture.py 5
python3 scripts/run_and_capture.py 5 --file 5-1-mlp-basics.py
python3 scripts/run_and_capture.py 5 --verify
```

## 프로젝트 구조

```text
basic_DL/
├── AGENTS.md
├── CODEX.md
├── CLAUDE.md
├── harness.md
├── docs/
│   ├── curri.md
│   └── ch{N}.md
├── lecture/
│   ├── chapter{N}.md
│   ├── assets/
│   └── slides/
├── schema/
│   └── chap{N}.md
├── content/
│   ├── research/
│   ├── drafts/
│   ├── graphics/
│   └── reviews/
├── practice/
│   ├── requirements.txt
│   └── chapter{N}/
│       ├── code/
│       ├── data/
│       └── results/
└── scripts/
    ├── run_and_capture.py
    ├── harness.sh
    └── verify.sh
```

## 작성 원칙

### 강의자료 모드

- 긴 산문보다 수업 흐름이 보이는 구조를 우선한다.
- 개념은 "왜 필요한가 -> 직관 -> 최소 예제 -> 결과 해석" 순서로 설명한다.
- 학부생이 따라올 수 있게 진입점은 쉽게 두되, 개념의 정확성을 낮추지 않는다.
- 장마다 체크 질문, 짧은 실습, 핵심 정리를 둔다.

### 코드와 결과

- 본문에는 3-8줄의 핵심 코드만 넣는다.
- 전체 코드는 `practice/chapter{N}/code/{파일명}.py`에 둔다.
- 본문 수치와 출력은 `practice/chapter{N}/results/*.log`에서 확인된 것만 사용한다.
- 결과가 환경에 따라 변동될 수 있으면 로그 생성 시각과 조건을 함께 남긴다.

### 참고문헌과 최신 정보

- 교재, 논문, 공식 문서를 인용할 때는 실제 출처를 확인한다.
- 최신 라이브러리 API, 모델명, 출시일, 가격, 정책은 웹 또는 공식 문서로 확인한다.
- 출처가 다른 도메인의 결과를 현재 맥락의 근거로 과장하지 않는다.

## 장별 권장 산출물

| 단계 | 산출물 | 위치 |
|---|---|---|
| 계획 | 장별 구성 계획 | `schema/chap{N}.md` |
| 리서치 | 교재/논문/공식문서 메모 | `content/research/` |
| 실습 | 실행 가능한 코드 | `practice/chapter{N}/code/` |
| 실행 증거 | 로그, evidence JSON | `practice/chapter{N}/results/` |
| 원고 | 강의자료 본문 | `docs/ch{N}.md` |
| 파생 | 수업 활동, 슬라이드 초안 | `lecture/` |

## 완료 기준

- 요청한 파일이 실제로 생성 또는 수정되었다.
- 실습 코드가 있으면 `scripts/run_and_capture.py`로 실행 증거를 남겼다.
- `./scripts/harness.sh`가 `HARNESS_PASS`를 출력했다.
- 실행하지 못한 검증이 있으면 이유를 명시한다.
