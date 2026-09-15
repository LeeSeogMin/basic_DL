# Gotchas — 반복해서 밟는 함정

이 저장소의 함정 목록은 이 파일 하나다. 2026-09-16에 `.claude/GOTCHAS.md`를 여기로 합쳤다.
한 번 푼 문제의 기록은 `.ai/lessons-learned.md`에 남긴다.

## 실행 결과

- 하네스(`./scripts/harness.sh`)를 돌리지 않고 성공을 보고하지 않는다.
- 본문 수치와 출력은 `scripts/run_and_capture.py`로 만든 `practice/chapter{N}/results/*.log`에서만 인용한다.
- 코드를 고친 뒤에는 `python3 scripts/run_and_capture.py {N} --verify`로 결과가 낡았는지 확인한다.
- 본문을 고쳤으면 `python3 scripts/md2pdf.py docs/ch{N}.md`로 PDF를 다시 만든다. 안 만들면 게이트 G3가 잡는다.

## 환경

- 루트 `.venv` 하나를 우선 쓴다.
- 공통 설치는 `pip install -r practice/requirements.txt`를 기준으로 한다.
- PyTorch/TensorFlow는 장별 필요에 따라 설치한다. 공통 설치에 강제하지 않는다. 환경별 설치 부담이 크다.

## 딥러닝 실습

- GPU 전용 예제를 쓰지 않는다. CPU fallback을 둔다.
- 수업용 기본값은 작게 둔다. 긴 학습은 선택 옵션으로 분리한다.
- 난수 seed를 고정하고 주요 설정을 로그에 출력한다.
- 합성 데이터는 코드 주석과 본문 양쪽에 합성 데이터라고 밝힌다. "실측"이라고 쓰지 않는다.

## 문서

- `docs/` 원고의 슬라이드 이미지 자리표시자(`!image.png` 등)는 실제 파일 없이 설명하지 않는다.
- 외부 모델·서비스의 최신 사양은 확인 없이 단정하지 않는다.
- `docs/ch{N}.md`는 0을 채우지 않은 번호를 쓴다. 새 자동화는 `chN`과 `chNN`을 모두 고려한다.

## 작업 범위

- 요청 범위 밖의 원고 전면 개편이나 파일명 변경은 하지 않는다. 발견한 것은 보고한다.
