# GOTCHAS — 딥러닝 기초 강의자료 지뢰 목록

## 실행 결과

- 본문 수치와 출력은 `scripts/run_and_capture.py`로 생성한 `practice/chapterN/results/*.log`에서만 인용한다.
- 코드 수정 후에는 `python3 scripts/run_and_capture.py N --verify`로 결과가 낡지 않았는지 확인한다.

## 환경

- 루트 `.venv` 하나를 우선 사용한다.
- 공통 설치는 `pip install -r practice/requirements.txt`를 기준으로 한다.
- PyTorch/TensorFlow는 장별 필요에 따라 설치한다. 공통 설치에 강제하지 않는다.

## 딥러닝 실습

- GPU 전용 예제 금지. CPU fallback을 둔다.
- 수업용 기본값은 작게 둔다. 긴 학습은 선택 옵션으로 분리한다.
- seed를 고정하고 주요 설정을 로그에 출력한다.
- 합성 데이터는 합성 데이터라고 명시한다.

## 문서

- 기존 `docs/` 원고의 슬라이드 이미지 자리표시자(`!image.png` 등)는 실제 자산 없이 설명하지 않는다.
- 외부 모델/서비스의 최신 사양은 확인 없이 단정하지 않는다.
