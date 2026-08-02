# Gotchas

## General

- 하네스(`./scripts/harness.sh`)를 실행하지 않고 성공을 보고하지 않는다.
- 실습 결과는 `practice/chapter{N}/results/*.log`에서만 인용한다.
- 요청 범위 밖의 원고 전면 개편이나 파일명 변경은 하지 않는다.

## This Project

- `docs/ch{N}.md`는 현재 패딩 없는 번호를 사용한다. 새 자동화는 `chN`과 `chNN`을 모두 고려한다.
- 딥러닝 프레임워크는 환경별 설치 부담이 크므로 공통 requirements에 강제하지 않는다.
- 합성 데이터로 만든 결과는 "실측"이라고 쓰지 않는다.

