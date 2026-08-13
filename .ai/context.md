# 프로젝트 컨텍스트 (2026-08-13 최종 갱신)

## 프로젝트
- 과목: 딥러닝 기초 (학부 2학년 대상, 15주)
- 목적: 강의자료, 실습 코드, 과제/프로젝트 자료를 일관된 구조로 구성
- 정본 문서: `contents.md`
- 저장소: git (main 브랜치, origin = github.com/LeeSeogMin/basic_DL)

## 수업 구조
- 각 주차 = **1교시 강의(90분) + 2교시 실습(90분)**
- 강의자료(`docs/ch{N}.md`)는 **Part A(강의) + Part B(실습)** 두 파트로 구성
- 거버넌스 반영: contents.md §4.0, 강의계획서.md §8, AGENTS.md §5, harness.md §6

## 완료 현황 (2026-08-13)

### 거버넌스·인프라
- 하네스 체계 (run_and_capture.py, harness.sh, verify.sh)
- Python venv + requirements.txt
- contents.md 15주 전체 설계
- 강의계획서.md 공식 강의계획서
- 두 교시 구조 거버넌스 문서 반영

### 장별 상태 — 15/15 완료
| 장 | docs/ | schema/ | practice/ | 상태 |
|---|---|---|---|---|
| 1 | Part A/B 재작성 | chap1.md | 코드 없음 (토론 중심) | **완료** |
| 2 | Part A/B 변환 | chap2.md | code+evidence | **완료** |
| 3 | 신규 작성 | chap3.md | code+evidence | **완료** |
| 4 | Part A/B 변환 | chap4.md | code+evidence | **완료** |
| 5 | 신규 작성 | chap5.md | code+evidence | **완료** |
| 6 | 신규 작성 | chap6.md | code+evidence | **완료** |
| 7 | 신규 작성 | chap7.md | code+evidence | **완료** |
| 8 | 신규 생성 | chap8.md | 코드 없음 (프로젝트) | **완료** |
| 9 | 신규 작성 | chap9.md | code+evidence | **완료** |
| 10 | 신규 작성 | chap10.md | code+evidence | **완료** |
| 11 | 신규 작성 | chap11.md | code+evidence | **완료** |
| 12 | 신규 작성 | chap12.md | code+evidence | **완료** |
| 13 | 신규 작성 | chap13.md | code+evidence | **완료** |
| 14 | 신규 작성 | chap14.md | 코드 없음 (프로젝트) | **완료** |
| 15 | lecture/final-presentation.md | — | — | **완료** |

### 수치 요약
- docs/ch*.md: 14개
- schema/chap*.md: 14개
- practice/chapter*/results/*.evidence.json: 11개 (코드 있는 장)
- lecture/final-presentation.md: 1개
- HARNESS_PASS: 통과

## 주요 제약
- 실행 결과는 `scripts/run_and_capture.py`로 생성한 로그에서만 인용
- 합성 데이터는 허용, 반드시 표시
- CPU에서 수업 시간 안에 실행 가능한 코드 우선
