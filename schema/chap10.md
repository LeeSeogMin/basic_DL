# 10주차 계획서 — 자연어 처리: 컴퓨터가 글을 다루는 방식 (2026-11-09 전면 개정)

## 1. 장의 역할

사용자 지시(2026-11-09)로 이론을 「자연어 처리의 전반과 발전 과정」으로 바꾸고, 실습을 Kiwi 설치·형태소 분석과 뉴스 제목 토픽 분류로 바꿨다. 이전의 단어 가방 유사도·이웃 단어 임베딩·감성 분류 실습은 뺐다(코드 `10-1-text-to-numbers.py`, `10-2-sentiment.py` 삭제).

## 2. 절과 실습의 배치

| 순서 | 제목 | 핵심 내용 | 그림·표 |
|---|---|---|---|
| 1 | 자연어 처리란 | 정의, 글자 번호에는 뜻이 없다, 자연어가 어려운 이유 | 그림 10.1 `fig10-1-text-to-vector.png` |
| 2 | 자연어 처리가 하는 일 | 작업 8가지 | 표 10.1 (Hugging Face Tasks) |
| 3 | 발전 과정 | 규칙(조지타운-IBM 1954, ELIZA 1966) → 통계 → 신경망(word2vec 2013, seq2seq 2014) → 트랜스포머(2017)·BERT(2018)·GPT-3(2020)·ChatGPT(2022-11), 한국어(KLUE 2021) | 그림 10.2 `fig10-3-nlp-history.png`, 표 10.2 |
| 4 | 한국어 형태소 분석 | 형태소, 형태소 분석기, Kiwi | — |
| 실습 10.1 | Kiwi 설치와 형태소 분석 | 공백 vs Kiwi, 사전 14 → 9 | `kiwi_compare.csv` |
| 5 | 문장을 숫자로 바꿔 분류하기 | 단어 가방·TF-IDF, 주제별 가중치, 예측 | 그림 10.3 `fig10-2-bag-of-words.png` |
| 실습 10.2 | 뉴스 제목 주제 분류 | KLUE YNAT, 공백 0.605 vs Kiwi 0.741, 기준선 0.097(훈련·시험 비율 차이) | 그림 10.4 `topic_confusion.png` |
| 6 | 주의할 점 | 틀림, 시기 변화, 개인정보, 데이터 라이선스 | — |

## 3. 실습

- `practice/chapter10/code/10-1-kiwi-morphemes.py` — kiwipiepy 0.24.1, 조작 지점 `MY_SENTENCES`, `SAME_WORD_SENTENCES`
- `practice/chapter10/code/10-2-topic-classification.py` — YNAT parquet을 Hugging Face에서 내려받아 `results/`에 저장(약 5MB), 훈련 10,000·시험 2,000(seed 42), TF-IDF(min_df=2) + LogisticRegression, 조작 지점 `MY_HEADLINES`, `TRAIN_SIZE`, `TEST_SIZE`
- 추가 패키지: kiwipiepy, pyarrow (공통 requirements에는 넣지 않고 본문 실행 환경에 설치 명령을 적었다)

## 4. 출처 확인 기록 (2026-11-09)

| 항목 | 확인 방법 |
|---|---|
| YNAT 크기·라벨·라이선스 | datasets-server info API(45,678/9,107, 7개 라벨), 데이터 카드(cc-by-sa-4.0), parquet의 date 열(2016~2019) |
| YNAT 출처·합의 라벨 | KLUE 논문 PDF(arXiv:2105.09680v4) "Yonhap News Agency", "keep 70,000 headlines ... majority consensus" |
| 조지타운-IBM, ELIZA | Wikipedia 문서 열람 |
| word2vec·seq2seq·Transformer·BERT·GPT-3·KLUE | arXiv API로 제목·첫 저자·게재일 확인 |
| ChatGPT 2022년 11월 | OpenAI 페이지는 403으로 열리지 않았다. 여러 보도에서 2022-11-30 공개로 일치 |
| Kiwi | GitHub README(Apache-2.0, pip install kiwipiepy) |

## 5. 제출물

1. `results/topic_confusion.png`
2. `results/my_headlines.csv`
