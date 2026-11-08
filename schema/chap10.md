# 10주차 계획서 — 텍스트와 임베딩: 문장을 숫자로 바꾸기

## 1. 장의 역할

10주차는 9주차와 짝을 이룬다. 2026-11-09에 9장과 같은 이론·실습 혼합 배치로 개정했다(§3). 9주차에서 "이미지는 숫자다"를 픽셀값 64개로 확인했다면, 10주차에서는 "문장도 숫자다"를 단어 가방 표로 확인한다. 문장 한 줄이 벡터 한 줄이 되는 과정을 학생이 눈으로 따라가게 만드는 것이 이 장의 목표다.

말로 넘어가지 않는다. 토큰으로 쪼갠 결과를 그대로 찍고, 단어 가방 표를 찍고, 0인 칸이 몇 %인지 센다. TF-IDF로 잰 유사도가 0이 나오는 짝을 찾아, 뜻이 같아도 단어가 다르면 0이 된다는 것을 숫자로 본다. 임베딩이 왜 필요한지는 이 0에서 나온다.

임베딩 모델을 학습시키지 않는다. 내려받지도 않는다. `TruncatedSVD`로 TF-IDF 벡터를 2차원까지 줄여 "차원을 줄여 뜻을 요약한다"는 생각만 손으로 확인한다. 진짜 임베딩과 무엇이 다른지는 본문에 명시한다.

`torch`, `tensorflow`, `transformers`를 쓰지 않는다. 형태소 분석기도 설치하지 않는다. 조사와 어미가 단어에 붙은 채로 남는 것을 그대로 두고, 그래서 무슨 일이 생기는지 실습 10.2의 "모르는 단어" 목록에서 확인한다.

대상은 학부 2학년이다. 1차시 이론 강의, 2차시 실습으로 나눠 진행한다.

## 2. 학습 목표

1. 컴퓨터가 글자를 직접 다루지 못하는 이유를 설명한다.
2. 문장을 토큰으로 쪼개고, 한국어에서 이 작업이 왜 까다로운지 말한다.
3. 단어 가방 표의 행과 열을 읽고, 두 문장의 코사인 유사도를 손으로 계산한다.
4. 뜻이 같은 두 문장의 유사도가 0이 되는 경우를 실험 결과로 설명한다.
5. 이웃 단어를 세어 만든 단어 벡터가 무엇을 해결하고 무엇을 해결하지 못하는지 설명한다.
6. 단어 가중치를 더해 긍정 확률을 손으로 계산하고, 학습 문장이 치우치면 무엇이 달라지는지 확인한다.
7. 자연어 처리의 대표 작업을 말한다.

## 3. 절과 실습의 배치 (2026-11-09 개정)

9장과 같이 이론 절 바로 뒤에 그 개념을 확인하는 실습을 둔다. 문제를 먼저 측정하고 해법을 뒤에 둔다. 각 실습 앞 절에서 예측을 적고, 실습 결과 읽기에서 예측을 판정한다.

| 순서 | 제목 | 핵심 내용 | 그림 |
|---|---|---|---|
| 1 | 컴퓨터는 글자를 모른다 | 글자 번호는 뜻과 무관하다 | `fig10-1-text-to-vector.png` |
| 2 | 토큰과 사전 | 토큰 규칙 두 줄, 사전 = 벡터 길이, 한국어 조사 문제 | — |
| 3 | 단어 가방과 코사인 유사도 | 표 읽기, 코사인 유사도 손계산(문장2·3 = 0.50), TF-IDF 칸 값, 예측 1(뜻이 같아도 0)·예측 2(뜻이 달라도 0보다 큼)·문제 3(0의 비율) | `fig10-2-bag-of-words.png`, `fig10-4-cosine.png` |
| 실습 10.1 | 문장을 숫자로 바꾸고 유사도의 한계를 측정한다 | 1~4단계, 예측 판정 | `sentence_map.png` |
| 4 | 임베딩 — 뜻은 이웃 단어에서 온다 | 저차원·조밀, 이웃 표 → TruncatedSVD → 단어 벡터, 실습 10.1 5단계 결과를 이 절에서 읽는다(9장 4절이 9-2 로그를 인용하는 방식), 이웃만 같으면 뜻이 반대여도 가까워진다, 이웃을 공유하게 문장을 고른 것은 사람이다 | `fig10-3-sentence-space.png`, `word_vectors.png` |
| 5 | 텍스트 분류와 감성 분석 | z = 절편 + 가중치 합, 시그모이드 손계산, 예측 1(모르는 단어)·예측 2(부정어)·예측 3(한쪽 라벨 단어) | — |
| 실습 10.2 | 리뷰 분류기와 헷갈리는 문장 | 정확도·혼동행렬, 헷갈리는 문장 8개, z 손계산, 단어 가중치 | `sentiment_confusion.png`, `confusing_sentences.png`, `word_weights.png` |
| 6 | 학습 문장이 치우치면 생기는 일 | "정말" +0.382, 알아채는 방법, 생성형 AI 학습 문장 | — |
| 7 | 자연어 처리가 하는 일 | 작업 표(Hugging Face Tasks 분류에서 고름), 헷갈리는 작업 구분, 주의점 | 표 10.3 |

기존 7절 「직접 움직여 보는 웹 자료」는 없앴다. 링크 4건은 4절·6절·7절의 「더 보기」로 옮겼다.

본문에 「책임 있는 AI 체크」 절을 두지 않는다. 주의점은 6절과 7절에 넣었다.

## 4. 실습 설계

실습은 두 개다. 각각 노트북에서 몇 초 안에 끝난다. GPU가 필요 없다.

두 실습의 문장은 모두 수업용으로 직접 쓴 **합성 한국어 문장**이다. 실제 리뷰를 긁어오지 않는다. 인터넷에서 데이터나 모델을 내려받지 않는다.

### 실습 10.1 — 문장을 숫자로 바꾸고 유사도의 한계를 측정한다

- 파일: `practice/chapter10/code/10-1-text-to-numbers.py`
- 데이터: 합성 한국어 문장 8개(`MY_SENTENCES`), 5단계용 합성 문장 22개(`CONTEXT_SENTENCES`)
- 학생 조작 지점: `MY_SENTENCES`, `MEANING_PAIRS`, `HAND_PAIR`, `CONTEXT_SENTENCES`, `WORD_PAIRS`, `SENTENCE_PAIRS`, `EMBED_DIM`
- 하는 일 다섯
  1. 문장을 토큰으로 쪼개 그대로 찍는다
  2. 단어 가방 표를 만들고 0인 칸의 비율을 센다
  3. 코사인 유사도를 손으로 한 번 풀어 찍고, TF-IDF 칸 값과 유사도 표를 찍는다
  4. `TruncatedSVD`로 2차원으로 줄여 평면에 점으로 찍는다
  5. 단어 x 이웃 단어 표를 만들어 `TruncatedSVD`로 3칸 단어 벡터를 만들고, 문장 벡터(단어 벡터 평균)의 유사도를 TF-IDF 유사도와 비교한다
- 산출물: `results/sentence_map.png`, `results/word_vectors.png`, `results/bag_of_words.csv`, `results/sentence_similarity.csv`

### 실습 10.2 — 리뷰 분류기와 헷갈리는 문장

- 파일: `practice/chapter10/code/10-2-sentiment.py`
- 데이터: 합성 리뷰 64개(긍정 32 + 부정 32)
- 모델: `CountVectorizer(binary=True)` + `LogisticRegression`. 기준선은 `DummyClassifier`
- 학생 조작 지점: `MY_TRICKY_SENTENCES`, `TEST_RATIO`, `NEAR_BAND`
- 하는 일: 훈련 44·테스트 20 분할, 정확도·혼동행렬, 헷갈리는 문장 8개와 아는 단어·모르는 단어, 첫 문장의 z = 절편 + 가중치 합과 시그모이드 손계산(`predict_proba`와 대조), 0.5 근처 문장, 단어 가중치
- 산출물: `results/sentiment_confusion.png`, `results/confusing_sentences.png`, `results/word_weights.png`, `results/sentiment_results.csv`, `results/confusing_sentences.csv`, `results/word_weights.csv`

### 개념 그림 생성

- 파일: `practice/chapter10/code/10-0-concept-figures.py`
- 산출물: `results/fig10-1-text-to-vector.png`, `results/fig10-2-bag-of-words.png`, `results/fig10-3-sentence-space.png`, `results/fig10-4-cosine.png`
- 한글 폰트 도우미: `practice/chapter10/code/_krfont.py` (폰트가 없으면 영어 라벨로 자동 전환)
- 이 네 장의 문장과 좌표는 설명을 위해 손으로 정한 값이다. 실험 결과가 아니라는 것을 코드와 로그에 적는다

## 5. 실행 증거

```bash
python scripts/run_and_capture.py 10
python scripts/run_and_capture.py 10 --verify
```

본문의 정확도, 표본 수, 유사도, 가중치, 확률, 혼동행렬 값은 `practice/chapter10/results/*.log`에서만 인용한다.

## 6. 외부 시각 자료(열어서 확인함)

| 자료 | 확인한 내용 |
|---|---|
| Google ML 단기집중과정(한국어), 임베딩 | 음식 5,000종 추천 예시로 원-핫 인코딩의 한계를 설명하고, 임베딩을 "희소 데이터의 저차원 표현"으로 정의한다. 그림 1·2에 음식 항목의 원-핫 벡터가 그려져 있다. 최종 수정 2025-05-15, 지원 중단 공지 없음 |
| Google ML 단기집중과정(한국어), 어휘와 원-핫 인코딩 | 카테고리 N개를 요소 N개짜리 벡터로 바꾸는 과정을 그림 세 장으로 나눠 보여준다. 희소 표현을 "1.0의 위치만 저장하는 방식"으로 설명한다 |
| Google 텍스트 분류 가이드(한국어) | 스팸 필터링 그림으로 주제 분류를 소개하고, 감정 분석을 "콘텐츠가 표현하는 의견의 유형을 파악"으로 정의한다. 데이터 수집부터 배포까지 7단계 워크플로우 그림이 있다. 토큰화·TF-IDF의 상세 설명은 이 페이지에 없다 |
| scikit-learn, Classification of text documents using sparse features | 20 newsgroups 문서 2,034개를 TF-IDF로 바꿔 분류한다. 혼동행렬 그림 2장과 클래스별 상위 단어 가중치 그림 2장이 있다. 메일 주소 메타데이터에서 나온 "caltech"가 거짓 특성으로 잡히는 예를 그대로 보여준다 |
| Hugging Face Tasks | Natural Language Processing 분류에 작업 12개(Feature Extraction, Fill-Mask, Question Answering, Sentence Similarity, Summarization, Table Question Answering, Text Classification, Text Generation, Text Ranking, Token Classification, Translation, Zero-Shot Classification)가 있다. 작업별 설명 문장은 목록 페이지에 없다. 2026-11-09 확인 |

빼기로 한 것

- TensorFlow Embedding Projector(`https://projector.tensorflow.org/`) — 본문이 자바스크립트로 그려져 WebFetch로 내용을 확인하지 못했다
- MLU-Explain, Logistic Regression(`https://mlu-explain.github.io/logistic-regression/`) — 제목 외에 본문이 읽히지 않았다
- 위키독스 『딥 러닝을 이용한 자연어 처리 입문』 Word2Vec 페이지(`https://wikidocs.net/22660`) — WebFetch는 403, curl은 Cloudflare 확인 페이지만 받아 내용을 확인하지 못했다(2026-11-09)

## 7. 제출물

수업 중 실습 실시 여부만 확인한다. 보고서를 쓰지 않는다.

1. `results/sentence_map.png`
2. `results/confusing_sentences.png`
