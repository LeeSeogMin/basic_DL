"""실습 10.2 - 뉴스 제목으로 주제(토픽)를 분류한다.

데이터: KLUE의 YNAT(연합뉴스 제목 주제 분류). 제목 하나에 주제 하나가 붙어 있다.
주제는 IT과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치 일곱 가지다.
라이선스: CC BY-SA 4.0 (https://huggingface.co/datasets/klue/klue)

다섯 단계로 진행한다.
  1단계  데이터를 내려받고(처음 한 번), 일부를 뽑아 훈련용과 시험용으로 나눈다
  2단계  제목을 공백으로 자른 경우와 Kiwi로 쪼갠 경우, 두 가지 사전을 만든다
  3단계  TF-IDF + 로지스틱 회귀로 학습하고 기준선·두 방법의 정확도를 비교한다
  4단계  주제마다 판단을 좌우하는 단어와 혼동행렬을 본다
  5단계  내가 쓴 제목의 주제를 맞혀 본다

필요한 패키지: kiwipiepy, pyarrow (python -m pip install kiwipiepy pyarrow)
처음 실행할 때 인터넷에서 약 5MB(parquet 파일 두 개)를 내려받아 results/ 폴더에 저장한다.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import _krfont

# ---------------------------------------------------------------------------
# 학생이 바꾸는 곳 1: 주제를 맞혀 볼 내 뉴스 제목
# ---------------------------------------------------------------------------
MY_HEADLINES = [
    "정부, 내년 예산안 국회에 제출",
    "새 스마트폰 배터리 수명 두 배로 늘린 기술 공개",
    "류현진, 7이닝 무실점 호투로 시즌 첫 승",
    "주가 이틀째 하락…환율은 올라",
    "올여름 바닷가 피서객 작년보다 늘어",
]

# ---------------------------------------------------------------------------
# 학생이 바꾸는 곳 2: 실행 설정
# ---------------------------------------------------------------------------
TRAIN_SIZE = 10000      # 훈련용 제목 수 (전체 45,678개 중 무작위로 뽑는다)
TEST_SIZE = 2000        # 시험용 제목 수 (검증 세트 9,107개 중 무작위로 뽑는다)
TOP_WORDS = 5           # 주제마다 보여 줄 단어 수
RANDOM_SEED = 42

KEEP_TAGS = ("NNG", "NNP", "VV", "VA", "XR", "SL", "SH")
TOPICS = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
URLS = {
    "train": "https://huggingface.co/api/datasets/klue/klue/parquet/ynat/train/0.parquet",
    "validation": "https://huggingface.co/api/datasets/klue/klue/parquet/ynat/validation/0.parquet",
}

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


def load_split(name: str) -> pd.DataFrame:
    """YNAT 한 묶음을 읽는다. 없으면 내려받아 results/에 저장한다."""
    path = RESULTS_DIR / f"ynat_{name}.parquet"
    if not path.exists():
        print(f"내려받는 중: {URLS[name]}")
        try:
            urllib.request.urlretrieve(URLS[name], path)
        except OSError as error:
            print(f"내려받기에 실패했다: {error}")
            print("인터넷 연결을 확인하거나, 위 주소의 파일을 직접 받아 아래 경로에 둔다.")
            print(f"    {path}")
            sys.exit(1)
    return pd.read_parquet(path)


def split_by_space(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return [token for token in cleaned.split() if token]


def make_kiwi_tokenizer():
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        print("kiwipiepy가 없다. python -m pip install kiwipiepy 로 설치한 뒤 다시 실행한다.")
        sys.exit(1)
    kiwi = Kiwi()

    def kiwi_content(texts: list[str]) -> list[list[str]]:
        return [[t.form for t in tokens if t.tag.split("-")[0] in KEEP_TAGS]
                for tokens in kiwi.tokenize(texts)]

    return kiwi_content


def fit_and_score(train_tokens, y_train, test_tokens, y_test):
    vectorizer = TfidfVectorizer(analyzer=lambda tokens: tokens, min_df=2)
    x_train = vectorizer.fit_transform(train_tokens)
    x_test = vectorizer.transform(test_tokens)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return vectorizer, model, pred, accuracy_score(y_test, pred)


def draw_confusion(matrix: np.ndarray, ko: bool, out_path: Path) -> None:
    L = _krfont.label
    names = TOPICS if ko else ["IT/Sci", "Economy", "Society", "Culture", "World",
                               "Sports", "Politics"]
    rate = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if rate[i, j] > 0.5 else "#333333")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10, rotation=30)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(L("모델이 고른 주제", "predicted topic", ko), fontsize=11)
    ax.set_ylabel(L("정답 주제", "true topic", ko), fontsize=11)
    ax.set_title(L("뉴스 제목 주제 분류 혼동행렬 (Kiwi 형태소, 시험용 제목)",
                   "Topic classification confusion matrix (Kiwi)", ko),
                 fontsize=12, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    print("=== 실습 10.2 뉴스 제목 주제 분류 ===")
    _krfont.report(ko)
    print()

    # 1단계 ---------------------------------------------------------------
    print("=== 1단계. 데이터 ===")
    train_all = load_split("train")
    valid_all = load_split("validation")
    print(f"KLUE YNAT 훈련 세트 {len(train_all):,}개, 검증 세트 {len(valid_all):,}개")
    train = train_all.sample(n=min(TRAIN_SIZE, len(train_all)), random_state=RANDOM_SEED)
    test = valid_all.sample(n=min(TEST_SIZE, len(valid_all)), random_state=RANDOM_SEED)
    print(f"이 실습에서 쓰는 양: 훈련 {len(train):,}개, 시험 {len(test):,}개 (seed {RANDOM_SEED})")
    print()
    counts = pd.DataFrame({
        "훈련": train["label"].value_counts().sort_index().to_numpy(),
        "시험": test["label"].value_counts().sort_index().to_numpy(),
    }, index=TOPICS)
    print("주제별 제목 수")
    print(counts.to_string())
    print()
    print("훈련용 제목 예시")
    for _, row in train.head(5).iterrows():
        print(f"  [{TOPICS[row['label']]}] {row['title']}")
    print()

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    # 2단계 ---------------------------------------------------------------
    print("=== 2단계. 두 가지 방법으로 제목을 쪼갠다 ===")
    kiwi_content = make_kiwi_tokenizer()
    space_train = [split_by_space(t) for t in train["title"]]
    space_test = [split_by_space(t) for t in test["title"]]
    kiwi_train = kiwi_content(list(train["title"]))
    kiwi_test = kiwi_content(list(test["title"]))
    example = train["title"].iloc[0]
    print(f"제목: {example}")
    print(f"  공백으로 자르기: {' | '.join(space_train[0])}")
    print(f"  Kiwi(뜻을 담은 형태소): {' | '.join(kiwi_train[0])}")
    print()

    # 3단계 ---------------------------------------------------------------
    print("=== 3단계. 학습과 정확도 ===")
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    base_acc = accuracy_score(y_test, baseline.predict(np.zeros((len(y_test), 1))))
    sv, _, _, space_acc = fit_and_score(space_train, y_train, space_test, y_test)
    kv, model, pred, kiwi_acc = fit_and_score(kiwi_train, y_train, kiwi_test, y_test)
    print("두 번 이상 나온 단어만 사전에 넣는다(min_df=2).")
    print(f"{'방법':<26}{'사전 크기':>10}{'시험 정확도':>12}")
    print(f"{'기준선(가장 많은 주제로만 답하기)':<20}{'-':>10}{base_acc:>12.3f}")
    print(f"{'공백으로 자르기':<22}{len(sv.vocabulary_):>10,}{space_acc:>12.3f}")
    print(f"{'Kiwi 형태소':<24}{len(kv.vocabulary_):>10,}{kiwi_acc:>12.3f}")
    print()

    # 4단계 ---------------------------------------------------------------
    print("=== 4단계. 주제마다 판단을 좌우하는 단어 (Kiwi 모델) ===")
    vocab = np.array(kv.get_feature_names_out())
    for k, topic in enumerate(TOPICS):
        top = vocab[np.argsort(model.coef_[k])[::-1][:TOP_WORDS]]
        print(f"  {topic:<6} {', '.join(top)}")
    print()

    matrix = confusion_matrix(y_test, pred, labels=range(len(TOPICS)))
    print("주제별 시험 정확도 (Kiwi 모델)")
    for k, topic in enumerate(TOPICS):
        print(f"  {topic:<6} {matrix[k, k] / matrix[k].sum():.3f}  ({matrix[k, k]}/{matrix[k].sum()})")
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(np.argmax(off), off.shape)
    print(f"가장 많이 헷갈린 짝: 정답 {TOPICS[i]} -> 예측 {TOPICS[j]} {off[i, j]}건")
    print()
    wrong = test[pred != y_test].head(6)
    print("틀린 제목 예시")
    for idx, row in wrong.iterrows():
        guess = pred[test.index.get_loc(idx)]
        print(f"  정답 {TOPICS[row['label']]:<5} 예측 {TOPICS[guess]:<5} {row['title']}")
    print()
    draw_confusion(matrix, ko, RESULTS_DIR / "topic_confusion.png")
    print("혼동행렬 그림: results/topic_confusion.png")
    print()

    # 5단계 ---------------------------------------------------------------
    print("=== 5단계. 내가 쓴 제목의 주제 ===")
    my_x = kv.transform(kiwi_content(MY_HEADLINES))
    probs = model.predict_proba(my_x)
    rows = []
    for text, p in zip(MY_HEADLINES, probs):
        order = np.argsort(p)[::-1]
        print(f"  {text}")
        print(f"    -> {TOPICS[order[0]]} {p[order[0]]:.2f}, 다음 {TOPICS[order[1]]} {p[order[1]]:.2f}")
        rows.append({"제목": text, "1순위": TOPICS[order[0]], "확률": round(float(p[order[0]]), 2),
                     "2순위": TOPICS[order[1]], "2순위 확률": round(float(p[order[1]]), 2)})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "my_headlines.csv", index=False, encoding="utf-8-sig")
    print("결과 표: results/my_headlines.csv")
    print()

    print("=== 정리 ===")
    print("주제 분류는 4장의 분류와 같은 틀이다. 제목을 단어로 쪼개 TF-IDF로 바꾸고 주제별 가중치를 배운다.")
    print("형태소로 쪼개면 조사만 다른 단어가 하나로 모여 사전이 작아진다.")


if __name__ == "__main__":
    main()
