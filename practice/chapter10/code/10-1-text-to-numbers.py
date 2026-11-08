"""실습 10.1 - 문장이 숫자로 바뀌는 과정을 단계별로 본다.

다섯 단계로 진행한다.
  1단계  문장을 토큰으로 쪼갠다
  2단계  단어 가방(bag of words) 표를 만들어 그대로 출력한다
  3단계  코사인 유사도를 손으로 한 번 계산하고, TF-IDF로 바꾼 뒤 문장 사이 유사도를 계산한다
  4단계  TruncatedSVD로 2차원으로 줄여 평면에 점으로 찍는다
  5단계  같은 문장에 함께 나온 이웃 단어를 세어 단어 벡터를 만들고,
         단어가 안 겹치던 문장 짝의 유사도가 어떻게 바뀌는지 본다

주의: 여기 쓰는 문장은 실제 리뷰가 아니라 수업용으로 직접 쓴 합성 문장이다.
인터넷에서 데이터나 모델을 내려받지 않는다. scikit-learn만 쓴다.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import _krfont

# ---------------------------------------------------------------------------
# 학생이 바꾸는 곳: 아래 문장을 자기 문장으로 바꾼다.
# 3~5글자짜리 짧은 문장 8개쯤이 그림에서 보기 좋다.
# 일부러 두 가지를 섞어 뒀다.
#   - 같은 단어를 공유하는 문장 (1번과 2번, 4번과 5번)
#   - 뜻은 같은데 단어가 하나도 안 겹치는 문장 (1번과 3번, 4번과 6번)
# ---------------------------------------------------------------------------
MY_SENTENCES = [
    "배송이 정말 빨라요",
    "배송이 하루 만에 왔어요",
    "택배가 하루 만에 도착했어요",
    "화면이 아주 선명해요",
    "화면이 밝고 선명해요",
    "액정이 밝고 또렷해요",
    "배터리가 하루 만에 닳아요",
    "가격이 너무 비싸요",
]

# 위 문장 중 "사람이 보기에 뜻이 거의 같은 짝"을 번호로 적는다.
# 컴퓨터가 이 짝을 얼마나 닮았다고 보는지 3단계에서 따로 출력한다.
# MY_SENTENCES를 바꾸면 이 번호도 같이 바꾼다.
MEANING_PAIRS = [(1, 3), (4, 6)]

# 3단계에서 코사인 유사도를 손으로 풀어 볼 문장 짝(번호)
HAND_PAIR = (2, 3)

# ---------------------------------------------------------------------------
# 학생이 바꾸는 곳 2: 5단계에서 단어 벡터를 만들 때 쓰는 문장.
# 단어의 뜻을 "같은 문장에 함께 나온 이웃 단어"로 나타낸다.
# "배송이"와 "택배가"는 한 문장에 같이 나오지 않게 썼다. 대신 "하루", "도착했어요",
# "빨라요" 같은 이웃을 나눠 갖게 썼다. 이웃을 나눠 갖도록 문장을 고른 것은 사람이다.
# 실제 임베딩은 이런 문장을 수억 개 모아 같은 계산을 한다.
# ---------------------------------------------------------------------------
CONTEXT_SENTENCES = [
    "배송이 정말 빨라요",
    "배송이 하루 만에 왔어요",
    "배송이 늦게 도착했어요",
    "배송이 빨라서 좋아요",
    "배송이 이틀 걸렸어요",
    "택배가 하루 만에 도착했어요",
    "택배가 정말 빨라요",
    "택배가 늦게 왔어요",
    "택배가 문 앞에 도착했어요",
    "화면이 밝고 선명해요",
    "화면이 아주 선명해요",
    "화면이 어두워서 아쉬워요",
    "화면이 크고 또렷해요",
    "액정이 밝고 또렷해요",
    "액정이 크고 선명해요",
    "액정이 어두워서 불편해요",
    "배터리가 하루 만에 닳아요",
    "배터리가 오래 가요",
    "배터리가 빨리 닳아요",
    "배터리가 금방 닳아요",
    "가격이 너무 비싸요",
    "가격이 저렴해서 좋아요",
]

# 5단계에서 비교할 단어 짝
WORD_PAIRS = [("배송이", "택배가"), ("화면이", "액정이"), ("배송이", "화면이")]

# 5단계에서 비교할 MY_SENTENCES의 문장 짝(번호)과 그 짝을 고른 이유
SENTENCE_PAIRS = [
    ((1, 3), "뜻이 같다"),
    ((4, 6), "뜻이 같다"),
    ((1, 4), "뜻이 다르다"),
    ((3, 7), "뜻이 다르다"),
]

EMBED_DIM = 3           # 단어 벡터의 길이. 이웃 단어 수(수십 개)를 이 길이로 줄인다.

RANDOM_SEED = 42

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def tokenize(text: str) -> list[str]:
    """문장을 토큰으로 쪼갠다.

    규칙은 두 줄이 전부다.
      1. 문장부호를 공백으로 바꾼다
      2. 공백으로 자른다

    한국어에서 이 규칙은 조사와 어미를 단어에 붙인 채로 남긴다.
    그래서 "배송이"와 "배송은"이 서로 다른 단어로 잡힌다.
    형태소 분석기를 쓰면 이 문제를 줄일 수 있지만, 이번 실습에서는 설치하지 않는다.
    붙어 있는 그대로 두고 무슨 일이 생기는지 확인하는 것이 이 실습의 목적이다.
    """
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return [token for token in cleaned.split() if token]


def step1_tokens() -> None:
    """1단계: 문장을 토큰으로 쪼개 그대로 보여준다."""
    print("=== 1단계. 문장을 토큰으로 쪼갠다 ===")
    for i, sentence in enumerate(MY_SENTENCES, start=1):
        tokens = tokenize(sentence)
        print(f"문장{i}  {sentence}")
        print(f"       -> {len(tokens)}개 토큰: {' | '.join(tokens)}")
    print()


def step2_bag_of_words() -> pd.DataFrame:
    """2단계: 단어 가방 표를 만들고 출력한다."""
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
    matrix = vectorizer.fit_transform(MY_SENTENCES)
    vocab = vectorizer.get_feature_names_out()

    table = pd.DataFrame(
        matrix.toarray(),
        index=[f"문장{i}" for i in range(1, len(MY_SENTENCES) + 1)],
        columns=vocab,
    )

    total_cells = table.size
    zero_cells = int((table.to_numpy() == 0).sum())

    print("=== 2단계. 단어 가방 표 ===")
    print(f"문장 {len(MY_SENTENCES)}개에서 서로 다른 단어 {len(vocab)}개가 나왔다.")
    print(f"표의 크기: {table.shape[0]}행 x {table.shape[1]}열 = 칸 {total_cells}개")
    print(f"그중 0인 칸: {zero_cells}개 ({zero_cells / total_cells * 100:.1f}%)")
    print()

    shared = [word for word in vocab if int(table[word].sum()) >= 2]
    print(f"두 문장 이상에 나온 단어는 {len(shared)}개다. 그 부분만 떼어 보면 이렇다.")
    print(table[shared].to_string())
    print()
    print("나머지 단어는 한 문장에만 나온다. 그 열은 1이 하나뿐이고 나머지가 전부 0이다.")
    print("전체 표는 results/bag_of_words.csv 에 저장했다.")
    print()

    table.to_csv(RESULTS_DIR / "bag_of_words.csv", encoding="utf-8-sig")
    return table


def step3_similarity() -> np.ndarray:
    """3단계: 코사인 유사도를 손으로 한 번 풀고, TF-IDF 유사도 표를 만든다."""
    print("=== 3단계. 코사인 유사도를 손으로 한 번 계산한다 ===")
    a, b = HAND_PAIR
    tokens_a = sorted(set(tokenize(MY_SENTENCES[a - 1])))
    tokens_b = sorted(set(tokenize(MY_SENTENCES[b - 1])))
    shared = [t for t in tokens_a if t in tokens_b]
    hand = len(shared) / np.sqrt(len(tokens_a) * len(tokens_b))
    print(f"  문장{a}: {MY_SENTENCES[a - 1]}   -> 1이 찍힌 칸 {len(tokens_a)}개")
    print(f"  문장{b}: {MY_SENTENCES[b - 1]}   -> 1이 찍힌 칸 {len(tokens_b)}개")
    print(f"  두 문장에 모두 1이 찍힌 칸(겹친 단어): {len(shared)}개 "
          f"({', '.join(shared) if shared else '없음'})")
    print("  코사인 유사도 = 같은 칸끼리 곱해 더한 값 / (각 줄의 길이를 곱한 값)")
    print(f"              = {len(shared)} / (√{len(tokens_a)} x √{len(tokens_b)})"
          f" = {len(shared)} / {np.sqrt(len(tokens_a) * len(tokens_b)):.2f}"
          f" = {hand:.2f}")
    print("  0과 1만 있는 줄이면 '겹친 단어 수 / √(단어 수 x 단어 수)'와 같다.")
    print()

    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
    tfidf = vectorizer.fit_transform(MY_SENTENCES)
    similarity = cosine_similarity(tfidf)
    vocab = list(vectorizer.get_feature_names_out())

    names = [f"문장{i}" for i in range(1, len(MY_SENTENCES) + 1)]
    table = pd.DataFrame(similarity.round(2), index=names, columns=names)

    print("=== 3단계. TF-IDF와 코사인 유사도 ===")
    print("TF-IDF는 흔한 단어의 무게를 낮추고 드문 단어의 무게를 높인다.")
    print(f"문장{a}의 TF-IDF 칸 값(0이 아닌 칸만, 줄의 길이를 1로 맞춘 값)")
    row = tfidf[a - 1].toarray()[0]
    for idx in np.argsort(-row):
        if row[idx] == 0:
            break
        count = sum(vocab[idx] in tokenize(s) for s in MY_SENTENCES)
        print(f"  {vocab[idx]:<6} {row[idx]:.2f}   (이 단어가 나온 문장 {count}개)")
    print(f"단어 가방으로 계산한 문장{a}-문장{b} 유사도는 {hand:.2f}, "
          f"TF-IDF로 계산하면 {similarity[a - 1, b - 1]:.2f}로 바뀐다.")
    print()
    print("코사인 유사도는 0에 가까울수록 안 닮았고, 1에 가까울수록 닮았다.")
    print()
    print(table.to_string())
    print()

    pairs = []
    for i in range(len(MY_SENTENCES)):
        for j in range(i + 1, len(MY_SENTENCES)):
            pairs.append((similarity[i, j], i, j))
    pairs.sort(reverse=True)

    print("가장 닮은 문장 짝 3개")
    for rank, (score, i, j) in enumerate(pairs[:3], start=1):
        print(f"  {rank}위  유사도 {score:.2f}")
        print(f"        문장{i + 1}: {MY_SENTENCES[i]}")
        print(f"        문장{j + 1}: {MY_SENTENCES[j]}")
    print()

    zero_pairs = [(score, i, j) for score, i, j in pairs if score == 0.0]
    print(f"전체 {len(pairs)}개 짝 가운데 유사도가 정확히 0인 짝이 {len(zero_pairs)}개다.")
    print("단어가 하나도 안 겹친다는 뜻이다.")
    print()

    print("사람이 보기에 뜻이 같은 짝을 컴퓨터는 어떻게 보는가")
    for a, b in MEANING_PAIRS:
        if not (1 <= a <= len(MY_SENTENCES) and 1 <= b <= len(MY_SENTENCES)):
            print(f"  문장{a}-문장{b}는 문장 번호 범위를 벗어났다. MEANING_PAIRS를 고칠 것.")
            continue
        score = similarity[a - 1, b - 1]
        print(f"  문장{a}: {MY_SENTENCES[a - 1]}")
        print(f"  문장{b}: {MY_SENTENCES[b - 1]}")
        print(f"  -> 컴퓨터가 계산한 유사도: {score:.2f}")
        print()

    table.to_csv(RESULTS_DIR / "sentence_similarity.csv", encoding="utf-8-sig")
    print("유사도 표는 results/sentence_similarity.csv 에 저장했다.")
    print()
    return tfidf, similarity


def step4_map(tfidf, ko: bool) -> None:
    """4단계: TruncatedSVD로 2차원으로 줄여 평면에 찍는다."""
    L = _krfont.label

    svd = TruncatedSVD(n_components=2, random_state=RANDOM_SEED)
    coords = svd.fit_transform(tfidf)
    explained = svd.explained_variance_ratio_

    print("=== 4단계. 2차원으로 줄여 평면에 찍기 ===")
    print(f"TF-IDF 벡터의 원래 차원: {tfidf.shape[1]}차원")
    print("TruncatedSVD로 2차원까지 줄인다.")
    print(f"두 축이 담아낸 정보의 비율: "
          f"가로축 {explained[0] * 100:.1f}%, 세로축 {explained[1] * 100:.1f}%, "
          f"합계 {explained.sum() * 100:.1f}%")
    print()
    print("각 문장의 2차원 좌표")
    coord_table = pd.DataFrame(
        coords.round(3),
        index=[f"문장{i}" for i in range(1, len(MY_SENTENCES) + 1)],
        columns=["가로축", "세로축"],
    )
    coord_table["문장"] = MY_SENTENCES
    print(coord_table.to_string())
    print()

    # 좌표가 소수점 셋째 자리까지 같은 문장은 그림에서 한 점에 겹친다.
    # 겹친 문장은 번호를 묶어 한 점 안에 적고, 로그에도 남긴다.
    groups: dict[tuple[float, float], list[int]] = {}
    for i in range(len(MY_SENTENCES)):
        key = (round(float(coords[i, 0]), 3), round(float(coords[i, 1]), 3))
        groups.setdefault(key, []).append(i + 1)

    overlapped = {key: members for key, members in groups.items() if len(members) > 1}
    if overlapped:
        print("2차원으로 줄인 뒤 완전히 같은 자리에 놓인 문장")
        for key, members in overlapped.items():
            names = ", ".join(f"문장{n}" for n in members)
            print(f"  {names}  ->  좌표 ({key[0]:.3f}, {key[1]:.3f})")
        print("남은 두 축에는 이 문장들을 갈라 놓을 정보가 없다는 뜻이다.")
        print("그림에서는 한 점 안에 번호를 묶어 적는다.")
        print()

    # 점 위에 문장을 그대로 쓰면 글자가 서로 겹친다.
    # 그래서 점 안에는 번호만 넣고, 문장 목록은 오른쪽에 따로 적는다.
    fig, (ax, ax_list) = plt.subplots(
        1, 2, figsize=(12.5, 5.8), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )

    ax.axhline(0, color="#E5E5E5", linewidth=1, zorder=1)
    ax.axvline(0, color="#E5E5E5", linewidth=1, zorder=1)
    for (x, y), members in groups.items():
        ax.scatter([x], [y], s=340, color=BLUE,
                   edgecolor="white", linewidth=1.8, zorder=3)
        ax.text(x, y, ",".join(str(n) for n in members),
                ha="center", va="center",
                fontsize=11.5 if len(members) == 1 else 9.5,
                color="white", zorder=4)

    ax.set_xlabel(L(f"가로축 (정보의 {explained[0] * 100:.0f}%)",
                    f"axis 1 ({explained[0] * 100:.0f}% of info)", ko), fontsize=11)
    ax.set_ylabel(L(f"세로축 (정보의 {explained[1] * 100:.0f}%)",
                    f"axis 2 ({explained[1] * 100:.0f}% of info)", ko), fontsize=11)
    ax.set_title(
        L("TF-IDF 문장 벡터를 2차원으로 줄인 지도",
          "Sentence vectors reduced to 2D", ko),
        fontsize=12.5, pad=10, loc="left",
    )
    ax.margins(0.32)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # 오른쪽 칸: 번호와 문장 목록
    ax_list.axis("off")
    ax_list.set_title(L("문장 목록", "sentence list", ko),
                      fontsize=12.5, pad=10, loc="left")
    if ko:
        lines = [f"{i + 1}.  {s}" for i, s in enumerate(MY_SENTENCES)]
    else:
        lines = [f"{i + 1}.  (see the run log)" for i in range(len(MY_SENTENCES))]
    for row, line in enumerate(lines):
        ax_list.text(0.02, 0.93 - row * 0.115, line, transform=ax_list.transAxes,
                     ha="left", va="top", fontsize=11, color="#333333")

    fig.suptitle(
        L("같은 단어를 쓴 문장끼리 가까이 모인다 (직접 만든 합성 문장)",
          "Sentences sharing words end up close together (synthetic sentences)", ko),
        fontsize=13.5, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = RESULTS_DIR / "sentence_map.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"문장 지도 그림: results/{out_path.name}")
    print()


def step5_word_vectors(tfidf_similarity: np.ndarray, ko: bool) -> None:
    """5단계: 이웃 단어를 세어 단어 벡터를 만들고, 문장 유사도를 다시 계산한다."""
    L = _krfont.label

    print("=== 5단계. 이웃 단어로 단어 벡터 만들기 ===")
    print(f"단어 벡터용 문장(CONTEXT_SENTENCES): {len(CONTEXT_SENTENCES)}개, 직접 쓴 합성 문장")

    counter = CountVectorizer(tokenizer=tokenize, token_pattern=None,
                              lowercase=False, binary=True)
    x = counter.fit_transform(CONTEXT_SENTENCES)
    words = list(counter.get_feature_names_out())
    index = {w: i for i, w in enumerate(words)}

    # (단어 x 단어) 표: 두 단어가 같은 문장에 함께 나온 문장 수
    together = (x.T @ x).toarray().astype(float)
    np.fill_diagonal(together, 0)
    print(f"서로 다른 단어 {len(words)}개 -> 이웃 표 {len(words)}행 x {len(words)}열")
    print("단어 하나 = 이웃 표의 한 줄 = 그 단어와 같은 문장에 나온 단어들의 횟수")
    print()

    print("--- 이웃 표에서 두 단어의 줄을 비교한다 ---")
    for w1, w2 in WORD_PAIRS:
        if w1 not in index or w2 not in index:
            print(f"  {w1}-{w2}: CONTEXT_SENTENCES에 없는 단어가 있다. WORD_PAIRS를 고칠 것.")
            continue
        r1, r2 = together[index[w1]], together[index[w2]]
        n1 = [words[j] for j in np.nonzero(r1)[0]]
        n2 = [words[j] for j in np.nonzero(r2)[0]]
        common = [w for w in n1 if w in n2]
        sim = cosine_similarity([r1], [r2])[0, 0]
        print(f"  {w1} - {w2}")
        print(f"    같은 문장에 함께 나온 횟수: {int(together[index[w1], index[w2]])}회")
        print(f"    {w1}의 이웃 {len(n1)}개, {w2}의 이웃 {len(n2)}개, 공유한 이웃 {len(common)}개"
              f" ({', '.join(common) if common else '없음'})")
        print(f"    두 줄의 코사인 유사도: {sim:.2f}")
    print()

    svd = TruncatedSVD(n_components=EMBED_DIM, random_state=RANDOM_SEED)
    vectors = svd.fit_transform(together)
    print(f"--- 이웃 표의 한 줄({len(words)}칸)을 TruncatedSVD로 {EMBED_DIM}칸으로 줄인다 ---")
    for w in ("배송이", "택배가", "화면이"):
        if w in index:
            nums = ", ".join(f"{v:+.2f}" for v in vectors[index[w]])
            print(f"  {w:<5} -> [{nums}]")
    for w1, w2 in WORD_PAIRS:
        if w1 in index and w2 in index:
            sim = cosine_similarity(vectors[[index[w1]]], vectors[[index[w2]]])[0, 0]
            print(f"  {w1} - {w2} 단어 벡터 유사도: {sim:.2f}")
    print()

    def sentence_vector(text: str) -> tuple[np.ndarray | None, list[str]]:
        known = [t for t in tokenize(text) if t in index]
        if not known:
            return None, known
        return np.mean([vectors[index[t]] for t in known], axis=0), known

    print("--- 문장 벡터 = 문장에 든 단어 벡터의 평균 ---")
    print(f"{'짝':<10}{'사람 판단':<10}{'TF-IDF':>8}{'단어 벡터':>10}")
    rows = []
    for (a, b), why in SENTENCE_PAIRS:
        if not (1 <= a <= len(MY_SENTENCES) and 1 <= b <= len(MY_SENTENCES)):
            print(f"  문장{a}-문장{b}는 문장 번호 범위를 벗어났다. SENTENCE_PAIRS를 고칠 것.")
            continue
        va, _ = sentence_vector(MY_SENTENCES[a - 1])
        vb, _ = sentence_vector(MY_SENTENCES[b - 1])
        if va is None or vb is None:
            print(f"  문장{a}-문장{b}: CONTEXT_SENTENCES에 아는 단어가 없는 문장이 있다.")
            continue
        emb = cosine_similarity([va], [vb])[0, 0]
        tf = tfidf_similarity[a - 1, b - 1]
        rows.append((f"{a}-{b}", why, tf, emb))
        print(f"문장{a}-문장{b:<4}{why:<10}{tf:>8.2f}{emb:>10.2f}")
    for a in sorted({n for (pair, _) in SENTENCE_PAIRS for n in pair}):
        if 1 <= a <= len(MY_SENTENCES):
            _, known = sentence_vector(MY_SENTENCES[a - 1])
            missing = [t for t in tokenize(MY_SENTENCES[a - 1]) if t not in index]
            note = f", 단어 벡터가 없는 단어: {', '.join(missing)}" if missing else ""
            print(f"  문장{a} {MY_SENTENCES[a - 1]} -> 평균에 쓴 단어 {len(known)}개{note}")
    print()
    print("주의: 이웃을 공유하도록 문장을 고른 것은 사람이다. 이 결과는 계산 방식을 보여줄 뿐,")
    print("컴퓨터가 스스로 뜻을 알아낸 증거가 아니다.")
    print()

    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r[2] for r in rows], height=0.38, color=GRAY,
            label=L("TF-IDF (단어가 겹쳐야 닮음)", "TF-IDF (needs shared words)", ko))
    ax.barh(y + 0.2, [r[3] for r in rows], height=0.38, color=BLUE,
            label=L("이웃 단어 벡터 (이웃이 겹쳐도 닮음)", "neighbor word vectors", ko))
    for i, r in enumerate(rows):
        ax.text(max(r[2], 0) + 0.02, i - 0.2, f"{r[2]:.2f}", va="center", fontsize=9.5)
        ax.text(max(r[3], 0) + 0.02, i + 0.2, f"{r[3]:.2f}", va="center", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels([L(f"문장{r[0]} ({r[1]})", f"pair {r[0]}", ko) for r in rows],
                       fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.15)
    ax.set_xlabel(L("코사인 유사도", "cosine similarity", ko), fontsize=10.5)
    ax.set_title(L("같은 문장 짝을 두 방법으로 비교 (직접 만든 합성 문장)",
                   "Same sentence pairs, two methods (synthetic sentences)", ko),
                 fontsize=12, loc="left", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              fontsize=9.5, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = RESULTS_DIR / "word_vectors.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"비교 그림: results/{out_path.name}")
    print()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    print("=== 실습 10.1 문장을 숫자로 바꾸기 ===")
    _krfont.report(ko)
    print("데이터: 수업용으로 직접 쓴 합성 문장(실제 리뷰 아님)")
    print(f"문장 수: {len(MY_SENTENCES)}개")
    print()

    step1_tokens()
    step2_bag_of_words()
    tfidf, tfidf_similarity = step3_similarity()
    step4_map(tfidf, ko)
    step5_word_vectors(tfidf_similarity, ko)

    print("=== 정리 ===")
    print("단어 가방은 단어가 겹칠 때만 두 문장을 닮았다고 본다.")
    print("뜻이 같아도 단어가 다르면 유사도가 0이 된다. 이 한계가 임베딩이 필요한 이유다.")
    print("이웃 단어로 만든 단어 벡터는 겹치는 단어가 없어도 이웃이 같으면 가깝게 놓는다.")
    print("대신 이웃을 공유하기만 하면 뜻이 반대인 문장도 가깝게 놓는다.")


if __name__ == "__main__":
    main()
