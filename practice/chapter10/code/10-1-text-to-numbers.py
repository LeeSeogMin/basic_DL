"""실습 10.1 - 문장이 숫자로 바뀌는 과정을 단계별로 본다.

네 단계로 진행한다.
  1단계  문장을 토큰으로 쪼갠다
  2단계  단어 가방(bag of words) 표를 만들어 그대로 출력한다
  3단계  TF-IDF로 바꾼 뒤 문장 사이 코사인 유사도를 잰다
  4단계  TruncatedSVD로 2차원으로 줄여 평면에 점으로 찍는다

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
    """3단계: TF-IDF로 바꾼 뒤 문장 사이 코사인 유사도를 잰다."""
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
    tfidf = vectorizer.fit_transform(MY_SENTENCES)
    similarity = cosine_similarity(tfidf)

    names = [f"문장{i}" for i in range(1, len(MY_SENTENCES) + 1)]
    table = pd.DataFrame(similarity.round(2), index=names, columns=names)

    print("=== 3단계. TF-IDF와 코사인 유사도 ===")
    print("TF-IDF는 흔한 단어의 무게를 낮추고 드문 단어의 무게를 높인다.")
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
        print(f"  -> 컴퓨터가 잰 유사도: {score:.2f}")
        print()

    table.to_csv(RESULTS_DIR / "sentence_similarity.csv", encoding="utf-8-sig")
    print("유사도 표는 results/sentence_similarity.csv 에 저장했다.")
    print()
    return tfidf


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
    tfidf = step3_similarity()
    step4_map(tfidf, ko)

    print("=== 정리 ===")
    print("단어 가방은 단어가 겹칠 때만 두 문장을 닮았다고 본다.")
    print("뜻이 같아도 단어가 다르면 유사도가 0이 된다. 이 한계가 임베딩이 필요한 이유다.")


if __name__ == "__main__":
    main()
