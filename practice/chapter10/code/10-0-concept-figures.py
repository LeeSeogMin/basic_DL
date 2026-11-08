"""10장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig10-1  문장이 토큰으로 쪼개지고 숫자 벡터가 되기까지의 흐름
  fig10-2  단어 가방 표의 모양(행이 문장, 열이 단어)
  fig10-3  자연어 처리의 발전 과정(연표)

이 그림들은 실험 결과가 아니라 개념 설명용 도식이다.
그림에 쓰인 문장은 모두 직접 만든 합성 문장이다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _krfont

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def box(ax, x, y, w, h, text, facecolor, fontsize=11, textcolor="white"):
    """모서리가 둥근 상자 하나를 그리고 가운데에 글자를 넣는다."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=facecolor,
            edgecolor="none",
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        linespacing=1.5,
    )


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color=color,
        )
    )


def blank_axes(ax, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def figure_text_to_vector(ko: bool, out_path: Path) -> None:
    """그림 10-1: 문장 한 줄이 숫자 벡터가 되기까지 네 단계."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    blank_axes(ax, xlim=(0, 10), ylim=(-0.9, 5.9))

    ax.set_title(
        L("문장 한 줄이 숫자 벡터가 되기까지",
          "How one sentence becomes a vector of numbers", ko),
        fontsize=13.5, pad=12, loc="left",
    )

    box(ax, 0.3, 4.5, 9.4, 1.0,
        L('1단계  원래 문장\n"배송이 정말 빨라요"',
          '1. raw sentence\n"the delivery was really fast"', ko),
        BLUE, 11.5)

    box(ax, 0.3, 3.0, 9.4, 1.0,
        L('2단계  토큰으로 쪼개기\n[ 배송이 ]  [ 정말 ]  [ 빨라요 ]',
          '2. split into tokens\n[ delivery ] [ really ] [ fast ]', ko),
        ORANGE, 11.5)

    box(ax, 0.3, 1.5, 9.4, 1.0,
        L("3단계  단어마다 번호를 붙인 사전\n배송이=0   정말=1   빨라요=2   느려요=3   화면이=4",
          "3. vocabulary with an index per word\ndelivery=0  really=1  fast=2  slow=3  screen=4", ko),
        GRAY, 11.5)

    box(ax, 0.3, 0.15, 9.4, 1.0,
        L("4단계  숫자 벡터\n[ 1, 1, 1, 0, 0 ]",
          "4. vector of numbers\n[ 1, 1, 1, 0, 0 ]", ko),
        GREEN, 11.5)

    for y_from, y_to in [(4.45, 4.05), (2.95, 2.55), (1.45, 1.20)]:
        arrow(ax, 5.0, y_from, 5.0, y_to)

    # 초록 상자 안에 초록 글씨를 쓰면 보이지 않는다. 상자 아래에 따로 적는다.
    ax.text(0.3, -0.45,
            L("이 마지막 줄 하나가 모델에 들어가는 입력이다.",
              "This last row is what the model receives as input.", ko),
            ha="left", va="center", fontsize=10.5, color=GREEN)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_bag_of_words(ko: bool, out_path: Path) -> None:
    """그림 10-2: 단어 가방 표의 생김새. 행이 문장, 열이 단어다."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 6.0))

    ax.set_title(
        L("단어 가방 표 — 행은 문장, 열은 단어, 칸은 나온 횟수",
          "Bag of words - rows are sentences, columns are words, cells are counts", ko),
        fontsize=13.5, pad=12, loc="left",
    )

    words = [
        L("배송이", "delivery", ko),
        L("빨라요", "fast", ko),
        L("느려요", "slow", ko),
        L("화면이", "screen", ko),
        L("선명해요", "sharp", ko),
    ]
    sentences = [
        L("배송이 정말 빨라요", "the delivery was fast", ko),
        L("배송이 너무 느려요", "the delivery was slow", ko),
        L("화면이 아주 선명해요", "the screen is sharp", ko),
    ]
    counts = [
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
    ]

    x0, y0 = 3.5, 1.2      # 표 본체의 왼쪽 아래 모서리
    cw, ch = 1.25, 1.05    # 칸 하나의 가로, 세로

    # 열 이름(단어)
    for j, word in enumerate(words):
        ax.text(x0 + cw * (j + 0.5), y0 + ch * 3 + 0.30, word,
                ha="center", va="bottom", fontsize=10.5, color="#333333",
                rotation=18)

    # 행 이름(문장)과 칸
    for i, sentence in enumerate(sentences):
        row_y = y0 + ch * (2 - i)
        ax.text(x0 - 0.25, row_y + ch / 2, sentence,
                ha="right", va="center", fontsize=10.5, color="#333333")
        for j in range(len(words)):
            value = counts[i][j]
            face = GREEN if value else "#EDEDED"
            text_color = "white" if value else "#999999"
            ax.add_patch(
                plt.Rectangle((x0 + cw * j, row_y), cw, ch,
                              facecolor=face, edgecolor="white", linewidth=2)
            )
            ax.text(x0 + cw * (j + 0.5), row_y + ch / 2, str(value),
                    ha="center", va="center", fontsize=15, color=text_color)

    ax.text(5.0, 0.35,
            L("칸의 대부분이 0이다. 단어 수가 늘수록 0의 비율이 커진다.\n"
              "이 표는 단어의 순서를 버린다. 같은 단어를 쓰면 순서가 달라도 같은 줄이 된다.",
              "Most cells are 0, and the share of zeros grows with the vocabulary.\n"
              "This table throws away word order: reordering a sentence gives the same row.", ko),
            ha="center", va="center", fontsize=10.5, color="#333333", linespacing=1.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_nlp_history(ko: bool, out_path: Path) -> None:
    """그림 10-3: 자연어 처리의 발전 과정(연표).

    연도와 사건은 본문 각주의 출처에서 확인한 것이다.
    가로 간격은 연도에 비례하지 않는다. 사건을 같은 간격으로 늘어놓았다.
    """
    L = _krfont.label
    events = [
        ("1954", L("조지타운-IBM\n기계번역 실험", "Georgetown-IBM", ko)),
        ("1966", L("ELIZA\n대화 프로그램", "ELIZA", ko)),
        (L("1990년대~", "1990s-", ko), L("통계 번역\n단어 가방·TF-IDF", "statistical MT", ko)),
        ("2013", L("word2vec\n단어 임베딩", "word2vec", ko)),
        ("2014", L("seq2seq\nRNN 번역", "seq2seq", ko)),
        ("2017", L("트랜스포머", "Transformer", ko)),
        ("2018", "BERT"),
        ("2020", "GPT-3"),
        ("2021", L("KLUE\n한국어 평가", "KLUE", ko)),
        ("2022", "ChatGPT"),
    ]
    eras = [
        (0, 1, BLUE, L("규칙 기반", "rule-based", ko),
         L("사람이 규칙과 사전을 적는다", "hand-written rules", ko)),
        (2, 2, ORANGE, L("통계 기반", "statistical", ko),
         L("단어 횟수를 센다", "count words", ko)),
        (3, 4, GREEN, L("신경망·임베딩", "neural", ko),
         L("단어를 실수 벡터로", "word vectors", ko)),
        (5, 9, RED, L("트랜스포머·대규모 언어모델", "Transformer / LLM", ko),
         L("큰 말뭉치로 먼저 배우고 일에 맞춘다", "pretrain, then adapt", ko)),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 3.6))
    ax.set_xlim(-0.6, len(events) - 0.4)
    ax.set_ylim(1.2, 5.0)
    ax.axis("off")

    for i0, i1, color, name, desc in eras:
        ax.add_patch(FancyBboxPatch((i0 - 0.45, 3.5), i1 - i0 + 0.9, 1.3,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=color, edgecolor="none", alpha=0.9))
        ax.text((i0 + i1) / 2, 4.38, name, ha="center", va="center",
                fontsize=11.5, color="white")
        ax.text((i0 + i1) / 2, 3.88, desc, ha="center", va="center",
                fontsize=9, color="white")

    ax.plot([-0.5, len(events) - 0.5], [3.0, 3.0], color=GRAY, linewidth=1.5)
    for i, (year, text) in enumerate(events):
        ax.plot([i], [3.0], "o", color="#333333", markersize=5)
        ax.text(i, 2.75, year, ha="center", va="top", fontsize=10.5, color="#333333")
        ax.text(i, 2.2, text, ha="center", va="top", fontsize=9, color="#333333",
                linespacing=1.4)

    ax.set_title(L("자연어 처리의 발전 과정 (가로 간격은 연도에 비례하지 않는다)",
                   "A short history of NLP (spacing is not to scale)", ko),
                 fontsize=13, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 10장 개념 그림 생성 ===")

    targets = [
        ("fig10-1-text-to-vector.png", figure_text_to_vector,
         "문장에서 숫자 벡터까지의 네 단계"),
        ("fig10-2-bag-of-words.png", figure_bag_of_words,
         "단어 가방 표의 생김새"),
        ("fig10-3-nlp-history.png", figure_nlp_history,
         "자연어 처리의 발전 과정"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")
    print("그림에 쓰인 문장은 직접 만든 합성 문장이다.")


if __name__ == "__main__":
    main()
