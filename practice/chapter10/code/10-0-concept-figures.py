"""10장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig10-1  문장이 토큰으로 쪼개지고 숫자 벡터가 되기까지의 흐름
  fig10-2  단어 가방 표의 모양(행이 문장, 열이 단어)
  fig10-3  비슷한 뜻의 문장이 가까이 모이는 평면

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


def figure_sentence_space(ko: bool, out_path: Path) -> None:
    """그림 10-3: 뜻이 비슷한 문장이 가까이 놓이는 평면(개념 도식)."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    # 좌표는 개념 설명을 위해 손으로 정한 값이다. 실험 결과가 아니다.
    groups = [
        (BLUE, L("배송 이야기", "about delivery", ko), [
            (1.5, 4.6, L("배송이 빨라요", "delivery was fast", ko)),
            (2.1, 4.1, L("하루 만에 왔어요", "it arrived in a day", ko)),
            (1.2, 3.9, L("택배가 금방 도착", "the parcel came quickly", ko)),
        ]),
        (ORANGE, L("화면 이야기", "about the screen", ko), [
            (4.6, 1.7, L("화면이 선명해요", "the screen is sharp", ko)),
            (5.2, 1.2, L("액정이 또렷해요", "the display is crisp", ko)),
            (4.2, 1.1, L("화질이 좋아요", "great picture quality", ko)),
        ]),
        (GREEN, L("가격 이야기", "about the price", ko), [
            (5.0, 4.5, L("가격이 비싸요", "it is expensive", ko)),
            (5.6, 4.0, L("값이 너무 높아요", "the price is too high", ko)),
        ]),
    ]

    for color, group_name, points in groups:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=190, color=color, edgecolor="white",
                   linewidth=1.5, zorder=3, label=group_name)
        for x, y, text in points:
            ax.text(x, y - 0.30, text, ha="center", va="top",
                    fontsize=10, color="#333333", zorder=4)

    # 화살표가 설명 글자를 가로지르지 않도록, 글자는 화살표 아래쪽 빈자리에 둔다.
    ax.annotate(
        "", xy=(4.4, 2.0), xytext=(2.6, 3.3),
        arrowprops=dict(arrowstyle="<->", color=RED, linewidth=1.6,
                        linestyle=(0, (4, 3))),
    )
    ax.text(2.3, 2.4,
            L("뜻이 다르면\n멀리 떨어진다", "different meaning\n-> far apart", ko),
            ha="left", va="center", fontsize=10.5, color=RED, linespacing=1.5)

    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 5.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(L("가로축(뜻을 요약한 첫 번째 축)",
                    "axis 1 (a summary direction of meaning)", ko), fontsize=10.5)
    ax.set_ylabel(L("세로축(두 번째 축)", "axis 2", ko), fontsize=10.5)
    ax.set_title(
        L("뜻이 비슷한 문장은 가까이 모인다 (개념 도식, 좌표는 손으로 정한 값)",
          "Sentences with similar meaning sit close together (schematic, hand-placed)", ko),
        fontsize=13, pad=12, loc="left",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

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
        ("fig10-3-sentence-space.png", figure_sentence_space,
         "뜻이 비슷한 문장이 가까이 모이는 평면"),
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
