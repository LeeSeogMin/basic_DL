"""11장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig11-1  다음 단어 예측이 되풀이되어 문장 하나가 만들어지는 과정
  fig11-2  확률 분포가 뾰족할 때와 평평할 때 생성 결과가 어떻게 갈리는가
  fig11-3  환각이 생기는 자리

fig11-2의 막대 높이는 _tinylm.py의 합성 학습 문장에서 실제로 센 확률이다.
나머지 두 장은 개념 설명용 도식이다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _krfont
import _tinylm

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"

# fig11-2에서 확률 분포를 들여다볼 단어와, 비교할 온도 세 개
WATCH_WORD = "민준이는"
WATCH_TEMPERATURES = [0.3, 1.0, 2.0]


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


def arrow(ax, x1, y1, x2, y2, color=GRAY, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=16,
            linewidth=1.6,
            color=color,
        )
    )


def curved_arrow(ax, x1, y1, x2, y2, rad=0.35, color=ORANGE):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color=color,
        )
    )


def blank_axes(ax, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def figure_next_token_loop(ko: bool, out_path: Path) -> None:
    """그림 11-1: 다음 단어 하나를 고르는 일을 되풀이해 문장을 만든다."""
    L = _krfont.label
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0),
                             gridspec_kw={"height_ratios": [1.25, 1.0]})

    # 위쪽: 한 번의 예측 고리
    ax = axes[0]
    blank_axes(ax, xlim=(0, 10), ylim=(0, 6.4))
    ax.set_title(
        L("언어모델이 한 번에 하는 일 — 다음 단어 하나를 고른다",
          "What the model does once: pick one next word", ko),
        fontsize=13, pad=10, loc="left",
    )

    box(ax, 0.2, 3.4, 2.4, 1.6,
        L("지금까지의 글\n\"민준이는 월요일에\"",
          "Text so far\n\"Minjun on Monday\"", ko), BLUE, 10.5)
    box(ax, 3.1, 3.4, 2.5, 1.6,
        L("모델\n다음 단어 확률을 계산",
          "Model\ncomputes next-word\nprobabilities", ko), GRAY, 10.5)
    box(ax, 6.1, 3.4, 3.7, 1.6,
        L("확률표\n도서관에서 80%\n실습실에서 20%",
          "Probability table\nlibrary 80%\nlab 20%", ko), ORANGE, 10.5)
    arrow(ax, 2.65, 4.2, 3.05, 4.2)
    arrow(ax, 5.65, 4.2, 6.05, 4.2)

    box(ax, 6.1, 1.1, 3.7, 1.3,
        L("확률대로 한 단어를 뽑는다\n→ \"도서관에서\"",
          "Sample one word\n-> \"library\"", ko), GREEN, 10.5)
    arrow(ax, 7.95, 3.35, 7.95, 2.45)

    curved_arrow(ax, 6.05, 1.75, 1.40, 3.35, rad=-0.30)
    ax.text(3.72, 2.55,
            L("뽑은 단어를 글 뒤에 붙이고 처음으로 돌아간다",
              "Append the word and go back to the start", ko),
            ha="center", fontsize=11, color=ORANGE)

    # 아래쪽: 되풀이되어 문장이 되는 과정
    ax = axes[1]
    blank_axes(ax, xlim=(0, 10), ylim=(0, 4.2))
    ax.set_title(
        L("이 일을 다섯 번 되풀이하면 문장 하나가 나온다",
          "Repeat five times and a sentence appears", ko),
        fontsize=13, pad=10, loc="left",
    )

    steps = [
        L("민준이는", "Minjun", ko),
        L("월요일에", "on Monday", ko),
        L("도서관에서", "in the library", ko),
        L("통계학을", "statistics", ko),
        L("공부한다", "studies", ko),
    ]
    colors = [BLUE, GREEN, GREEN, GREEN, GREEN]
    x = 0.25
    for i, (word, color) in enumerate(zip(steps, colors)):
        box(ax, x, 1.8, 1.68, 1.0, word, color, 11)
        if i < len(steps) - 1:
            arrow(ax, x + 1.72, 2.3, x + 1.90, 2.3)
        x += 1.94

    ax.text(0.25, 1.25,
            L("사람이 정한 시작 단어", "given by the user", ko),
            ha="left", fontsize=10, color=BLUE)
    ax.text(9.75, 1.25,
            L("나머지 네 단어는 모델이 확률로 골랐다",
              "the other four were sampled by the model", ko),
            ha="right", fontsize=10, color=GREEN)

    ax.text(5.0, 0.35,
            L("모델은 문장 전체를 미리 계획하지 않는다. 매번 다음 한 단어만 고른다.",
              "The model never plans the whole sentence. It picks one next word at a time.", ko),
            ha="center", fontsize=10.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_sharp_vs_flat(ko: bool, out_path: Path) -> None:
    """그림 11-2: 온도를 바꾸면 확률 분포 모양과 생성 결과가 갈린다."""
    L = _krfont.label
    vocab, _counts, probs, _corpus = _tinylm.build_model()

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True)

    notes = [
        L("뾰족하다 — 늘 같은 답", "sharp - always the same", ko),
        L("원래 분포", "original", ko),
        L("평평하다 — 답이 흩어진다", "flat - answers scatter", ko),
    ]

    for ax, temperature, note in zip(axes, WATCH_TEMPERATURES, notes):
        pairs = _tinylm.next_word_table(vocab, probs, WATCH_WORD, temperature)
        labels = [word for word, _ in pairs]
        values = [p for _, p in pairs]
        if not ko:
            labels = [f"w{i + 1}" for i in range(len(labels))]

        bars = ax.bar(labels, values, color=BLUE, width=0.62)
        bars[0].set_color(ORANGE)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0%", "50%", "100%"], fontsize=10)
        ax.set_title(f"T = {temperature}\n{note}", fontsize=11.5, pad=8)
        ax.tick_params(axis="x", labelrotation=20, labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        for rect, value in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + 0.03,
                    f"{value * 100:.0f}%", ha="center", fontsize=9.5,
                    color="#333333")

    axes[0].set_ylabel(L("다음 단어로 뽑힐 확률", "probability of being picked", ko),
                       fontsize=10.5)

    fig.suptitle(
        L(f"\"{WATCH_WORD}\" 다음에 올 단어의 확률 — 온도를 바꾸면 모양이 달라진다",
          "Next-word probabilities after the first word - temperature reshapes them", ko),
        fontsize=13, y=0.99,
    )
    fig.text(0.5, 0.005,
             L("막대 높이는 합성 학습 문장에서 실제로 센 확률이다. 순서는 바뀌지 않고 격차만 바뀐다.",
               "Bar heights come from the synthetic training sentences. The order stays; only the gaps change.", ko),
             ha="center", fontsize=10, color="#333333")

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_where_hallucination(ko: bool, out_path: Path) -> None:
    """그림 11-3: 환각이 생기는 자리."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 7.4))

    ax.set_title(
        L("환각이 생기는 자리 — 모델에게 \"모른다\"라는 선택지가 없다",
          "Where hallucination comes from: \"I don't know\" is not on the list", ko),
        fontsize=13, pad=12, loc="left",
    )

    box(ax, 0.3, 5.2, 9.4, 1.0,
        L("사람의 질문: \"민준이는 화요일에 어디서 무엇을 하는가\"",
          "Question: \"Where does Minjun study on Tuesday?\"", ko), BLUE, 11.5)

    box(ax, 0.3, 3.1, 4.5, 1.6,
        L("모델이 만드는 것\n다음 단어 후보와 확률뿐",
          "What the model produces\nonly next-word candidates\nand their probabilities", ko),
        GRAY, 11)
    box(ax, 5.2, 3.1, 4.5, 1.6,
        L("표에 없는 것\n\"모른다\"  \"확실하지 않다\"\n\"출처가 없다\"",
          "Not on the list\n\"I don't know\"\n\"I am not sure\"", ko), RED, 11)
    arrow(ax, 2.55, 5.15, 2.55, 4.75)
    arrow(ax, 4.85, 3.90, 5.15, 3.90)

    box(ax, 0.3, 1.3, 4.5, 1.3,
        L("그래서 가장 그럴듯한\n다음 단어를 계속 고른다",
          "So it keeps picking the\nmost plausible next word", ko), ORANGE, 11)
    box(ax, 5.2, 1.3, 4.5, 1.3,
        L("결과: 어법은 맞고\n사실은 틀린 문장",
          "Result: fluent grammar,\nwrong fact", ko), RED, 11)
    arrow(ax, 2.55, 3.05, 2.55, 2.65)
    arrow(ax, 4.85, 1.95, 5.15, 1.95)

    ax.text(5.0, 0.45,
            L("모델은 사실을 확인하지 않는다. 학습 자료에서 자주 붙어 나온 단어를 이어 붙일 뿐이다.\n"
              "그래서 학습 자료에 한 번도 없던 조합이 아주 자연스러운 문장으로 나온다.",
              "The model never checks facts. It chains words that co-occurred often in training data,\n"
              "so combinations that never appeared can come out as fluent sentences.", ko),
            ha="center", fontsize=10.5, color="#333333", linespacing=1.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 11장 개념 그림 생성 ===")
    print("학습 문장은 코드 안에 넣은 합성 한국어 문장이다. 외부 API를 쓰지 않는다.")
    print()

    targets = [
        ("fig11-1-next-token-loop.png", figure_next_token_loop,
         "다음 단어 예측이 되풀이되어 문장이 되는 과정"),
        ("fig11-2-sharp-vs-flat.png", figure_sharp_vs_flat,
         "확률 분포가 뾰족할 때와 평평할 때"),
        ("fig11-3-where-hallucination.png", figure_where_hallucination,
         "환각이 생기는 자리"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("fig11-2의 막대 높이는 합성 학습 문장에서 센 실제 확률이다.")
    print("fig11-1과 fig11-3은 개념 설명용 도식이며, 실험 결과가 아니다.")


if __name__ == "__main__":
    main()
