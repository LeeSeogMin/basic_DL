"""1장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig1-1  인공지능 · 머신러닝 · 딥러닝의 포함 관계와 생성형 AI의 자리
  fig1-2  규칙을 사람이 쓰던 방식에서 데이터로 배우는 방식으로 넘어온 흐름
  fig1-3  지금 AI가 잘하는 일과 못하는 일을 가르는 두 가지 기준

이 그림들은 데이터 분석 결과가 아니라 개념 설명용 도식이다.
실험 결과나 측정값이 아니므로 수치를 넣지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

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
    ax.set_aspect("auto")
    ax.axis("off")


def figure_ai_ml_dl(ko: bool, out_path: Path) -> None:
    """그림 1-1: 세 개의 원이 겹겹이 들어 있는 포함 관계."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 6.7))
    ax.set_aspect("equal", adjustable="box")

    cx = 3.5

    # 바깥부터 안쪽으로: 인공지능 > 머신러닝 > 딥러닝
    ax.add_patch(Circle((cx, 3.0), 2.95, facecolor="#DCE6F0", edgecolor=BLUE, linewidth=2.0))
    ax.add_patch(Circle((cx, 2.55), 2.15, facecolor="#FBE0C6", edgecolor=ORANGE, linewidth=2.0))
    ax.add_patch(Circle((cx, 2.15), 1.30, facecolor="#D5EACC", edgecolor=GREEN, linewidth=2.0))

    ax.text(cx, 5.45, L("인공지능", "Artificial Intelligence", ko),
            ha="center", fontsize=14, color=BLUE)
    ax.text(cx, 4.25, L("머신러닝", "Machine Learning", ko),
            ha="center", fontsize=13, color="#B4610F")
    ax.text(cx, 2.85, L("딥러닝", "Deep Learning", ko),
            ha="center", va="center", fontsize=12.5, color="#2E6B27")

    # 생성형 AI: 딥러닝 안쪽을 크게 덮고 머신러닝 쪽으로 조금 넘어간다.
    ax.add_patch(
        Ellipse((cx, 1.70), 3.0, 1.25, angle=0,
                facecolor="none", edgecolor=RED, linewidth=2.0, linestyle=(0, (5, 3)))
    )
    ax.text(cx, 1.70, L("생성형 AI", "Generative AI", ko),
            ha="center", va="center", fontsize=12.5, color=RED)

    # 오른쪽 설명 줄
    lines = [
        (5.62, BLUE, L("인공지능 — 사람이 하던 판단을 기계가 대신하는 일 전부.\n"
                       "규칙을 사람이 직접 써 넣은 프로그램도 여기에 들어간다.",
                       "AI - any machine that takes over a human judgement,\n"
                       "including programs whose rules a person wrote by hand.", ko)),
        (4.05, "#B4610F", L("머신러닝 — 규칙을 사람이 쓰지 않고,\n"
                            "예시 데이터를 보고 모델이 판단 기준을 찾는다.",
                            "Machine learning - the model finds the rule\n"
                            "from example data instead of a person writing it.", ko)),
        (2.55, "#2E6B27", L("딥러닝 — 머신러닝 중에서 층을 여러 겹 쌓은\n"
                            "신경망을 쓰는 갈래다.",
                            "Deep learning - the branch of ML that stacks\n"
                            "many layers of a neural network.", ko)),
        (1.05, RED, L("생성형 AI — 문장·그림·소리를 만들어 내는 쪽이다.\n"
                      "지금 쓰는 것은 대부분 딥러닝으로 만든다.",
                      "Generative AI - systems that produce text, images, sound.\n"
                      "Today they are built with deep learning.", ko)),
    ]
    for y, color, text in lines:
        ax.text(6.75, y, text, ha="left", va="center", fontsize=10.3,
                color=color, linespacing=1.6)

    ax.set_title(
        L("바깥 원이 안쪽 원을 품는다", "Each circle contains the next", ko),
        fontsize=13, pad=10, loc="left",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_good_and_bad(ko: bool, out_path: Path) -> None:
    """그림 1-3: 두 가지 기준으로 나눈 네 칸."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 6.6))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 8))

    ax.set_title(
        L("두 가지를 물으면 AI에게 맡길 일인지 갈린다",
          "Two questions decide whether a task suits today's AI", ko),
        fontsize=13, pad=12, loc="left",
    )

    cells = [
        # (x, y, 색, 본문)
        (2.35, 4.35, GREEN,
         L("지금 AI가 잘한다\n\n사진 속 글자 읽기\n스팸 메일 거르기\n음성을 받아쓰기",
           "Today's AI does well\n\nreading text in photos\nfiltering spam\nspeech to text", ko)),
        (5.95, 4.35, ORANGE,
         L("초안까지만 맡긴다\n\n글 초안 쓰기\n그림 만들기\n아이디어 나열하기",
           "Draft only\n\ndrafting text\ngenerating images\nlisting ideas", ko)),
        (2.35, 0.85, BLUE,
         L("아직 어렵다\n\n드문 고장 진단하기\n처음 보는 상황 판단하기",
           "Still hard\n\ndiagnosing rare faults\njudging unseen situations", ko)),
        (5.95, 0.85, RED,
         L("사람이 정한다\n\n징계 결정하기\n진로 정하기\n책임이 따르는 결정",
           "A person decides\n\ndisciplinary decisions\ncareer choices\nany accountable call", ko)),
    ]
    for x, y, color, text in cells:
        box(ax, x, y, 3.4, 2.9, text, color, 10.6)

    # 가로축 이름표: 정답이 하나로 정해지는가
    ax.text(4.05, 7.55, L("정답이 하나로 정해진다",
                          "the answer is one fixed thing", ko),
            ha="center", fontsize=11.5, color="#333333")
    ax.text(7.65, 7.55, L("정답이 여러 개다",
                          "many answers are acceptable", ko),
            ha="center", fontsize=11.5, color="#333333")

    # 세로축 이름표: 비슷한 예시가 데이터에 많이 있는가
    ax.text(2.20, 5.80, L("비슷한 예시가\n데이터에 많다",
                          "many similar\nexamples exist", ko),
            ha="right", va="center", fontsize=11.5, color="#333333", linespacing=1.5)
    ax.text(2.20, 2.30, L("비슷한 예시가\n거의 없다",
                          "almost no\nsimilar examples", ko),
            ha="right", va="center", fontsize=11.5, color="#333333", linespacing=1.5)

    ax.text(5.05, 0.25,
            L("초록 칸이 지금 AI를 쓰는 자리다. 빨간 칸은 정확도 문제가 아니라 책임 문제다.",
              "The green cell is where today's AI is used. The red cell is about accountability, not accuracy.",
              ko),
            ha="center", fontsize=10.4, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_rule_to_data(ko: bool, out_path: Path) -> None:
    """그림 1-2: 규칙을 사람이 쓰던 방식에서 데이터로 배우는 방식으로."""
    L = _krfont.label
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 6.6))

    # 위쪽 — 사람이 규칙을 쓰던 방식
    ax = axes[0]
    blank_axes(ax)
    ax.set_title(
        L("먼저 온 방식 — 사람이 규칙을 문장으로 써 넣는다",
          "The earlier way: a person writes the rules as sentences", ko),
        fontsize=13, pad=10, loc="left",
    )
    box(ax, 0.2, 2.2, 2.4, 1.9,
        L("사람이 규칙을 쓴다\n\"60점 이상이면 합격\"",
          "A person writes the rule\n\"score >= 60 -> pass\"", ko), BLUE, 10.6)
    box(ax, 3.3, 2.2, 2.6, 1.9,
        L("프로그램이 규칙을\n그대로 적용한다",
          "The program applies\nthe rule as written", ko), GRAY, 10.6)
    box(ax, 6.6, 2.2, 3.2, 1.9,
        L("규칙을 말로 쓸 수 있는\n문제까지만 풀린다",
          "Only problems whose rule\ncan be written in words", ko), GRAY, 10.6)
    arrow(ax, 2.66, 3.15, 3.24, 3.15)
    arrow(ax, 5.96, 3.15, 6.54, 3.15)
    ax.text(5.0, 1.3,
            L("고양이 사진과 강아지 사진을 가르는 규칙은 문장으로 쓰다가 막힌다",
              "Writing a rule that separates cat photos from dog photos breaks down", ko),
            ha="center", fontsize=10.5, color=RED)

    # 아래쪽 — 데이터에서 배우는 방식
    ax = axes[1]
    blank_axes(ax)
    ax.set_title(
        L("지금 방식 — 예시를 모아 주면 모델이 기준을 찾는다",
          "The current way: collect examples and the model finds the rule", ko),
        fontsize=13, pad=10, loc="left",
    )
    box(ax, 0.2, 2.2, 2.4, 1.9,
        L("사람이 예시를 모은다\n(사진 + 정답)",
          "A person collects examples\n(photo + answer)", ko), BLUE, 10.6)
    box(ax, 3.3, 2.2, 2.6, 1.9,
        L("모델이 예시를 보고\n판단 기준을 찾는다",
          "The model looks at them\nand finds the rule", ko), ORANGE, 10.6)
    box(ax, 6.6, 2.2, 3.2, 1.9,
        L("말로 못 쓰던 문제도\n풀리기 시작한다",
          "Problems no one could\nwrite down start to work", ko), GREEN, 10.6)
    arrow(ax, 2.66, 3.15, 3.24, 3.15)
    arrow(ax, 5.96, 3.15, 6.54, 3.15)
    ax.text(5.0, 1.3,
            L("바뀐 것은 규칙의 유무가 아니라 규칙을 쓰는 주체다. 사람에서 모델로 넘어갔다",
              "What changed is not whether rules exist, but who writes them: the model, not the person.", ko),
            ha="center", fontsize=10.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 1장 개념 그림 생성 ===")

    targets = [
        ("fig1-1-ai-ml-dl.png", figure_ai_ml_dl,
         "인공지능·머신러닝·딥러닝 포함 관계와 생성형 AI의 자리"),
        ("fig1-2-rule-to-data.png", figure_rule_to_data,
         "규칙을 쓰던 방식에서 데이터로 배우는 방식으로"),
        ("fig1-3-good-and-bad.png", figure_good_and_bad,
         "AI가 잘하는 일과 못하는 일을 가르는 두 기준"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")
    print("1주차 실습 두 개는 브라우저에서 하므로 실습용 파이썬 코드가 없다.")


if __name__ == "__main__":
    main()
