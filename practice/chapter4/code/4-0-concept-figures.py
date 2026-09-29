"""4장 이론 강의용 개념 그림을 만든다.

만드는 그림은 두 장이다.
  fig4-1  규칙 기반 방식과 머신러닝 방식의 흐름 비교
  fig4-2  훈련 데이터와 테스트 데이터를 나누는 이유

정밀도·재현율의 우선순위는 그림이 아니라 본문 표 4.6·4.7로 설명한다.

이 그림들은 데이터 분석 결과가 아니라 개념 설명용 도식이다.
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


def figure_rule_vs_learning(ko: bool, out_path: Path) -> None:
    """그림 4-1: 사람이 규칙을 쓰는 방식과 모델이 규칙을 찾는 방식."""
    L = _krfont.label
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2))

    # 위쪽: 규칙 기반
    ax = axes[0]
    blank_axes(ax)
    ax.set_title(
        L("규칙 기반 방식 — 사람이 판단 기준을 직접 쓴다",
          "Rule-based: a human writes the rule", ko),
        fontsize=13, pad=10, loc="left",
    )
    box(ax, 0.2, 2.0, 2.2, 2.0, L("사람", "Human", ko), BLUE, 12)
    box(ax, 3.0, 2.0, 3.0, 2.0,
        L("규칙을 직접 작성\n\"60점 이상이면 합격\"",
          "Writes the rule\n\"score >= 60 -> pass\"", ko), ORANGE, 11)
    box(ax, 6.6, 2.0, 3.2, 2.0,
        L("컴퓨터\n규칙을 그대로 적용",
          "Computer\napplies the rule", ko), GRAY, 11)
    arrow(ax, 2.45, 3.0, 2.95, 3.0)
    arrow(ax, 6.05, 3.0, 6.55, 3.0)
    ax.text(5.0, 1.2,
            L("규칙을 말로 쓸 수 없는 문제(고양이 사진 구분)에서는 막힌다",
              "Breaks down when the rule cannot be written in words",  ko),
            ha="center", fontsize=10.5, color=RED)

    # 아래쪽: 머신러닝
    ax = axes[1]
    blank_axes(ax)
    ax.set_title(
        L("머신러닝 방식 — 모델이 데이터에서 판단 기준을 찾는다",
          "Machine learning: the model finds the rule from data", ko),
        fontsize=13, pad=10, loc="left",
    )
    box(ax, 0.2, 2.0, 2.2, 2.0,
        L("사람", "Human", ko), BLUE, 12)
    box(ax, 3.0, 2.0, 3.0, 2.0,
        L("예시 데이터를 준비\n(입력 + 정답)",
          "Prepares examples\n(input + answer)", ko), ORANGE, 11)
    box(ax, 6.6, 2.0, 3.2, 2.0,
        L("모델\n판단 기준을 찾아냄",
          "Model\nlearns the rule", ko), GREEN, 11)
    arrow(ax, 2.45, 3.0, 2.95, 3.0)
    arrow(ax, 6.05, 3.0, 6.55, 3.0)
    ax.text(5.0, 1.2,
            L("규칙이 사라진 것이 아니다. 규칙을 쓰는 주체가 사람에서 모델로 바뀐 것이다",
              "The rule does not disappear; the model writes it instead of the human", ko),
            ha="center", fontsize=10.5, color=GRAY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_train_test(ko: bool, out_path: Path) -> None:
    """그림 4-2: 훈련 데이터와 테스트 데이터를 나누는 이유."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 5.4))

    ax.set_title(
        L("가진 데이터를 둘로 나눈다", "Split the data you have", ko),
        fontsize=13, pad=12, loc="left",
    )

    # 전체 데이터 막대
    box(ax, 0.3, 3.6, 9.4, 0.9,
        L("내가 가진 전체 데이터 160개", "All data I have: 160 rows", ko),
        GRAY, 11)

    arrow(ax, 3.0, 3.5, 2.4, 2.9)
    arrow(ax, 7.0, 3.5, 7.6, 2.9)

    # 훈련 / 테스트
    box(ax, 0.3, 1.7, 6.3, 1.1,
        L("훈련 데이터 112개\n모델이 보고 배우는 데이터",
          "Training set: 112\nthe model learns from this", ko), BLUE, 11)
    box(ax, 6.9, 1.7, 2.8, 1.1,
        L("테스트 데이터 48개\n모델이 못 본 데이터",
          "Test set: 48\nthe model never sees this", ko), ORANGE, 11)

    ax.text(3.45, 1.15,
            L("연습 문제", "practice problems", ko),
            ha="center", fontsize=11, color=BLUE)
    ax.text(8.3, 1.15,
            L("시험 문제", "exam problems", ko),
            ha="center", fontsize=11, color=ORANGE)

    ax.text(5.0, 0.35,
            L("연습 문제만 잘 푸는 학생을 공부 잘한다고 말할 수 없다.\n"
              "모델도 마찬가지여서, 못 본 데이터에서 재야 실력이 드러난다.",
              "A student who only solves practice problems is not proven.\n"
              "The same holds for a model: score it on data it never saw.", ko),
            ha="center", fontsize=10.5, color="#333333", linespacing=1.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 4장 개념 그림 생성 ===")

    targets = [
        ("fig4-1-rule-vs-learning.png", figure_rule_vs_learning,
         "규칙 기반 방식과 머신러닝 방식 비교"),
        ("fig4-2-train-test-split.png", figure_train_test,
         "훈련 데이터와 테스트 데이터 분할"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")


if __name__ == "__main__":
    main()
