"""5장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig5-1  뉴런 하나의 구조 (입력 -> 가중치 곱 -> 합 -> 편향 -> 활성화 -> 출력)
  fig5-2  활성화 함수가 없으면 층을 쌓아도 직선 하나와 같다
  fig5-3  층을 쌓으면 판단 영역이 어떻게 접히는가

fig5-2와 fig5-3은 손으로 그린 도식이 아니라, 아래에 적어 둔 가중치를
실제로 계산해서 그린 그림이다. 쓰는 신경망은 다음과 같다.

  은닉 뉴런 1:  h1 = x1 + x2 - 0.5
  은닉 뉴런 2:  h2 = x1 + x2 - 1.5
  출력 뉴런  :  out = f(h1) - 3 * f(h2) - 0.25

f 가 ReLU면 XOR을 풀고, f 가 없으면(그냥 통과) out 이 직선 하나로 주저앉는다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _krfont

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"

# 판단 영역을 칠할 때 쓰는 옅은 색 (파랑=0, 주황=1)
REGION_CMAP = ListedColormap(["#D6E1EE", "#FBDFC4"])

# 위 주석에 적어 둔 신경망의 가중치
H1_W = (1.0, 1.0, -0.5)     # (x1 가중치, x2 가중치, 편향)
H2_W = (1.0, 1.0, -1.5)
OUT_W = (1.0, -3.0, -0.25)  # (h1 가중치, h2 가중치, 편향)

# XOR 네 점: 두 입력이 다르면 1, 같으면 0
XOR_POINTS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_LABELS = np.array([0, 1, 1, 0])


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
            mutation_scale=15,
            linewidth=1.6,
            color=color,
        )
    )


def blank_axes(ax, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def grid_xy(low: float = -0.6, high: float = 1.6, step: int = 400):
    """판단 영역을 칠하기 위한 격자를 만든다."""
    axis = np.linspace(low, high, step)
    gx, gy = np.meshgrid(axis, axis)
    return gx, gy


def network_output(gx, gy, use_relu: bool):
    """위에 적어 둔 2층 신경망의 출력을 계산한다."""
    h1 = H1_W[0] * gx + H1_W[1] * gy + H1_W[2]
    h2 = H2_W[0] * gx + H2_W[1] * gy + H2_W[2]
    if use_relu:
        h1 = np.maximum(h1, 0.0)
        h2 = np.maximum(h2, 0.0)
    return OUT_W[0] * h1 + OUT_W[1] * h2 + OUT_W[2]


def draw_xor_points(ax, ko: bool, show_legend: bool = False) -> None:
    """XOR 네 점을 찍는다."""
    L = _krfont.label
    for value, color, ko_name, en_name in [
        (0, BLUE, "정답 0", "label 0"),
        (1, ORANGE, "정답 1", "label 1"),
    ]:
        mask = XOR_LABELS == value
        ax.scatter(
            XOR_POINTS[mask, 0], XOR_POINTS[mask, 1],
            c=color, s=190, edgecolor="black", linewidth=1.1, zorder=5,
            label=L(ko_name, en_name, ko),
        )
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.46),
                  ncol=2, fontsize=10, frameon=False)


def figure_neuron_anatomy(ko: bool, out_path: Path) -> None:
    """그림 5-1: 뉴런 하나가 무엇을 계산하는가."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    blank_axes(ax, xlim=(0, 10.4), ylim=(0, 5.6))

    ax.set_title(
        L("뉴런 하나가 하는 계산", "What a single neuron computes", ko),
        fontsize=14, pad=12, loc="left",
    )

    # 입력 두 개
    box(ax, 0.2, 3.3, 1.5, 0.9, L("입력 x1", "input x1", ko), BLUE, 11)
    box(ax, 0.2, 1.3, 1.5, 0.9, L("입력 x2", "input x2", ko), BLUE, 11)

    # 가중치 곱
    box(ax, 2.4, 3.3, 1.7, 0.9, L("x1 × W1", "x1 * W1", ko), ORANGE, 11)
    box(ax, 2.4, 1.3, 1.7, 0.9, L("x2 × W2", "x2 * W2", ko), ORANGE, 11)
    arrow(ax, 1.75, 3.75, 2.35, 3.75)
    arrow(ax, 1.75, 1.75, 2.35, 1.75)

    # 합 + 편향
    box(ax, 4.8, 2.1, 1.9, 1.4,
        L("모두 더하고\n편향 B를 더한다", "sum, then\nadd bias B", ko), GRAY, 11)
    arrow(ax, 4.15, 3.75, 4.75, 3.1)
    arrow(ax, 4.15, 1.75, 4.75, 2.5)

    # 활성화
    box(ax, 7.0, 2.1, 1.7, 1.4,
        L("활성화 함수\nf( z )", "activation\nf( z )", ko), GREEN, 11)
    arrow(ax, 6.75, 2.8, 6.95, 2.8)

    # 출력
    box(ax, 9.0, 2.35, 1.2, 0.9, L("출력", "output", ko), BLUE, 11)
    arrow(ax, 8.75, 2.8, 8.95, 2.8)

    ax.text(5.75, 1.55, "z = W1·x1 + W2·x2 + B",
            ha="center", fontsize=11.5, color="#333333")
    ax.text(5.2, 0.55,
            L("가중치 W는 각 입력을 얼마나 중요하게 볼지 정하고,\n"
              "편향 B는 얼마나 쉽게 켜질지 정한다.",
              "Weights W set how much each input matters;\n"
              "the bias B sets how easily the neuron turns on.", ko),
            ha="center", fontsize=10.5, color="#333333", linespacing=1.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_why_activation(ko: bool, out_path: Path) -> None:
    """그림 5-2: 활성화 함수를 빼면 두 층이 직선 하나로 주저앉는다."""
    L = _krfont.label
    gx, gy = grid_xy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.6))

    panels = [
        (axes[0], False,
         L("활성화 함수 없음 — 경계가 직선 하나", "No activation - one straight line", ko),
         L("두 층을 쌓아도 결과는 out = -2·x1 - 2·x2 + 3.75\n직선 하나짜리 모델과 같다",
           "Even with two layers: out = -2*x1 - 2*x2 + 3.75\nsame as a single straight line", ko)),
        (axes[1], True,
         L("ReLU를 넣음 — 경계가 두 번 꺾인다", "With ReLU - the boundary bends twice", ko),
         L("같은 가중치인데 판단 영역이 띠 모양이 된다.\nXOR 네 점을 모두 맞힌다",
           "Same weights, but the region becomes a band.\nAll four XOR points are correct", ko)),
    ]

    for ax, use_relu, title, caption in panels:
        out = network_output(gx, gy, use_relu)
        ax.contourf(gx, gy, (out > 0).astype(int), levels=[-0.5, 0.5, 1.5],
                    cmap=REGION_CMAP)
        ax.contour(gx, gy, out, levels=[0.0], colors=[GREEN], linewidths=2.4)
        draw_xor_points(ax, ko, show_legend=True)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel("x1", fontsize=11)
        ax.set_ylabel("x2", fontsize=11)
        ax.set_title(title, fontsize=12.5, pad=10)
        ax.text(0.5, -0.26, caption, ha="center", va="top",
                transform=ax.transAxes,
                fontsize=9.5, color="#333333", linespacing=1.5)
        ax.grid(alpha=0.2)

    fig.suptitle(
        L("활성화 함수가 없으면 층을 쌓아도 직선 하나와 같다",
          "Without an activation function, stacked layers stay a single line", ko),
        fontsize=14, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_folding_layers(ko: bool, out_path: Path) -> None:
    """그림 5-3: 은닉 뉴런이 그은 선 두 개를 출력 뉴런이 합친다."""
    L = _krfont.label
    gx, gy = grid_xy()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.2))

    h1 = H1_W[0] * gx + H1_W[1] * gy + H1_W[2]
    h2 = H2_W[0] * gx + H2_W[1] * gy + H2_W[2]
    out = network_output(gx, gy, use_relu=True)

    panels = [
        (axes[0], h1,
         L("은닉 뉴런 1이 그은 선", "hidden neuron 1 draws a line", ko),
         L("주황색이 이 뉴런이 켜지는 영역이다\nx1 + x2 > 0.5 인 쪽에서 켜진다",
           "orange marks where this neuron turns on\nturns on where x1 + x2 > 0.5", ko)),
        (axes[1], h2,
         L("은닉 뉴런 2가 그은 선", "hidden neuron 2 draws a line", ko),
         L("주황색이 이 뉴런이 켜지는 영역이다\nx1 + x2 > 1.5 인 쪽에서 켜진다",
           "orange marks where this neuron turns on\nturns on where x1 + x2 > 1.5", ko)),
        (axes[2], out,
         L("출력 뉴런이 둘을 합친 영역", "output neuron combines the two", ko),
         L("주황색이 1로 판정하는 영역이다\n두 반평면의 차이가 띠 하나로 남는다",
           "orange marks where the output predicts 1\ntwo half-planes leave one band", ko)),
    ]

    for ax, field, title, caption in panels:
        ax.contourf(gx, gy, (field > 0).astype(int), levels=[-0.5, 0.5, 1.5],
                    cmap=REGION_CMAP)
        ax.contour(gx, gy, field, levels=[0.0], colors=[GREEN], linewidths=2.2)
        draw_xor_points(ax, ko, show_legend=True)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel("x1", fontsize=11)
        ax.set_ylabel("x2", fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        ax.text(0.5, -0.26, caption, ha="center", va="top",
                transform=ax.transAxes,
                fontsize=9.5, color="#333333", linespacing=1.5)
        ax.grid(alpha=0.2)

    fig.suptitle(
        L("층을 쌓으면 직선 여러 개가 하나의 판단 영역으로 접힌다",
          "Stacking layers folds several straight lines into one region", ko),
        fontsize=14, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 5장 개념 그림 생성 ===")

    targets = [
        ("fig5-1-neuron-anatomy.png", figure_neuron_anatomy,
         "뉴런 하나의 구조"),
        ("fig5-2-why-activation.png", figure_why_activation,
         "활성화 함수가 없으면 층을 쌓아도 직선 하나"),
        ("fig5-3-folding-layers.png", figure_folding_layers,
         "층을 쌓으면 판단 영역이 접힌다"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    # 그림에 쓴 신경망이 XOR 네 점을 실제로 맞히는지 숫자로 확인한다.
    print()
    print("=== 그림 5-2, 5-3에 쓴 신경망의 XOR 네 점 판정 ===")
    print(f"{'x1':>4}{'x2':>5}{'정답':>6}{'활성화 없음':>12}{'ReLU 사용':>11}")
    for (x1, x2), answer in zip(XOR_POINTS, XOR_LABELS):
        linear = int(network_output(x1, x2, use_relu=False) > 0)
        relu = int(network_output(x1, x2, use_relu=True) > 0)
        print(f"{x1:>4.0f}{x2:>5.0f}{answer:>6d}{linear:>12d}{relu:>11d}")
    print()
    print("이 그림은 개념 설명용이며, 학습으로 찾은 가중치가 아니라 손으로 정한 가중치다.")


if __name__ == "__main__":
    main()
