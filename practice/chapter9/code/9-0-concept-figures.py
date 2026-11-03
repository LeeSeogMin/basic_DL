"""9장 이론 강의용 개념 그림을 만든다.

만드는 그림은 여섯 장이다.
  fig9-1  픽셀과 채널 - 흑백은 숫자 판 한 장, 컬러는 세 장
  fig9-2  합성곱 한 번의 계산 - 작은 창이 이미지 위를 훑으며 곱하고 더한다
  fig9-3  pooling - 크기를 줄이면서 강한 신호만 남긴다
  fig9-4  CNN의 층 순서 - 판의 크기와 장수가 층마다 어떻게 바뀌는가
  fig9-5  실습 9.1 신경망의 일부 - 입력 64, 은닉 32, 출력 10 중 몇 개만 그린다
  fig9-6  컴퓨터 비전의 대표 작업 - 같은 장면에 분류·검출·분할·설명 생성의 출력

이 그림들은 데이터 분석 결과가 아니라 개념 설명용 도식이다.
숫자는 설명을 위해 손으로 고른 값이다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

import _krfont

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def blank_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_grid(
    ax,
    x0,
    y0,
    cell,
    values,
    fmt="{:.0f}",
    facecolor="white",
    fontsize=9,
    textcolor="#222222",
    edgecolor="#B0B0B0",
):
    """왼쪽 위 모서리가 (x0, y0+높이)인 격자를 그리고 각 칸에 숫자를 쓴다.

    facecolor 에 함수를 주면 칸마다 다른 색을 칠한다. 함수는 (행, 열)을 받는다.
    """
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * cell
            y = y0 + (rows - 1 - r) * cell
            fc = facecolor(r, c) if callable(facecolor) else facecolor
            ax.add_patch(
                Rectangle(
                    (x, y), cell, cell,
                    facecolor=fc, edgecolor=edgecolor, linewidth=0.8,
                )
            )
            ax.text(
                x + cell / 2, y + cell / 2, fmt.format(values[r, c]),
                ha="center", va="center", fontsize=fontsize, color=textcolor,
            )


def outline(ax, x, y, w, h, color, linewidth=2.6):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="none", edgecolor=color, linewidth=linewidth)
    )


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=color,
        )
    )


# ---------------------------------------------------------------------------
# 그림 9-1: 픽셀과 채널
# ---------------------------------------------------------------------------

GRAY_PATCH = np.array([
    [0, 0, 90, 180, 60, 0],
    [0, 70, 240, 255, 150, 0],
    [30, 200, 255, 255, 220, 40],
    [60, 240, 255, 255, 240, 80],
    [0, 120, 230, 240, 160, 10],
    [0, 0, 80, 140, 50, 0],
], dtype=float)


def figure_pixel_and_channel(ko: bool, out_path: Path) -> None:
    """그림 9-1: 흑백 이미지는 숫자 판 한 장, 컬러 이미지는 세 장."""
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))

    # 왼쪽: 흑백
    ax = axes[0]
    blank_axes(ax, (-0.4, 6.4), (-1.9, 6.9))
    ax.set_title(
        L("흑백 이미지 = 숫자 판 한 장", "Grayscale image = one sheet of numbers", ko),
        fontsize=13, pad=12,
    )
    draw_grid(
        ax, 0, 0, 1.0, GRAY_PATCH,
        facecolor=lambda r, c: plt.cm.gray(GRAY_PATCH[r, c] / 255.0),
        fontsize=9,
        textcolor="#FFFFFF",
        edgecolor="#666666",
    )
    # 어두운 칸 위의 흰 글씨가 안 보이지 않도록 밝은 칸은 검은 글씨로 덮어 쓴다.
    for r in range(6):
        for c in range(6):
            if GRAY_PATCH[r, c] > 140:
                ax.text(
                    c + 0.5, (5 - r) + 0.5, f"{GRAY_PATCH[r, c]:.0f}",
                    ha="center", va="center", fontsize=9, color="#222222",
                )
    ax.text(
        3.0, -0.7,
        L("칸 하나가 픽셀 하나다. 0이 검정, 255가 흰색이다.",
          "One cell is one pixel. 0 is black, 255 is white.", ko),
        ha="center", fontsize=10.5, color="#333333",
    )
    ax.text(
        3.0, -1.5,
        L("6 x 6 이미지 = 숫자 36개",
          "A 6 x 6 image = 36 numbers", ko),
        ha="center", fontsize=10.5, color=BLUE,
    )

    # 오른쪽: 컬러 3채널
    ax = axes[1]
    blank_axes(ax, (-0.6, 10.2), (-1.9, 6.9))
    ax.set_title(
        L("컬러 이미지 = 숫자 판 세 장", "Color image = three sheets of numbers", ko),
        fontsize=13, pad=12,
    )

    # 뒤에 있는 판부터 그린다. 마지막에 그린 것이 맨 앞이다.
    plane_colors = ["#4C78A8", "#5AA75A", "#C94C4C"]
    plane_names = [L("파랑 판", "Blue plane", ko),
                   L("초록 판", "Green plane", ko),
                   L("빨강 판", "Red plane", ko)]
    for i in range(3):
        dx = 1.8 - 0.9 * i
        dy = 1.8 - 0.9 * i
        ax.add_patch(
            Rectangle(
                (dx, dy), 4.0, 4.0,
                facecolor=plane_colors[i], edgecolor="white",
                linewidth=1.6, alpha=0.85,
            )
        )
        # 격자선
        for k in range(1, 4):
            ax.plot([dx + k, dx + k], [dy, dy + 4.0], color="white", linewidth=0.7)
            ax.plot([dx, dx + 4.0], [dy + k, dy + k], color="white", linewidth=0.7)
    # 이름표는 판 위에 겹치지 않도록 판 오른쪽 바깥에 세로로 쌓아 붙인다.
    # 위에서부터 파랑, 초록, 빨강 순서라서 판이 쌓인 순서와 그대로 맞는다.
    for i in range(3):
        y = 5.5 - 0.9 * i
        ax.add_patch(
            Rectangle(
                (6.2, y - 0.22), 0.44, 0.44,
                facecolor=plane_colors[i], edgecolor="white", linewidth=1.0,
            )
        )
        ax.text(
            6.8, y, plane_names[i],
            ha="left", va="center", fontsize=11, color=plane_colors[i],
        )

    ax.text(
        4.8, -0.7,
        L("같은 자리의 숫자 세 개가 색 하나를 만든다.",
          "Three numbers at the same position make one color.", ko),
        ha="center", fontsize=10.5, color="#333333",
    )
    ax.text(
        4.8, -1.5,
        L("빨강 200, 초록 40, 파랑 60 -> 붉은색 픽셀",
          "R 200, G 40, B 60 -> a reddish pixel", ko),
        ha="center", fontsize=10.5, color=RED,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 9-2: 합성곱 한 번
# ---------------------------------------------------------------------------

CONV_INPUT = np.array([
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0],
], dtype=float)

CONV_KERNEL = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=float)


def figure_convolution_step(ko: bool, out_path: Path) -> None:
    """그림 9-2: 3x3 창 하나에서 곱하고 더하는 과정."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    blank_axes(ax, (-0.5, 13.6), (-2.6, 7.8))

    cell = 0.9

    # 입력 이미지 (0.4 ~ 4.9 를 차지한다)
    draw_grid(ax, 0.4, 1.6, cell, CONV_INPUT, facecolor="#EFF3F8", fontsize=11)
    ax.text(
        0.4 + 2.5 * cell, 1.6 + 5 * cell + 0.95,
        L("입력 이미지 (5 x 5)", "Input image (5 x 5)", ko),
        ha="center", fontsize=11.5, color="#333333",
    )
    # 왼쪽 위 3x3 창 표시
    win_x = 0.4
    win_y = 1.6 + 2 * cell
    outline(ax, win_x, win_y, 3 * cell, 3 * cell, ORANGE)
    ax.text(
        win_x + 1.5 * cell, win_y + 3 * cell + 0.15,
        L("지금 보는 창", "current window", ko),
        ha="center", fontsize=10, color=ORANGE,
    )

    # 필터 (5.9 ~ 8.6 을 차지한다)
    fx = 5.9
    draw_grid(ax, fx, 2.5, cell, CONV_KERNEL, facecolor="#FDECD9", fontsize=11)
    ax.text(
        fx + 1.5 * cell, 2.5 + 3 * cell + 0.45,
        L("필터 (3 x 3)", "Filter (3 x 3)", ko),
        ha="center", fontsize=11.5, color=ORANGE,
    )
    ax.text(
        fx + 1.5 * cell, 2.5 - 0.5,
        L("세로선 검출 필터", "vertical-edge filter", ko),
        ha="center", fontsize=10, color="#333333",
    )

    ax.text(5.40, 3.85, "x", ha="center", va="center", fontsize=16, color=GRAY)
    ax.text(9.15, 3.85, "=", ha="center", va="center", fontsize=16, color=GRAY)

    # 출력 (9.7 ~ 12.4 를 차지한다)
    out_values = np.full((3, 3), np.nan)
    out_values[0, 0] = float(np.sum(CONV_INPUT[0:3, 0:3] * CONV_KERNEL))
    ox = 9.7

    def out_fmt(v):
        return "?" if np.isnan(v) else f"{v:.0f}"

    rows, cols = out_values.shape
    for r in range(rows):
        for c in range(cols):
            x = ox + c * cell
            y = 2.5 + (rows - 1 - r) * cell
            filled = not np.isnan(out_values[r, c])
            ax.add_patch(
                Rectangle(
                    (x, y), cell, cell,
                    facecolor="#E6F0E2" if filled else "#F5F5F5",
                    edgecolor=GREEN if filled else "#CCCCCC",
                    linewidth=2.0 if filled else 0.8,
                )
            )
            ax.text(
                x + cell / 2, y + cell / 2, out_fmt(out_values[r, c]),
                ha="center", va="center", fontsize=11,
                color="#222222" if filled else "#AAAAAA",
            )
    ax.text(
        ox + 1.5 * cell, 2.5 + 3 * cell + 0.45,
        L("출력 (3 x 3)", "Output (3 x 3)", ko),
        ha="center", fontsize=11.5, color=GREEN,
    )
    ax.text(
        ox + 1.5 * cell, 2.5 - 0.5,
        L("주황 창 하나가 초록 칸 하나가 된다",
          "one orange window -> one green cell", ko),
        ha="center", fontsize=10, color="#333333",
    )
    ax.text(
        ox + 1.5 * cell, 2.5 - 1.15,
        L("창을 한 칸씩 옮겨 나머지 칸을 채운다",
          "slide the window to fill the rest", ko),
        ha="center", fontsize=10, color="#333333",
    )

    # 계산식
    ax.text(
        6.5, 0.55,
        L("창의 9개 숫자와 필터의 9개 숫자를 자리마다 곱해서 전부 더한다",
          "multiply the 9 window numbers by the 9 filter numbers, then add them up", ko),
        ha="center", fontsize=11, color="#333333",
    )
    ax.text(
        6.5, -0.35,
        "(0x-1)+(0x0)+(1x1) + (0x-1)+(1x0)+(1x1) + (1x-1)+(1x0)+(0x1) = 1",
        ha="center", fontsize=11.5, color=GREEN,
    )
    ax.text(
        6.5, -1.4,
        L("숫자가 커진 자리에는 필터가 찾는 무늬가 있다.\n"
          "이 필터는 왼쪽이 어둡고 오른쪽이 밝은 자리에서 큰 값을 낸다.",
          "A large value means the pattern the filter looks for is there.\n"
          "This filter fires where the left side is dark and the right side is bright.", ko),
        ha="center", fontsize=10.5, color=GRAY, linespacing=1.6,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 9-3: pooling
# ---------------------------------------------------------------------------

POOL_INPUT = np.array([
    [1, 3, 2, 1],
    [4, 6, 0, 2],
    [5, 2, 7, 3],
    [1, 0, 4, 8],
], dtype=float)


def figure_pooling(ko: bool, out_path: Path) -> None:
    """그림 9-3: 2x2 최대 pooling."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    blank_axes(ax, (-0.5, 12.5), (-2.4, 6.4))

    cell = 1.0
    block_colors = ["#DCE6F1", "#FDE6CE", "#DFEEDA", "#F8DADA"]
    block_edges = [BLUE, ORANGE, GREEN, RED]

    def block_index(r, c):
        return (r // 2) * 2 + (c // 2)

    draw_grid(
        ax, 0.5, 1.4, cell, POOL_INPUT,
        facecolor=lambda r, c: block_colors[block_index(r, c)],
        fontsize=12,
    )
    # 2x2 덩어리 테두리
    for br in range(2):
        for bc in range(2):
            outline(
                ax,
                0.5 + bc * 2 * cell,
                1.4 + (1 - br) * 2 * cell,
                2 * cell, 2 * cell,
                block_edges[br * 2 + bc],
                linewidth=2.2,
            )
    ax.text(
        0.5 + 2 * cell, 1.4 + 4 * cell + 0.45,
        L("입력 (4 x 4)", "Input (4 x 4)", ko),
        ha="center", fontsize=12, color="#333333",
    )

    arrow(ax, 5.1, 3.4, 7.1, 3.4, GRAY)
    ax.text(
        6.1, 3.75,
        L("2 x 2 최대 pooling", "2 x 2 max pooling", ko),
        ha="center", fontsize=11, color=GRAY,
    )

    pooled = np.array([[6, 2], [5, 8]], dtype=float)
    draw_grid(
        ax, 7.7, 2.4, cell, pooled,
        facecolor=lambda r, c: block_colors[r * 2 + c],
        fontsize=13,
    )
    for r in range(2):
        for c in range(2):
            outline(
                ax, 7.7 + c * cell, 2.4 + (1 - r) * cell, cell, cell,
                block_edges[r * 2 + c], linewidth=2.2,
            )
    ax.text(
        7.7 + cell, 2.4 + 2 * cell + 0.45,
        L("출력 (2 x 2)", "Output (2 x 2)", ko),
        ha="center", fontsize=12, color="#333333",
    )

    ax.text(
        6.0, 0.55,
        L("덩어리마다 가장 큰 숫자 하나만 남긴다. 숫자 16개가 4개로 줄었다.",
          "Keep only the largest number in each block: 16 numbers become 4.", ko),
        ha="center", fontsize=11, color="#333333",
    )
    ax.text(
        6.0, -0.35,
        L("파란 덩어리의 1, 3, 4, 6 중에서 6을 고른다",
          "From the blue block 1, 3, 4, 6, pick 6", ko),
        ha="center", fontsize=10.5, color=BLUE,
    )
    ax.text(
        6.0, -1.5,
        L("자리가 한두 칸 흔들려도 가장 큰 값은 잘 바뀌지 않는다.\n"
          "그래서 pooling 뒤에는 위치가 조금 달라진 그림도 비슷한 숫자가 나온다.",
          "The maximum barely changes when the pattern shifts by a cell or two.\n"
          "So after pooling, slightly shifted images give similar numbers.", ko),
        ha="center", fontsize=10.5, color=GRAY, linespacing=1.6,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 9-4: CNN의 층 순서
# ---------------------------------------------------------------------------

def stack(ax, x, y, size, count, color, gap=0.22):
    """같은 크기의 판 count장을 비스듬히 겹쳐 그린다. 맨 앞 판의 왼쪽 아래가 (x, y)다."""
    for i in reversed(range(count)):
        ax.add_patch(
            Rectangle(
                (x + i * gap, y + i * gap), size, size,
                facecolor=color, edgecolor="white", linewidth=1.4, alpha=0.9,
            )
        )


def figure_cnn_structure(ko: bool, out_path: Path) -> None:
    """그림 9-4: 8x8 입력에 필터 4개를 쓰는 작은 CNN의 층 순서."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    blank_axes(ax, (-0.3, 16.2), (-2.6, 5.4))

    # 입력 8x8 한 장
    stack(ax, 0.0, 0.6, 2.4, 1, "#9E9E9E")
    ax.text(1.2, 3.4, L("입력\n8x8, 1장", "input\n8x8, 1 sheet", ko),
            ha="center", fontsize=11)

    arrow(ax, 2.7, 1.8, 3.9, 1.8)
    ax.text(3.3, 2.15, L("합성곱\n필터 4개", "conv\n4 filters", ko),
            ha="center", fontsize=10, color=ORANGE)

    # 합성곱 결과 6x6 네 장
    stack(ax, 4.1, 0.6, 1.8, 4, "#F5B97A")
    ax.text(5.35, 3.45, L("6x6, 4장\n필터 하나에 판 하나", "6x6, 4 sheets\none per filter", ko),
            ha="center", fontsize=10.5)

    arrow(ax, 6.9, 1.8, 8.1, 1.8)
    ax.text(7.5, 2.15, L("ReLU\n2x2 pooling", "ReLU\n2x2 pooling", ko),
            ha="center", fontsize=10, color=BLUE)

    # pooling 결과 3x3 네 장
    stack(ax, 8.3, 0.9, 0.9, 4, "#9CB8D8")
    ax.text(9.1, 3.0, L("3x3, 4장", "3x3, 4 sheets", ko), ha="center", fontsize=10.5)

    arrow(ax, 10.1, 1.8, 11.0, 1.8)
    ax.text(10.55, 2.15, L("펴기", "flatten", ko), ha="center", fontsize=10, color=GRAY)

    # 펴서 36개
    for i in range(12):
        ax.add_patch(Rectangle((11.2, 0.0 + i * 0.3), 0.3, 0.3,
                               facecolor="#9CB8D8", edgecolor="white", linewidth=0.8))
    ax.text(11.35, 3.9, L("숫자 36개", "36 numbers", ko), ha="center", fontsize=10.5)
    ax.text(11.35, -0.35, "...", ha="center", fontsize=12, color=GRAY)

    arrow(ax, 11.8, 1.8, 12.9, 1.8)
    ax.text(12.35, 2.15, L("완전연결층", "dense", ko), ha="center", fontsize=10, color=GREEN)

    for i in range(10):
        ax.add_patch(Rectangle((13.1, 0.0 + i * 0.36), 0.36, 0.36,
                               facecolor="#B7D9AE", edgecolor="white", linewidth=0.8))
        ax.text(13.7, 0.18 + i * 0.36, str(i), va="center", fontsize=9, color="#333333")
    ax.text(13.6, 3.9, L("숫자 0~9의\n점수 10개", "10 scores\nfor 0-9", ko),
            ha="center", fontsize=10.5)

    ax.text(
        8.0, -1.35,
        L("판 크기는 8 -> 6 -> 3으로 줄고, 판 장수는 1 -> 4로 는다. 학습으로 찾는 값은 필터 4개의 숫자 36개와 완전연결층의 가중치다.",
          "Sheet size shrinks 8 -> 6 -> 3 while the number of sheets grows 1 -> 4.", ko),
        ha="center", fontsize=10.5, color="#333333",
    )
    ax.text(
        8.0, -2.15,
        L("이 그림은 층 순서와 판 크기를 보여 주는 개념 도식이다.",
          "Concept diagram of layer order and sheet sizes.", ko),
        ha="center", fontsize=10, color=GRAY,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 9-5: 실습 9.1 신경망의 일부
# ---------------------------------------------------------------------------

def column_positions(n_top, n_bottom, y_top, step):
    """위쪽 n_top개, 점 세 개 자리 하나, 아래쪽 n_bottom개의 y 좌표를 돌려준다."""
    top = [y_top - i * step for i in range(n_top)]
    dots = y_top - n_top * step
    bottom = [dots - (i + 1) * step for i in range(n_bottom)]
    return top, dots, bottom


def figure_mlp_partial(ko: bool, out_path: Path) -> None:
    """그림 9-5: 64 -> 32 -> 10 신경망에서 층마다 몇 개만 골라 그린다."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    blank_axes(ax, (-0.6, 16.0), (-1.9, 7.4))

    # 왼쪽: 8x8 이미지 자리
    cell = 0.3
    draw_grid(ax, -0.3, 2.0, cell, np.arange(64).reshape(8, 8),
              facecolor="#E6E6E6", fontsize=5.5, textcolor="#555555")
    ax.text(0.9, 4.65, L("8 x 8 이미지\n(칸 번호)", "8 x 8 image\n(cell numbers)", ko),
            ha="center", fontsize=10.5)
    arrow(ax, 2.3, 3.2, 3.5, 3.2)
    ax.text(2.9, 3.5, L("펴기", "flatten", ko), ha="center", fontsize=10, color=GRAY)

    r = 0.27
    x_in, x_hid, x_out = 4.6, 9.0, 13.0
    step = 0.95
    in_top, in_dots, in_bot = column_positions(3, 2, 5.9, step)
    hid_top, hid_dots, hid_bot = column_positions(3, 1, 5.4, step * 1.15)
    out_top, out_dots, out_bot = column_positions(3, 1, 5.4, step * 1.15)

    inputs = list(zip(in_top + in_bot, ["0", "1", "2", "62", "63"]))
    hiddens = list(zip(hid_top + hid_bot, ["1", "2", "3", "32"]))
    outputs = list(zip(out_top + out_bot, ["0", "1", "2", "9"]))

    # 연결선: 그린 노드끼리 전부 잇고, 은닉 뉴런 1이 받는 선만 주황으로 강조한다
    for yi, _ in inputs:
        for k, (yh, _) in enumerate(hiddens):
            color, lw, z = (ORANGE, 1.8, 2) if k == 0 else ("#C8C8C8", 0.8, 1)
            ax.plot([x_in + r, x_hid - r], [yi, yh], color=color, linewidth=lw, zorder=z)
    for yh, _ in hiddens:
        for yo, _ in outputs:
            ax.plot([x_hid + r, x_out - r], [yh, yo], color="#C8C8C8", linewidth=0.8, zorder=1)

    def nodes(x, items, face, edge):
        for y, _ in items:
            ax.add_patch(Circle((x, y), r, facecolor=face, edgecolor=edge,
                                linewidth=1.4, zorder=3))

    nodes(x_in, inputs, "#E6E6E6", GRAY)
    nodes(x_hid, hiddens, "#C9D7EA", BLUE)
    nodes(x_out, outputs, "#CFE6C9", GREEN)
    ax.add_patch(Circle((x_hid, hiddens[0][0]), r, facecolor="#FBD3A8",
                        edgecolor=ORANGE, linewidth=2.0, zorder=4))

    for y, name in inputs:
        ax.text(x_in - r - 0.12, y, L(f"{name}번 칸", f"cell {name}", ko),
                ha="right", va="center", fontsize=10)
    for y, name in hiddens:
        ax.text(x_hid, y, name, ha="center", va="center", fontsize=9.5, zorder=5)
    for y, name in outputs:
        ax.text(x_out + r + 0.15, y, L(f"숫자 {name}의 점수", f"score for {name}", ko),
                ha="left", va="center", fontsize=10)
    for x, y in [(x_in, in_dots), (x_hid, hid_dots), (x_out, out_dots)]:
        ax.text(x, y, "⋮", ha="center", va="center", fontsize=20, color=GRAY)

    ax.text(x_in, 6.75, L("입력 64개", "input 64", ko), ha="center", fontsize=12.5)
    ax.text(x_hid, 6.75, L("은닉층 뉴런 32개", "hidden 32", ko), ha="center",
            fontsize=12.5, color=BLUE)
    ax.text(x_out, 6.75, L("출력 10개", "output 10", ko), ha="center",
            fontsize=12.5, color=GREEN)

    ax.text((x_in + x_hid) / 2, -0.55,
            L("가중치 64 x 32 = 2,048개", "weights 64 x 32 = 2,048", ko),
            ha="center", fontsize=11)
    ax.text((x_hid + x_out) / 2, -0.55,
            L("가중치 32 x 10 = 320개", "weights 32 x 10 = 320", ko),
            ha="center", fontsize=11)
    ax.text(x_hid + 0.45, hiddens[0][0] + 0.55,
            L("뉴런 1은 64개를 전부 받는다\n(가중치 64개 + 편향 1개)",
              "neuron 1 takes all 64\n(64 weights + 1 bias)", ko),
            ha="left", va="bottom", fontsize=10, color=ORANGE)

    ax.text(
        7.7, -1.5,
        L("층마다 일부만 그린 개념 도식이다. 입력 64개 중 5개, 은닉층 32개 중 4개, 출력 10개 중 4개를 그렸다.",
          "Concept diagram showing only some nodes of each layer.", ko),
        ha="center", fontsize=10, color=GRAY,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 9-6: 같은 장면, 다른 출력 - 컴퓨터 비전의 대표 작업
# ---------------------------------------------------------------------------

SCENE_COLORS = {          # 클래스 번호 -> (원본 장면 색, 분할 결과 색)
    0: ("#CFE3F5", "#F4F4F4"),   # 하늘
    1: ("#8C8C8C", "#B0B0B0"),   # 도로
    2: ("#9CC98A", "#D5E8CC"),   # 풀밭
    3: ("#6F87A3", BLUE),        # 자동차
    4: ("#A8846A", ORANGE),      # 사람
}


def make_scene() -> np.ndarray:
    """60x90 칸짜리 장면을 클래스 번호 판으로 만든다. 손으로 정한 도형이다."""
    h, w = 60, 90
    scene = np.zeros((h, w), dtype=int)          # 하늘
    scene[30:, :] = 2                            # 풀밭
    scene[38:54, :] = 1                          # 도로
    scene[36:48, 12:46] = 3                      # 자동차 몸통
    scene[29:37, 20:38] = 3                      # 자동차 지붕
    yy, xx = np.mgrid[0:h, 0:w]
    for cx in (19, 39):                          # 바퀴
        scene[(yy - 48) ** 2 + (xx - cx) ** 2 <= 16] = 3
    scene[(yy - 21) ** 2 + (xx - 67) ** 2 <= 12] = 4   # 사람 머리
    scene[25:41, 64:71] = 4                      # 몸통
    scene[41:53, 64:67] = 4                      # 다리
    scene[41:53, 68:71] = 4
    return scene


def paint(scene: np.ndarray, which: int) -> np.ndarray:
    """클래스 번호 판을 색 이미지로 바꾼다. which=0 원본 색, 1 분할 색."""
    rgb = np.zeros(scene.shape + (3,))
    for cls, colors in SCENE_COLORS.items():
        rgb[scene == cls] = matplotlib_rgb(colors[which])
    return rgb


def matplotlib_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def bounding_box(scene: np.ndarray, cls: int) -> tuple[int, int, int, int]:
    """클래스 칸을 모두 감싸는 가장 작은 상자 (x0, y0, x1, y1)."""
    ys, xs = np.nonzero(scene == cls)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def figure_vision_tasks(ko: bool, out_path: Path) -> None:
    """그림 9-6: 같은 장면에 대해 네 작업이 내놓는 출력을 나란히 그린다."""
    L = _krfont.label
    scene = make_scene()
    original = paint(scene, 0)

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9))
    titles = [
        L("이미지 분류", "classification", ko),
        L("객체 검출", "object detection", ko),
        L("영상 분할", "segmentation", ko),
        L("이미지 설명 생성", "captioning", ko),
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # 1) 분류: 사진 전체에 이름표 하나
    axes[0].imshow(original, interpolation="nearest")
    axes[0].text(45, 57, L("출력: 도로 장면", "output: road scene", ko),
                 ha="center", va="bottom", fontsize=11, color="white",
                 bbox=dict(facecolor=GRAY, edgecolor="none", pad=3))

    # 2) 검출: 물체마다 상자와 이름표
    axes[1].imshow(original, interpolation="nearest")
    for cls, name, color in ((3, L("자동차", "car", ko), BLUE),
                             (4, L("사람", "person", ko), ORANGE)):
        x0, y0, x1, y1 = bounding_box(scene, cls)
        axes[1].add_patch(Rectangle((x0 - 1.5, y0 - 1.5), x1 - x0 + 2, y1 - y0 + 2,
                                    facecolor="none", edgecolor=color, linewidth=2.6))
        axes[1].text(x0 - 1.5, y0 - 2.5, name, fontsize=10.5, color="white",
                     va="bottom", bbox=dict(facecolor=color, edgecolor="none", pad=2))

    # 3) 분할: 픽셀마다 이름표 -> 색으로 칠한다
    axes[2].imshow(paint(scene, 1), interpolation="nearest")
    legend = [(3, L("자동차", "car", ko)), (4, L("사람", "person", ko)),
              (1, L("도로", "road", ko)), (2, L("풀밭", "grass", ko)),
              (0, L("하늘", "sky", ko))]
    for i, (cls, name) in enumerate(legend):
        axes[2].add_patch(Rectangle((2 + i * 17.5, 1.5), 4, 4,
                                    facecolor=SCENE_COLORS[cls][1], edgecolor="#555555",
                                    linewidth=0.6))
        axes[2].text(7.0 + i * 17.5, 3.6, name, va="center", fontsize=9.5)

    # 4) 설명 생성: 문장 한 줄
    axes[3].imshow(original, interpolation="nearest")
    axes[3].text(45, 2, L("출력: \"도로에 자동차가 있고\n옆에 사람이 서 있다\"",
                           "output: \"a car on the road,\na person standing nearby\"", ko),
                 ha="center", va="top", fontsize=10.5, color="white",
                 bbox=dict(facecolor=GREEN, edgecolor="none", pad=3))

    fig.text(
        0.5, 0.02,
        L("같은 장면을 넣어도 작업에 따라 출력이 다르다. 장면은 설명용으로 손으로 그린 도형이다.",
          "Same input, different outputs. The scene is hand-drawn for illustration.", ko),
        ha="center", fontsize=10.5, color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 9장 개념 그림 생성 ===")

    targets = [
        ("fig9-1-pixel-and-channel.png", figure_pixel_and_channel,
         "픽셀과 채널"),
        ("fig9-2-convolution-step.png", figure_convolution_step,
         "합성곱 한 번의 계산"),
        ("fig9-3-pooling.png", figure_pooling,
         "pooling이 하는 일"),
        ("fig9-4-cnn-structure.png", figure_cnn_structure,
         "CNN의 층 순서"),
        ("fig9-5-mlp-partial.png", figure_mlp_partial,
         "실습 9.1 신경망의 일부"),
        ("fig9-6-vision-tasks.png", figure_vision_tasks,
         "컴퓨터 비전의 대표 작업"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")
    print("그림 안의 숫자는 설명을 위해 손으로 고른 값이다.")


if __name__ == "__main__":
    main()
