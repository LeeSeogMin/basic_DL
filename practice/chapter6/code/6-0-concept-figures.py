"""6장 이론 강의용 개념 그림을 만든다.

만드는 그림은 다섯 장이다.
  fig6-cross-entropy 분류의 손실 - 정답 쪽 확률이 낮을수록 손실이 커진다
  fig6-2            경사하강법의 한 걸음 - 기울기를 보고 내려갈 방향과 보폭을 정한다
  fig6-backprop     역전파 - 출력 쪽에서 입력 쪽으로 곱셈을 이어 가며 기울기를 구한다
  fig6-3            epoch와 batch - 전체를 몇 번 도는가, 한 번에 몇 개씩 보는가
  fig6-4            학습 곡선 읽는 법 - 곡선 모양 다섯 가지와 세로축 눈금 함정

이 그림들은 실험 결과가 아니라 개념 설명용 도식이다.
교차 엔트로피와 역전파 그림의 숫자는 아래 상수로 직접 계산해 로그에 남긴다.
회귀의 평균제곱오차 예시(점 5개)는 그림 없이 숫자만 로그에 남긴다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _krfont

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"

# 평균제곱오차 예시 (점 5개). 개념 설명용으로 손으로 고른 값이다.
EXAMPLE_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
EXAMPLE_Y = np.array([3.0, 5.8, 6.0, 9.0, 9.4])
EXAMPLE_SLOPE = 1.5
EXAMPLE_INTERCEPT = 2.0


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


def arrow(ax, x1, y1, x2, y2, color=GRAY, width=1.6):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=width,
            color=color,
        )
    )


def mse_example() -> tuple[np.ndarray, float]:
    """회귀의 평균제곱오차 예시: 차이를 재고, 제곱하고, 평균 낸다."""
    pred = EXAMPLE_SLOPE * EXAMPLE_X + EXAMPLE_INTERCEPT
    diff = EXAMPLE_Y - pred
    return diff, float((diff ** 2).mean())


def figure_one_step(ko: bool, out_path: Path) -> None:
    """그림 6-2: 한 걸음을 정하는 방법."""
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    grid = np.linspace(-1.5, 7.5, 200)
    curve = (grid - 3.0) ** 2 + 2.0

    cases = [
        (0.5, L("지금 위치가 최저점 왼쪽", "current point is left of the bottom", ko)),
        (6.0, L("지금 위치가 최저점 오른쪽", "current point is right of the bottom", ko)),
    ]

    for ax, (w_now, subtitle) in zip(axes, cases):
        loss_now = (w_now - 3.0) ** 2 + 2.0
        slope = 2.0 * (w_now - 3.0)

        ax.plot(grid, curve, color=BLUE, linewidth=2.6)
        ax.axvline(3.0, color=GRAY, linestyle=":", linewidth=1.4)
        ax.text(3.0, 1.3, L("손실이 가장 낮은 곳", "lowest loss", ko),
                ha="center", fontsize=9.5, color=GRAY,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none"))

        # 접선: 지금 위치에서의 기울기
        tangent_x = np.linspace(w_now - 1.6, w_now + 1.6, 20)
        ax.plot(tangent_x, loss_now + slope * (tangent_x - w_now),
                color=ORANGE, linewidth=2.2, linestyle="--")
        ax.scatter([w_now], [loss_now], s=130, color=RED, zorder=4,
                   edgecolor="white", linewidth=1.0)

        # 내려갈 방향 화살표
        direction = -1.0 if slope > 0 else 1.0
        arrow(ax, w_now, loss_now + 1.3, w_now + direction * 1.4, loss_now + 1.3,
              color=GREEN, width=2.2)
        ax.text(w_now + direction * 0.7, loss_now + 2.2,
                L("이쪽으로 간다", "go this way", ko),
                ha="center", fontsize=10.5, color=GREEN)

        ax.text(
            3.0, 14.6,
            L(f"기울기 = {slope:+.1f}", f"slope = {slope:+.1f}", ko),
            ha="center", fontsize=11.5, color=ORANGE,
        )

        ax.set_xlim(-1.5, 7.5)
        ax.set_ylim(0, 17.0)
        ax.set_xlabel(L("바꿀 수 있는 값 w", "the value w we can change", ko), fontsize=11)
        ax.set_ylabel(L("손실", "loss", ko), fontsize=11)
        ax.set_title(subtitle, fontsize=12, pad=10)
        ax.grid(alpha=0.25)

    fig.suptitle(
        L("한 걸음 = 기울기가 알려준 반대 방향으로, 학습률만큼",
          "One step = opposite to the slope, scaled by the learning rate", ko),
        fontsize=14,
    )
    fig.text(
        0.5, 0.015,
        L("새 위치 = 지금 위치 - 학습률 × 기울기      (기울기가 양수면 왼쪽, 음수면 오른쪽)",
          "new w = current w - learning rate x slope   (slope > 0: move left, slope < 0: move right)", ko),
        ha="center", fontsize=11.5, color="#333333",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_epoch_batch(ko: bool, out_path: Path) -> None:
    """그림 6-3: epoch와 batch의 관계."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    ax.set_title(
        L("데이터 12개 · 배치 크기 4 · 에포크 3회",
          "12 rows, batch size 4, 3 epochs", ko),
        fontsize=14, pad=14, loc="left",
    )

    batch_colors = [BLUE, ORANGE, GREEN]

    for epoch in range(3):
        row_y = 6.3 - epoch * 1.9
        ax.text(0.35, row_y + 0.4,
                L(f"에포크 {epoch + 1}", f"epoch {epoch + 1}", ko),
                ha="left", va="center", fontsize=11.5, color="#333333")

        for batch in range(3):
            x0 = 2.6 + batch * 3.5
            for k in range(4):
                ax.add_patch(
                    plt.Rectangle(
                        (x0 + k * 0.62, row_y + 0.15), 0.5, 0.5,
                        facecolor=batch_colors[batch], alpha=0.85,
                        edgecolor="white", linewidth=1.0,
                    )
                )
            arrow(ax, x0 + 1.24, row_y + 0.05, x0 + 1.24, row_y - 0.55, color=GRAY)
            ax.text(x0 + 1.24, row_y - 0.85,
                    L("갱신 1번", "1 update", ko),
                    ha="center", fontsize=9.5, color=GRAY)

    # 배치 이름표
    for batch in range(3):
        x0 = 2.6 + batch * 3.5
        ax.text(x0 + 1.24, 7.45,
                L(f"배치 {batch + 1} (4개)", f"batch {batch + 1} (4)", ko),
                ha="center", fontsize=11, color=batch_colors[batch])

    box(ax, 0.35, 0.15, 12.9, 0.85,
        L("에포크 1회 = 전체 12개를 한 번 다 본다 = 가중치 갱신 3번   |   "
          "에포크 3회 = 갱신 9번",
          "1 epoch = all 12 rows seen once = 3 updates   |   3 epochs = 9 updates", ko),
        GRAY, 11.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# 그림 6-4 아랫줄에서 쓰는 눈금 함정 예시. 본문 6절에 적은 두 숫자를 그대로 쓴다.
SCALE_TRAP_HIGH = (237.0, 4.0)
SCALE_TRAP_LOW = (4.1, 4.0)


def figure_curve_shapes(ko: bool, out_path: Path) -> None:
    """그림 6-4: 학습 곡선의 모양 다섯 가지와 세로축 눈금 함정.

    윗줄 다섯 칸은 표 6.3의 다섯 줄과 같은 순서다. 축에 숫자를 넣지 않는다.
    모양만 비교하는 도식이고 실험 결과가 아니라는 것을 눈금 없음으로 나타낸다.
    """
    L = _krfont.label
    e = np.linspace(0.0, 1.0, 400)

    shapes = [
        (
            L("① 빠르게 내려가다\n평평해진다", "1) drops fast,\nthen flattens", ko),
            0.08 + 0.92 * np.exp(-6.0 * e),
            GREEN,
        ),
        (
            L("② 아주 천천히\n내려간다", "2) drops\nvery slowly", ko),
            0.08 + 0.92 * np.exp(-0.45 * e),
            ORANGE,
        ),
        (
            L("③ 위로 솟거나\n튄다", "3) rises or\nbounces", ko),
            0.16 * np.exp(1.5 * e) + 0.07 * np.exp(1.5 * e) * np.sin(13.0 * e),
            RED,
        ),
        (
            L("④ 처음부터\n평평하다", "4) flat from\nthe start", ko),
            np.full_like(e, 0.72),
            RED,
        ),
        (
            L("⑤ 지그재그로 흔들리며\n내려간다", "5) zigzags on\nthe way down", ko),
            0.10 + 0.90 * np.exp(-4.0 * e)
            + 0.06 * np.sin(34.0 * e) + 0.035 * np.sin(57.0 * e),
            BLUE,
        ),
    ]

    # tight_layout은 add_subplot으로 만든 칸의 폭을 고르게 잡지 못해
    # 칸마다 크기가 달라진다. 그래서 여백을 GridSpec에 직접 준다.
    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(
        2, 10, height_ratios=[1.0, 1.25],
        left=0.085, right=0.985, top=0.845, bottom=0.095,
        hspace=0.55, wspace=0.95,
    )

    for i, (title, curve, color) in enumerate(shapes):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        ax.plot(e, curve, color=color, linewidth=2.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.45)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5, pad=9, linespacing=1.4)
        ax.set_xlabel(L("에포크", "epoch", ko), fontsize=9.5, labelpad=3)
        if i == 0:
            ax.set_ylabel(L("손실", "loss", ko), fontsize=9.5, labelpad=10)

    # 아랫줄: 모양은 같고 세로축 눈금만 다른 두 곡선
    traps = [
        (gs[1, 0:5], SCALE_TRAP_HIGH, (0.0, 265.0), BLUE),
        (gs[1, 5:10], SCALE_TRAP_LOW, (3.975, 4.125), ORANGE),
    ]

    for spec, (start, end), ylim, color in traps:
        ax = fig.add_subplot(spec)
        ax.plot(e, end + (start - end) * np.exp(-5.0 * e), color=color, linewidth=2.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_xlabel(L("에포크", "epoch", ko), fontsize=10)
        ax.set_ylabel(L("손실", "loss", ko), fontsize=10)
        ax.set_title(
            L(f"손실 {start:g} 에서 {end:g} 까지 — {start - end:g} 만큼 줄었다",
              f"loss {start:g} to {end:g} - dropped by {start - end:g}", ko),
            fontsize=11.5, pad=8, color=color,
        )
        ax.grid(alpha=0.25)

    fig.suptitle(
        L("학습 곡선 — 모양 다섯 가지(위)와 세로축 눈금 함정(아래)",
          "Learning curves - five shapes (top) and the y-axis scale trap (bottom)", ko),
        fontsize=14, y=0.962,
    )
    fig.text(
        0.5, 0.022,
        L("아래 두 곡선의 모양은 같다. 세로축 눈금을 읽지 않으면 둘을 같은 성과로 본다.",
          "The two curves below have the same shape. Without reading the y-axis, "
          "they look like the same result.", ko),
        ha="center", fontsize=11.5, color="#333333",
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# 교차 엔트로피 그림에서 점을 찍을 확률. "정답 쪽에 준 확률"이다.
CE_PROBS = [0.9, 0.5, 0.1, 0.01]


def figure_cross_entropy(ko: bool, out_path: Path) -> list[tuple[float, float, float]]:
    """교차 엔트로피 그림: 정답 쪽 확률 p에 대한 손실 -log(p).

    왼쪽 칸은 -log(p) 곡선과 점 네 개, 오른쪽 칸은 같은 p에서
    (1 - p)^2 (MSE식)과 -log(p)를 나란히 놓는다.
    돌려주는 값: (p, -log p, (1-p)^2) 목록.
    """
    L = _krfont.label
    p = np.linspace(0.005, 1.0, 400)
    ce = -np.log(p)
    sq = (1.0 - p) ** 2

    rows = [(pi, float(-np.log(pi)), float((1.0 - pi) ** 2)) for pi in CE_PROBS]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    # 왼쪽: -log(p)
    ax = axes[0]
    ax.plot(p, ce, color=RED, linewidth=2.6)
    offsets = {0.9: (-0.30, 0.55), 0.5: (0.03, 0.45), 0.1: (0.05, 0.25), 0.01: (0.05, 0.0)}
    for pi, ce_i, _ in rows:
        ax.scatter([pi], [ce_i], s=80, color=RED, zorder=3,
                   edgecolor="white", linewidth=0.8)
        dx, dy = offsets[pi]
        ax.text(pi + dx, ce_i + dy,
                L(f"p={pi:g} → 손실 {ce_i:.2f}", f"p={pi:g} -> loss {ce_i:.2f}", ko),
                fontsize=10, color="#333333", va="center")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 5.6)
    ax.set_xlabel(L("정답 쪽에 준 확률 p", "probability given to the true class, p", ko),
                  fontsize=11)
    ax.set_ylabel(L("손실 = -log(p)", "loss = -log(p)", ko), fontsize=11)
    ax.set_title(L("① 정답 확률이 낮을수록 손실이 커진다",
                   "1) lower probability on the answer, larger loss", ko),
                 fontsize=12.5, pad=10)
    ax.grid(alpha=0.25)

    # 오른쪽: MSE식과 비교
    ax = axes[1]
    ax.plot(p, ce, color=RED, linewidth=2.6,
            label=L("교차 엔트로피 -log(p)", "cross-entropy -log(p)", ko))
    ax.plot(p, sq, color=BLUE, linewidth=2.6,
            label=L("제곱 오차 (1 - p)²", "squared error (1 - p)^2", ko))
    ax.axhline(1.0, color=BLUE, linestyle=":", linewidth=1.2)
    ax.text(0.40, 1.18, L("제곱 오차는 1을 넘지 못한다", "squared error never exceeds 1", ko),
            fontsize=10, color=BLUE)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 5.6)
    ax.set_xlabel(L("정답 쪽에 준 확률 p", "probability given to the true class, p", ko),
                  fontsize=11)
    ax.set_ylabel(L("손실", "loss", ko), fontsize=11)
    ax.set_title(L("② 확신하고 틀릴 때 두 손실의 차이",
                   "2) confidently wrong: the two losses differ", ko),
                 fontsize=12.5, pad=10)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle(
        L("분류의 손실 - 정답에 준 확률만 보고 -log를 씌운다",
          "Classification loss - take -log of the probability on the answer", ko),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return rows


# 역전파 그림의 계산 사슬:
#   x -(x w1)-> h -(x w2)-> z -(시그모이드)-> 통과 확률 p -> 손실 -log(p)
# 정답은 "통과"(1)다. w2를 음수로 두어 처음 예측이 틀린 쪽(p < 0.5)에서 시작한다.
BP_X = 2.0
BP_W1 = 0.5
BP_W2 = -1.0
BP_TARGET = 1.0
BP_NUDGE = 0.001


def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def backprop_numbers() -> dict[str, float]:
    """역전파 예제의 순전파 값, 역전파 값, 수치 확인 값을 계산한다."""

    def loss_of(w1: float, w2: float) -> float:
        h = BP_X * w1
        z = h * w2
        return -float(np.log(sigmoid(z)))   # 정답이 통과이므로 통과 확률에 -log

    # 순전파
    h = BP_X * BP_W1
    z = h * BP_W2
    p = sigmoid(z)
    loss = -float(np.log(p))

    # 역전파: 뒤에서 받은 값 x 이 칸에서 곱한 수
    # 시그모이드와 교차 엔트로피를 이어 붙이면, 손실이 z에 민감한 정도는 p - 정답 하나로 줄어든다.
    d_z = p - BP_TARGET
    d_w2 = d_z * h                # w2에는 h가 곱해져 들어갔다
    d_h = d_z * BP_W2             # h에는 w2가 곱해졌다
    d_w1 = d_h * BP_X             # w1에는 x가 곱해져 들어갔다

    # 수치 확인: 가중치를 조금 바꿔 손실이 실제로 얼마나 변하는지 잰다
    num_w1 = (loss_of(BP_W1 + BP_NUDGE, BP_W2) - loss) / BP_NUDGE
    num_w2 = (loss_of(BP_W1, BP_W2 + BP_NUDGE) - loss) / BP_NUDGE

    return {
        "h": h, "z": z, "p": p, "loss": loss,
        "d_z": d_z, "d_w2": d_w2, "d_h": d_h, "d_w1": d_w1,
        "num_w1": num_w1, "num_w2": num_w2,
    }


def figure_backprop(ko: bool, out_path: Path) -> dict[str, float]:
    """역전파 그림: 윗줄 순전파(값), 아랫줄 역전파(손실이 그 값에 얼마나 민감한가)."""
    L = _krfont.label
    n = backprop_numbers()

    fig, ax = plt.subplots(figsize=(12.6, 4.8))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 5.9)
    ax.axis("off")

    # 칸 네 개: x, h, z, (통과 확률과 손실)
    node_x = [0.3, 3.3, 6.3, 9.5]
    node_w = [1.8, 1.8, 1.8, 3.6]
    node_y = 3.4
    node_h = 1.0
    names = [
        L(f"입력 x\n{BP_X:g}", f"input x\n{BP_X:g}", ko),
        L(f"h = x × w1\n{n['h']:g}", f"h = x * w1\n{n['h']:g}", ko),
        L(f"z = h × w2\n{n['z']:g}", f"z = h * w2\n{n['z']:g}", ko),
        L(f"통과 확률 p = 시그모이드(z) = {n['p']:.3f}\n손실 = -log(p) = {n['loss']:.3f}",
          f"p = sigmoid(z) = {n['p']:.3f}\nloss = -log(p) = {n['loss']:.3f}", ko),
    ]
    colors = [GRAY, BLUE, BLUE, RED]
    for xi, wi, name, c in zip(node_x, node_w, names, colors):
        box(ax, xi, node_y, wi, node_h, name, c, fontsize=10.5)

    # 윗줄: 순전파 화살표와 곱한 가중치
    forward_labels = [
        L(f"× w1 ({BP_W1:g})", f"x w1 ({BP_W1:g})", ko),
        L(f"× w2 ({BP_W2:g})", f"x w2 ({BP_W2:g})", ko),
        L("시그모이드", "sigmoid", ko),
    ]
    for i, lab in enumerate(forward_labels):
        x1 = node_x[i] + node_w[i] + 0.08
        x2 = node_x[i + 1] - 0.08
        arrow(ax, x1, node_y + 0.75, x2, node_y + 0.75, color=BLUE, width=2.0)
        ax.text((x1 + x2) / 2, node_y + 1.05, lab, ha="center", fontsize=10, color=BLUE)

    ax.text(0.3, 5.35,
            L("① 순전파 - 왼쪽에서 오른쪽으로 값을 계산한다",
              "1) forward pass - compute values left to right", ko),
            fontsize=12.5, color=BLUE)

    # 아랫줄: 역전파 화살표와 곱셈
    backward = [
        (3, 2, L(f"p - 정답 = {n['d_z']:.3f}", f"p - target = {n['d_z']:.3f}", ko)),
        (2, 1, L(f"{n['d_z']:.3f} × w2 = {n['d_h']:.3f}",
                 f"{n['d_z']:.3f} * w2 = {n['d_h']:.3f}", ko)),
    ]
    for src, dst, lab in backward:
        x1 = node_x[src] - 0.08
        x2 = node_x[dst] + node_w[dst] + 0.08
        arrow(ax, x1, node_y + 0.25, x2, node_y + 0.25, color=RED, width=2.0)
        ax.text((x1 + x2) / 2, node_y - 0.25, lab, ha="center", fontsize=10.5, color=RED)

    # 가중치 기울기 상자 두 개
    grad_y = 1.0
    box(ax, 5.7, grad_y, 4.0, 1.0,
        L(f"w2의 기울기 = {n['d_z']:.3f} × h({n['h']:g}) = {n['d_w2']:.3f}",
          f"grad of w2 = {n['d_z']:.3f} * h({n['h']:g}) = {n['d_w2']:.3f}", ko),
        ORANGE, fontsize=10.5)
    box(ax, 1.0, grad_y, 4.0, 1.0,
        L(f"w1의 기울기 = {n['d_h']:.3f} × x({BP_X:g}) = {n['d_w1']:.3f}",
          f"grad of w1 = {n['d_h']:.3f} * x({BP_X:g}) = {n['d_w1']:.3f}", ko),
        ORANGE, fontsize=10.5)
    arrow(ax, 7.4, node_y - 0.05, 7.6, grad_y + 1.05, color=ORANGE, width=1.6)
    arrow(ax, 4.2, node_y - 0.05, 3.0, grad_y + 1.05, color=ORANGE, width=1.6)

    ax.text(0.3, 0.25,
            L("② 역전파 - 오른쪽에서 왼쪽으로, 뒤에서 받은 값 × 이 칸에서 곱한 수",
              "2) backward pass - right to left, (value from behind) x (number multiplied here)",
              ko),
            fontsize=12.5, color=RED)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return n


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 6장 개념 그림 생성 ===")

    rows = figure_cross_entropy(ko, RESULTS_DIR / "fig6-cross-entropy.png")
    print("저장: results/fig6-cross-entropy.png  (분류의 손실, 교차 엔트로피)")
    print("  정답 쪽 확률 p    교차 엔트로피 -log(p)    제곱 오차 (1-p)^2")
    for pi, ce_i, sq_i in rows:
        print(f"  {pi:>12g}    {ce_i:>20.3f}    {sq_i:>17.3f}")

    figure_one_step(ko, RESULTS_DIR / "fig6-2-one-step.png")
    print("저장: results/fig6-2-one-step.png  (경사하강법의 한 걸음)")

    n = figure_backprop(ko, RESULTS_DIR / "fig6-backprop.png")
    print("저장: results/fig6-backprop.png  (역전파)")
    print(f"  순전파: x = {BP_X:g}, w1 = {BP_W1:g}, w2 = {BP_W2:g}, 정답 = 통과({BP_TARGET:g})")
    print(f"          h = {n['h']:g}, z = {n['z']:g}, 통과 확률 p = {n['p']:.3f}, 손실 -log(p) = {n['loss']:.3f}")
    print(f"  역전파: 손실이 z에 민감한 정도 = p - 정답 = {n['d_z']:.3f}")
    print(f"          손실이 h에 민감한 정도 = {n['d_z']:.3f} x w2 = {n['d_h']:.3f}")
    print(f"          w2의 기울기 = {n['d_w2']:.3f}, w1의 기울기 = {n['d_w1']:.3f}")
    print(f"  수치 확인: 가중치를 {BP_NUDGE:g} 키웠을 때 (손실 변화 ÷ {BP_NUDGE:g})")
    print(f"          w1: {n['num_w1']:.3f}  (역전파 값 {n['d_w1']:.3f})")
    print(f"          w2: {n['num_w2']:.3f}  (역전파 값 {n['d_w2']:.3f})")

    diff, mse = mse_example()
    print("회귀의 손실 예시 (그림 없음): 평균제곱오차")
    print(f"  예시 5개의 차이(정답 - 예측): {np.array2string(diff, precision=1, floatmode='fixed')}")
    print(f"  차이를 제곱한 값: {np.array2string(diff ** 2, precision=2, floatmode='fixed')}")
    print(f"  제곱한 값의 평균(손실): {mse:.2f}")

    figure_epoch_batch(ko, RESULTS_DIR / "fig6-3-epoch-batch.png")
    print("저장: results/fig6-3-epoch-batch.png  (epoch와 batch의 관계)")

    figure_curve_shapes(ko, RESULTS_DIR / "fig6-4-curve-shapes.png")
    print("저장: results/fig6-4-curve-shapes.png  (학습 곡선의 모양과 눈금 함정)")
    print(f"  눈금 함정 왼쪽: {SCALE_TRAP_HIGH[0]:g} -> {SCALE_TRAP_HIGH[1]:g}"
          f" (차이 {SCALE_TRAP_HIGH[0] - SCALE_TRAP_HIGH[1]:g})")
    print(f"  눈금 함정 오른쪽: {SCALE_TRAP_LOW[0]:g} -> {SCALE_TRAP_LOW[1]:g}"
          f" (차이 {SCALE_TRAP_LOW[0] - SCALE_TRAP_LOW[1]:g})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")


if __name__ == "__main__":
    main()
