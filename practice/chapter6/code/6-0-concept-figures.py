"""6장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig6-1  손실이 무엇인가 - 예측과 정답의 차이를 하나의 숫자로 모으는 과정
  fig6-2  경사하강법의 한 걸음 - 기울기를 보고 내려갈 방향과 보폭을 정한다
  fig6-3  epoch와 batch - 전체를 몇 번 도는가, 한 번에 몇 개씩 보는가

이 그림들은 실험 결과가 아니라 개념 설명용 도식이다.
그림 6-1에 찍히는 숫자는 아래 EXAMPLE_X, EXAMPLE_Y로 직접 계산한 값이다.
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

# 그림 6-1에서 쓰는 작은 예시 (점 5개). 개념 설명용으로 손으로 고른 값이다.
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


def figure_what_is_loss(ko: bool, out_path: Path) -> tuple[np.ndarray, float]:
    """그림 6-1: 차이를 모아 하나의 숫자로 만든다."""
    L = _krfont.label
    pred = EXAMPLE_SLOPE * EXAMPLE_X + EXAMPLE_INTERCEPT
    diff = EXAMPLE_Y - pred
    squared = diff ** 2
    mse = float(squared.mean())

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    # 왼쪽: 점, 선, 그리고 둘 사이의 차이
    ax = axes[0]
    grid = np.linspace(0.4, 5.6, 50)
    ax.plot(grid, EXAMPLE_SLOPE * grid + EXAMPLE_INTERCEPT,
            color=GREEN, linewidth=2.6,
            label=L("모델의 예측", "model prediction", ko))
    ax.scatter(EXAMPLE_X, EXAMPLE_Y, s=90, color=BLUE, zorder=3,
               edgecolor="white", linewidth=0.8,
               label=L("정답", "true answer", ko))
    for xi, yi, pi, di in zip(EXAMPLE_X, EXAMPLE_Y, pred, diff):
        ax.plot([xi, xi], [pi, yi], color=RED, linewidth=2.2, zorder=2)
        # 차이 숫자는 점 쪽에 붙여 둔다. 가운데에 두면 초록 선과 겹친다.
        ax.text(xi + 0.18, pi + 0.75 * di, f"{di:+.1f}",
                fontsize=10, color=RED, va="center")

    ax.set_xlim(0.3, 5.9)
    ax.set_ylim(2.0, 11.0)
    ax.set_xlabel(L("입력", "input", ko), fontsize=11)
    ax.set_ylabel(L("출력", "output", ko), fontsize=11)
    ax.set_title(L("① 예측과 정답의 차이를 잰다",
                   "1) measure each gap", ko), fontsize=12.5, pad=10)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    ax.grid(alpha=0.25)

    # 오른쪽: 차이를 제곱해 쌓고, 평균 하나로 줄인다
    ax = axes[1]
    positions = np.arange(1, len(squared) + 1)
    ax.bar(positions, squared, width=0.55, color=RED, alpha=0.85)
    for p, s in zip(positions, squared):
        ax.text(p, s + 0.03, f"{s:.2f}", ha="center", fontsize=9.5, color="#333333")

    ax.set_xlim(0.3, 5.9)
    ax.set_ylim(0, 1.45)
    ax.set_xticks(positions)
    ax.set_xlabel(L("점 번호", "point index", ko), fontsize=11)
    ax.set_ylabel(L("차이를 제곱한 값", "squared gap", ko), fontsize=11)
    ax.set_title(L("② 제곱해서 더하고 개수로 나눈다",
                   "2) square, sum, divide by count", ko), fontsize=12.5, pad=10)
    ax.grid(alpha=0.25, axis="y")

    ax.text(
        3.1, 1.22,
        L(f"손실 = {squared.sum():.2f} ÷ 5 = {mse:.2f}",
          f"loss = {squared.sum():.2f} / 5 = {mse:.2f}", ko),
        ha="center", fontsize=13, color="white",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=BLUE, edgecolor="none"),
    )

    fig.suptitle(
        L("손실 - 여러 개의 틀린 정도를 숫자 하나로 모은다",
          "Loss - many gaps collapsed into a single number", ko),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return diff, mse


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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 6장 개념 그림 생성 ===")

    diff, mse = figure_what_is_loss(ko, RESULTS_DIR / "fig6-1-what-is-loss.png")
    print("저장: results/fig6-1-what-is-loss.png  (손실을 하나의 숫자로 모으는 과정)")
    print(f"  예시 5개의 차이: {np.array2string(diff, precision=1, floatmode='fixed')}")
    print(f"  차이를 제곱해 평균 낸 값(손실): {mse:.2f}")

    figure_one_step(ko, RESULTS_DIR / "fig6-2-one-step.png")
    print("저장: results/fig6-2-one-step.png  (경사하강법의 한 걸음)")

    figure_epoch_batch(ko, RESULTS_DIR / "fig6-3-epoch-batch.png")
    print("저장: results/fig6-3-epoch-batch.png  (epoch와 batch의 관계)")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")


if __name__ == "__main__":
    main()
