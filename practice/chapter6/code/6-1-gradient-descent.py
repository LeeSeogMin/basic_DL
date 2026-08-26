"""실습 6.1 - 경사하강법이 답을 찾아가는 과정을 눈으로 본다.

4장에서는 MY_SLOPE와 MY_INTERCEPT를 손으로 바꿔 가며 정확도를 올렸다.
여기서는 그 손조정을 기계가 대신한다. 기계가 쓰는 규칙은 한 줄이다.

    다음 w = 지금 w - 학습률 x 기울기

손실 곡선은 w 하나만 바꾸면 되는 가장 단순한 모양으로 골랐다.

    손실(w) = (w - 3)^2 + 2

이 곡선은 w = 3에서 가장 낮고, 그때 손실은 2다. 답을 우리는 이미 알고 있다.
알고 시작하는 이유는, 기계가 그 답을 어떻게 찾아가는지 걸음마다 확인하기 위해서다.

scikit-learn을 쓰지 않고 numpy로 직접 계산한다. 식이 그대로 보이도록 하기 위해서다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.
LEARNING_RATE = 0.1    # 한 걸음의 보폭을 정하는 값
START_W = -2.0         # 어디에서 출발할 것인가
N_STEPS = 15           # 몇 걸음 걸을 것인가
# ---------------------------------------------------------------------------

TRUE_BEST_W = 3.0      # 손실이 가장 낮아지는 w
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


def loss(w: float) -> float:
    """손실 곡선. w가 3에서 멀어질수록 값이 커진다."""
    return (w - TRUE_BEST_W) ** 2 + 2.0


def gradient(w: float) -> float:
    """기울기. 어느 쪽으로 가야 손실이 늘어나는지 알려주는 값이다.

    손실(w) = (w - 3)^2 + 2 를 w로 미분하면 2 * (w - 3) 이 된다.
    부호가 양수면 오른쪽으로 갈수록 손실이 커진다는 뜻이므로 왼쪽으로 가야 한다.
    """
    return 2.0 * (w - TRUE_BEST_W)


def walk_down(start_w: float, learning_rate: float, n_steps: int) -> pd.DataFrame:
    """한 걸음씩 내려가며 매 걸음의 값을 표로 모은다."""
    rows = []
    w = float(start_w)

    for step in range(n_steps + 1):
        current_loss = loss(w)
        current_grad = gradient(w)
        stride = learning_rate * current_grad     # 이번 걸음에서 움직이는 거리
        next_w = w - stride

        rows.append({
            "걸음": step,
            "w": w,
            "손실": current_loss,
            "기울기": current_grad,
            "이동거리": -stride,
            "다음w": next_w,
        })

        if not np.isfinite(next_w):
            print(f"[중단] {step}걸음에서 값이 너무 커져 계산을 멈춘다.")
            break
        w = next_w

    return pd.DataFrame(rows)


def save_path_plot(ko: bool, table: pd.DataFrame, out_path: Path) -> None:
    """왼쪽에는 곡선 위의 궤적, 오른쪽에는 걸음별 손실을 그린다."""
    L = _krfont.label
    ws = table["w"].to_numpy(dtype=float)
    losses = table["손실"].to_numpy(dtype=float)
    finite = np.isfinite(ws) & np.isfinite(losses)
    ws, losses = ws[finite], losses[finite]

    lo = min(float(ws.min()), TRUE_BEST_W) - 1.5
    hi = max(float(ws.max()), TRUE_BEST_W) + 1.5

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # 왼쪽: 손실 곡선 위를 걸어 내려가는 궤적
    ax = axes[0]
    grid = np.linspace(lo, hi, 400)
    ax.plot(grid, (grid - TRUE_BEST_W) ** 2 + 2.0,
            color="#4C78A8", linewidth=2.6,
            label=L("손실 곡선", "loss curve", ko))
    ax.plot(ws, losses, color="#F58518", linewidth=1.4, alpha=0.8, zorder=2)
    ax.scatter(ws, losses, s=70, color="#F58518", zorder=3,
               edgecolor="white", linewidth=0.8,
               label=L("한 걸음마다의 위치", "position at each step", ko))
    ax.scatter([ws[0]], [losses[0]], s=150, color="#E45756", zorder=4,
               edgecolor="white", linewidth=1.0,
               label=L("출발점", "start", ko))
    ax.scatter([TRUE_BEST_W], [loss(TRUE_BEST_W)], s=170, marker="*",
               color="#54A24B", zorder=4, edgecolor="white", linewidth=0.8,
               label=L("가장 낮은 곳", "lowest point", ko))

    ax.set_xlim(lo, hi)
    ax.set_xlabel(L("바꿀 수 있는 값 w", "the value w", ko), fontsize=11.5)
    ax.set_ylabel(L("손실", "loss", ko), fontsize=11.5)
    ax.set_title(L("손실 곡선 위를 내려가는 길",
                   "walking down the loss curve", ko), fontsize=13, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=9.5, frameon=False)
    ax.grid(alpha=0.25)

    # 오른쪽: 걸음이 늘어날수록 손실이 어떻게 변하는가
    ax = axes[1]
    steps = np.arange(len(losses))
    ax.plot(steps, losses, marker="o", markersize=5.5,
            color="#F58518", linewidth=2.0)
    ax.axhline(loss(TRUE_BEST_W), color="#54A24B", linestyle="--", linewidth=1.6,
               label=L("도달할 수 있는 가장 낮은 손실",
                       "lowest possible loss", ko))
    ax.set_xlabel(L("걸음 수", "step", ko), fontsize=11.5)
    ax.set_ylabel(L("손실", "loss", ko), fontsize=11.5)
    ax.set_title(L("걸음이 쌓이면서 손실이 내려간다",
                   "loss goes down as steps accumulate", ko), fontsize=13, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=1, fontsize=9.5, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle(
        L(f"경사하강법 (학습률 {LEARNING_RATE}, 출발 w = {START_W}, {N_STEPS}걸음)",
          f"Gradient descent (lr={LEARNING_RATE}, start w={START_W}, {N_STEPS} steps)", ko),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    table = walk_down(START_W, LEARNING_RATE, N_STEPS)

    csv_path = RESULTS_DIR / "gradient_descent_steps.csv"
    table.round(4).to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_path = RESULTS_DIR / "gradient_descent_path.png"
    save_path_plot(ko, table, plot_path)

    # ------------------------------------------------------------------
    print("=== 실습 6.1 경사하강법이 답을 찾아가는 과정 ===")
    _krfont.report(ko)
    print("손실 곡선: 손실(w) = (w - 3)^2 + 2")
    print("갱신 규칙: 다음 w = 지금 w - 학습률 x 기울기")
    print(f"학습률 {LEARNING_RATE}, 출발 w = {START_W}, {N_STEPS}걸음")
    print()

    print("=== 걸음마다의 기록 ===")
    print(f"{'걸음':>4}{'w':>10}{'손실':>12}{'기울기':>12}{'이동거리':>12}{'다음w':>10}")
    for row in table.itertuples(index=False):
        print(
            f"{row.걸음:>4}{row.w:>10.3f}{row.손실:>12.3f}"
            f"{row.기울기:>12.3f}{row.이동거리:>12.3f}{row.다음w:>10.3f}"
        )
    print()

    first = table.iloc[0]
    last = table.iloc[-1]
    print("=== 처음과 끝 비교 ===")
    print(f"출발: w = {first.w:.3f}, 손실 = {first.손실:.3f}")
    print(f"도착: w = {last.w:.3f}, 손실 = {last.손실:.3f}")
    print(f"가장 낮은 곳: w = {TRUE_BEST_W:.3f}, 손실 = {loss(TRUE_BEST_W):.3f}")
    print(f"아직 남은 거리: {abs(last.w - TRUE_BEST_W):.3f}")
    print()

    print("=== 저장된 결과 ===")
    print(f"궤적 그림: results/{plot_path.name}")
    print(f"걸음 기록: results/{csv_path.name}")
    print()
    print("LEARNING_RATE, START_W, N_STEPS를 바꿔 다시 실행하면 길이 달라진다.")


if __name__ == "__main__":
    main()
