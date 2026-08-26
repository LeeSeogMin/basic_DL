"""실습 6.2 - 학습률 세 가지로 같은 문제를 학습시켜 손실 곡선을 겹쳐 본다.

실습 6.1에서는 걸음 규칙만 봤다. 여기서는 실제 데이터에 그 규칙을 적용한다.

문제: 점 200개를 지나는 직선 하나를 찾는다.
      예측 = w x 입력 + b   에서 w와 b를 경사하강법으로 찾는다.
손실: 평균제곱오차(MSE) = 평균((예측 - 정답)^2)

학습률만 세 가지로 바꾸고 나머지는 전부 같게 둔다.
  너무 작다 -> 200번을 돌아도 바닥에 닿지 못한다
  알맞다   -> 손실이 빠르게 내려가 바닥에 붙는다
  너무 크다 -> 최저점을 건너뛰며 손실이 커지고, 결국 값이 넘쳐 inf가 된다

이 실습은 배치를 나누지 않는다. 한 에포크에서 200개를 한 번에 보고 가중치를 한 번 갱신한다.
즉 에포크 200회 = 갱신 200번이다.

주의: 이 데이터는 실제 관측 자료가 아니라 컴퓨터로 만든 합성 데이터다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.
LEARNING_RATES = [0.0001, 0.02, 0.12]   # 너무 작다 / 알맞다 / 너무 크다
N_EPOCHS = 200                          # 전체 데이터를 몇 번 도는가
START_W = 0.0                           # 출발점 (기울기)
START_B = 0.0                           # 출발점 (절편)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_POINTS = 200
TRUE_W = 2.5
TRUE_B = 1.0
NOISE_SD = 2.0

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

COLORS = ["#4C78A8", "#54A24B", "#E45756"]


def make_synthetic_points() -> tuple[np.ndarray, np.ndarray]:
    """합성 데이터를 만든다. 정답 직선은 y = 2.5x + 1.0 이고 잡음을 섞었다."""
    rng = np.random.default_rng(RANDOM_SEED)
    x = rng.uniform(0.0, 10.0, N_POINTS)
    y = TRUE_W * x + TRUE_B + rng.normal(0.0, NOISE_SD, N_POINTS)
    return x, y


def mse(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """평균제곱오차. 예측과 정답의 차이를 제곱해 평균 낸 값이다."""
    error = w * x + b - y
    return float(np.mean(error ** 2))


def train(x: np.ndarray, y: np.ndarray, learning_rate: float) -> dict:
    """한 학습률로 N_EPOCHS번 학습하고 에포크마다 손실을 기록한다."""
    w, b = float(START_W), float(START_B)
    history = [mse(x, y, w, b)]
    blew_up_at = None

    with np.errstate(over="ignore", invalid="ignore"):
        for epoch in range(1, N_EPOCHS + 1):
            error = w * x + b - y
            grad_w = float(2.0 * np.mean(error * x))   # w를 키우면 손실이 얼마나 커지는가
            grad_b = float(2.0 * np.mean(error))       # b를 키우면 손실이 얼마나 커지는가

            w = w - learning_rate * grad_w
            b = b - learning_rate * grad_b

            current = mse(x, y, w, b)
            history.append(current)

            if not np.isfinite(current):
                blew_up_at = epoch
                break

    return {
        "learning_rate": learning_rate,
        "history": np.array(history, dtype=float),
        "final_w": w,
        "final_b": b,
        "final_loss": history[-1],
        "blew_up_at": blew_up_at,
    }


def best_possible_loss(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """이 데이터에서 직선 하나로 도달할 수 있는 가장 낮은 손실을 구한다.

    경사하강법으로 찾아가야 할 목적지가 어디인지 미리 계산해 둔다.
    """
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    w_best, b_best = float(coef[0]), float(coef[1])
    return w_best, b_best, mse(x, y, w_best, b_best)


def save_curve_plot(ko: bool, runs: list[dict], floor: float, out_path: Path) -> None:
    """손실 곡선을 두 칸으로 나눠 그린다.

    발산하는 곡선을 한 칸에 같이 그리면 세로 눈금이 300자리까지 늘어나
    나머지 두 곡선이 바닥에 붙은 직선으로 뭉개진다. 그래서 칸을 나눈다.
      왼쪽  손실이 실제로 줄어드는 구간만 확대한다. 발산 곡선은 위로 빠져나간다.
      오른쪽 발산 곡선 하나만 전체 범위로 그린다.
    """
    L = _krfont.label
    names_ko = ["너무 작다", "알맞다", "너무 크다"]
    names_en = ["too small", "just right", "too large"]

    top = 400.0   # 왼쪽 칸의 세로 눈금 위쪽 한계

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    # ------------------------------------------------------------------
    # 왼쪽: 손실이 실제로 줄어드는 구간
    ax = axes[0]
    for run, color, ko_name, en_name in zip(runs, COLORS, names_ko, names_en):
        history = run["history"]
        epochs = np.arange(len(history))
        drawable = np.isfinite(history) & (history > 0)
        # 이 칸의 눈금 밖으로 한참 벗어난 값까지 넘기면 matplotlib이 눈금을 만들다
        # 넘침 경고를 낸다. 화면 위쪽 조금까지만 잘라서 넘긴다.
        inside = drawable & (history <= top * 1.5)
        ax.plot(
            epochs[inside], history[inside],
            color=color, linewidth=2.4,
            label=L(f"학습률 {run['learning_rate']} - {ko_name}",
                    f"lr {run['learning_rate']} - {en_name}", ko),
        )

        # 위쪽 한계를 넘어 화면 밖으로 나가는 곡선에는 나간 지점을 적는다.
        above = np.where(drawable & (history > top))[0]
        if above.size:
            exit_epoch = int(above[0])
            ax.annotate(
                L(f"{exit_epoch}에포크에서 화면 위로 벗어난다",
                  f"leaves the top at epoch {exit_epoch}", ko),
                xy=(exit_epoch, top * 0.95),
                xytext=(N_EPOCHS * 0.42, top * 0.62),
                fontsize=10.5, color=color, ha="left",
                arrowprops=dict(arrowstyle="->", color=color, linewidth=1.4),
            )

    ax.axhline(floor, color="#79706E", linestyle="--", linewidth=1.6,
               label=L("직선 하나로 도달할 수 있는 가장 낮은 손실",
                       "lowest loss a straight line can reach", ko))
    ax.set_yscale("log")
    ax.set_xlim(0, N_EPOCHS)
    ax.set_ylim(floor * 0.6, top)
    ax.set_xlabel(L("에포크 (전체 데이터를 몇 번 돌았는가)",
                    "epoch (passes over the whole dataset)", ko), fontsize=11.5)
    ax.set_ylabel(L("손실 (로그 눈금)", "loss (log scale)", ko), fontsize=11.5)
    ax.set_title(L("① 손실이 줄어드는 구간만 확대",
                   "1) zoomed to the useful range", ko), fontsize=12.5, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=1, fontsize=10, frameon=False)
    ax.grid(alpha=0.25, which="both")

    # ------------------------------------------------------------------
    # 오른쪽: 발산 곡선 하나만 전체 범위로.
    # 손실이 10의 170제곱까지 커지므로 로그 눈금으로도 눈금표를 만들 수 없다.
    # 그래서 손실 자체가 아니라 "손실이 10의 몇 제곱인가"를 그린다.
    ax = axes[1]
    diverged = runs[-1]
    history = diverged["history"]
    epochs = np.arange(len(history))
    drawable = np.isfinite(history) & (history > 0)
    digits = np.log10(history[drawable])

    ax.plot(epochs[drawable], digits, color=COLORS[-1], linewidth=2.4)
    ax.axhline(np.log10(floor), color="#79706E", linestyle="--", linewidth=1.6)
    ax.text(N_EPOCHS * 0.5, np.log10(floor) + 2.0,
            L("도달했어야 할 손실", "the loss it should have reached", ko),
            ha="center", va="bottom", fontsize=10.5, color="#79706E")

    last_epoch = int(epochs[drawable][-1])
    last_digits = float(digits[-1])
    ax.scatter([last_epoch], [last_digits], s=130, marker="X",
               color=COLORS[-1], zorder=4, edgecolor="white", linewidth=0.9)
    if diverged["blew_up_at"] is not None:
        note = L(f"{diverged['blew_up_at']}에포크에서 숫자가 넘쳐 inf",
                 f"overflows to inf at epoch {diverged['blew_up_at']}", ko)
    else:
        note = L(f"{N_EPOCHS}에포크에서 손실이 {last_digits:.0f}자리",
                 f"{last_digits:.0f} digits at epoch {N_EPOCHS}", ko)
    # 주석은 곡선 오른쪽 아래 빈 자리에 둔다. 곡선 위에 겹치지 않는다.
    ax.annotate(note, xy=(last_epoch, last_digits),
                xytext=(N_EPOCHS * 0.99, last_digits * 0.42),
                fontsize=10.5, color=COLORS[-1], ha="right",
                arrowprops=dict(arrowstyle="->", color=COLORS[-1], linewidth=1.4))

    ax.set_xlim(0, N_EPOCHS)
    ax.set_ylim(-1.0, last_digits * 1.22)
    ax.set_xlabel(L("에포크", "epoch", ko), fontsize=11.5)
    ax.set_ylabel(L("손실이 10의 몇 제곱인가", "loss as a power of 10", ko), fontsize=11.5)
    ax.set_title(
        L(f"② 학습률 {diverged['learning_rate']} 하나만 전체 범위로",
          f"2) lr {diverged['learning_rate']} on its full range", ko),
        fontsize=12.5, pad=10,
    )
    ax.grid(alpha=0.25)

    fig.suptitle(
        L("학습률 세 가지의 손실 곡선 (합성 데이터)",
          "Loss curves for three learning rates (synthetic data)", ko),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def format_loss(value: float) -> str:
    """inf와 nan도 표에 그대로 적는다."""
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf"
    if abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    x, y = make_synthetic_points()
    w_best, b_best, floor = best_possible_loss(x, y)

    runs = [train(x, y, lr) for lr in LEARNING_RATES]

    names = ["너무 작다", "알맞다", "너무 크다"]
    table = pd.DataFrame([
        {
            "학습률": run["learning_rate"],
            "평가": name,
            "처음 손실": round(float(run["history"][0]), 4),
            "마지막 손실": format_loss(float(run["final_loss"])),
            "찾은 w": format_loss(float(run["final_w"])),
            "찾은 b": format_loss(float(run["final_b"])),
            "발산 시점(에포크)": "-" if run["blew_up_at"] is None else run["blew_up_at"],
        }
        for run, name in zip(runs, names)
    ])
    csv_path = RESULTS_DIR / "learning_rate_table.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_path = RESULTS_DIR / "learning_rate_curves.png"
    save_curve_plot(ko, runs, floor, plot_path)

    # ------------------------------------------------------------------
    print("=== 실습 6.2 학습률 세 가지 비교 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 관측 자료 아님)")
    print(f"점 {N_POINTS}개, 정답 직선 y = {TRUE_W}x + {TRUE_B} (잡음 표준편차 {NOISE_SD})")
    print(f"출발점 w = {START_W}, b = {START_B}, 에포크 {N_EPOCHS}회")
    print("배치를 나누지 않는다. 에포크 1회 = 가중치 갱신 1번이다.")
    print()

    print("=== 목적지 ===")
    print(f"직선 하나로 도달할 수 있는 가장 낮은 손실: {floor:.4f}")
    print(f"그때의 w = {w_best:.4f}, b = {b_best:.4f}")
    print()

    print("=== 학습률별 결과 ===")
    print(table.to_string(index=False))
    print()

    print("=== 에포크별 손실 (일부만 뽑아 봄) ===")
    checkpoints = [0, 1, 5, 20, 50, 100, N_EPOCHS]
    header = f"{'에포크':>6}" + "".join(f"{f'lr={lr}':>16}" for lr in LEARNING_RATES)
    print(header)
    for epoch in checkpoints:
        cells = ""
        for run in runs:
            history = run["history"]
            value = history[epoch] if epoch < len(history) else float("inf")
            cells += f"{format_loss(float(value)):>16}"
        print(f"{epoch:>6}{cells}")
    print()

    print("=== 저장된 결과 ===")
    print(f"손실 곡선: results/{plot_path.name}")
    print(f"비교표: results/{csv_path.name}")
    print()
    print("LEARNING_RATES와 N_EPOCHS를 바꿔 다시 실행하면 곡선 모양이 달라진다.")


if __name__ == "__main__":
    main()
