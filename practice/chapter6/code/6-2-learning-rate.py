"""실습 6.2 - 학습률 세 가지로 같은 분류 문제를 학습시켜 손실 곡선을 겹쳐 본다.

실습 6.1에서는 걸음 규칙만 봤다. 여기서는 실제 데이터에 그 규칙을 적용한다.

문제: 공부시간을 보고 학생 200명의 시험 통과 여부를 맞힌다(4장과 같은 종류의 분류).
      통과 확률 = 시그모이드(w x 입력 + b)   에서 w와 b를 경사하강법으로 찾는다.
손실: 교차 엔트로피 = 평균(-log(정답 쪽에 준 확률))

학습률만 세 가지로 바꾸고 나머지는 전부 같게 둔다.
  너무 작다 -> 200번을 돌아도 바닥에 닿지 못한다
  알맞다   -> 손실이 빠르게 내려가 바닥에 붙는다
  너무 크다 -> 최저점을 건너뛰며 손실이 처음보다 커지고 오르내린다

이 실습은 배치를 나누지 않는다. 한 에포크에서 200명을 한 번에 보고 가중치를 한 번 갱신한다.
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
LEARNING_RATES = [0.003, 0.3, 30.0]   # 너무 작다 / 알맞다 / 너무 크다
N_EPOCHS = 200                        # 전체 데이터를 몇 번 도는가
SHIFT_INPUT = True                    # True면 입력 = 공부시간 - 5, False면 공부시간 그대로
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_STUDENTS = 200
TRUE_W = 1.2      # 합성 데이터를 만들 때 쓴 통과 확률 = 시그모이드(1.2 x 공부시간 - 6.0)
TRUE_B = -6.0
SHIFT = 5.0       # SHIFT_INPUT이 True일 때 공부시간에서 빼는 값
START_W = 0.0
START_B = 0.0

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

COLORS = ["#4C78A8", "#54A24B", "#E45756"]
NAMES_KO = ["너무 작다", "알맞다", "너무 크다"]
NAMES_EN = ["too small", "just right", "too large"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """0과 1 사이의 확률로 바꾼다. tanh로 쓰면 z가 아주 커도 넘치지 않는다."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def make_synthetic_students() -> tuple[np.ndarray, np.ndarray]:
    """합성 데이터를 만든다. 공부시간이 길수록 통과 확률이 높다."""
    rng = np.random.default_rng(RANDOM_SEED)
    hours = rng.uniform(0.0, 10.0, N_STUDENTS)
    pass_prob = sigmoid(TRUE_W * hours + TRUE_B)
    passed = (rng.uniform(size=N_STUDENTS) < pass_prob).astype(float)
    return hours, passed


def to_input(hours: np.ndarray) -> np.ndarray:
    return hours - SHIFT if SHIFT_INPUT else hours


def cross_entropy(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """교차 엔트로피. 학생마다 정답 쪽에 준 확률에 -log를 씌워 평균 낸다.

    -log(시그모이드(z))를 그대로 계산하면 z가 클 때 log(0)이 나온다.
    log(1 + e^z) - y*z 는 같은 값을 넘침 없이 계산하는 식이다.
    """
    z = w * x + b
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def accuracy(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """통과 확률 0.5 이상을 통과로 판정했을 때 맞힌 비율."""
    return float(np.mean((sigmoid(w * x + b) >= 0.5) == (y == 1.0)))


def train(x: np.ndarray, y: np.ndarray, learning_rate: float) -> dict:
    """한 학습률로 N_EPOCHS번 학습하고 에포크마다 손실을 기록한다."""
    w, b = float(START_W), float(START_B)
    history = [cross_entropy(x, y, w, b)]

    for _ in range(N_EPOCHS):
        p = sigmoid(w * x + b)
        grad_w = float(np.mean((p - y) * x))   # w를 키우면 손실이 얼마나 커지는가
        grad_b = float(np.mean(p - y))         # b를 키우면 손실이 얼마나 커지는가

        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        history.append(cross_entropy(x, y, w, b))

    history = np.array(history, dtype=float)
    return {
        "learning_rate": learning_rate,
        "history": history,
        "final_w": w,
        "final_b": b,
        "final_loss": float(history[-1]),
        "peak_loss": float(history.max()),
        "n_rises": int(np.sum(np.diff(history) > 0)),   # 손실이 직전보다 커진 에포크 수
        "accuracy": accuracy(x, y, w, b),
    }


def best_possible_loss(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """이 데이터에서 w와 b 두 값으로 도달할 수 있는 가장 낮은 손실을 구한다.

    경사하강법으로 찾아가야 할 목적지를 미리 계산해 둔다. 계산에는 곡률까지 쓰는
    다른 방법(뉴턴법)을 쓴다. 학생이 알아야 할 내용은 아니다.
    """
    w, b = 0.0, 0.0
    for _ in range(50):
        p = sigmoid(w * x + b)
        grad = np.array([np.mean((p - y) * x), np.mean(p - y)])
        s = p * (1.0 - p)
        hess = np.array([[np.mean(s * x * x), np.mean(s * x)],
                         [np.mean(s * x), np.mean(s)]])
        step = np.linalg.solve(hess, grad)
        w, b = w - float(step[0]), b - float(step[1])
    return w, b, cross_entropy(x, y, w, b)


def save_curve_plot(ko: bool, runs: list[dict], floor: float,
                    hours: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    """왼쪽은 에포크별 손실, 오른쪽은 학습률마다 찾은 통과 확률 곡선이다."""
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # 왼쪽: 손실 곡선
    ax = axes[0]
    for run, color, ko_name, en_name in zip(runs, COLORS, NAMES_KO, NAMES_EN):
        history = run["history"]
        ax.plot(np.arange(len(history)), history, color=color, linewidth=2.2,
                label=L(f"학습률 {run['learning_rate']:g} - {ko_name}",
                        f"lr {run['learning_rate']:g} - {en_name}", ko))
    ax.axhline(floor, color="#79706E", linestyle="--", linewidth=1.6,
               label=L("도달할 수 있는 가장 낮은 손실", "lowest reachable loss", ko))
    ax.set_xlim(0, N_EPOCHS)
    ax.set_ylim(0, max(run["peak_loss"] for run in runs) * 1.08)
    ax.set_xlabel(L("에포크 (전체 데이터를 몇 번 돌았는가)",
                    "epoch (passes over the whole dataset)", ko), fontsize=11.5)
    ax.set_ylabel(L("손실 (교차 엔트로피)", "loss (cross-entropy)", ko), fontsize=11.5)
    ax.set_title(L("① 에포크별 손실", "1) loss per epoch", ko), fontsize=12.5, pad=10)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.grid(alpha=0.25)

    # 오른쪽: 학생 점과 통과 확률 곡선
    ax = axes[1]
    rng = np.random.default_rng(0)   # 점이 겹치지 않게 위아래로 조금 흩뜨린다(그림 전용)
    jitter = rng.uniform(-0.04, 0.04, len(y))
    ax.scatter(hours, y + jitter, s=16, color="#333333", alpha=0.45,
               label=L("학생 (1 = 통과, 0 = 탈락)", "students (1 = pass, 0 = fail)", ko))
    grid = np.linspace(0.0, 10.0, 300)
    for run, color, ko_name, en_name in zip(runs, COLORS, NAMES_KO, NAMES_EN):
        prob = sigmoid(run["final_w"] * to_input(grid) + run["final_b"])
        ax.plot(grid, prob, color=color, linewidth=2.4,
                label=L(f"학습률 {run['learning_rate']:g}이 찾은 곡선",
                        f"curve found with lr {run['learning_rate']:g}", ko))
    ax.axhline(0.5, color="#79706E", linestyle=":", linewidth=1.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel(L("공부시간", "study hours", ko), fontsize=11.5)
    ax.set_ylabel(L("통과 확률", "probability of passing", ko), fontsize=11.5)
    ax.set_title(L(f"② {N_EPOCHS}에포크 뒤 찾은 통과 확률 곡선",
                   f"2) probability curves after {N_EPOCHS} epochs", ko),
                 fontsize=12.5, pad=10)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), fontsize=9.5, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle(
        L("학습률 세 가지의 손실 곡선 (합성 데이터)",
          "Loss curves for three learning rates (synthetic data)", ko),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    hours, y = make_synthetic_students()
    x = to_input(hours)
    w_best, b_best, floor = best_possible_loss(x, y)

    runs = [train(x, y, lr) for lr in LEARNING_RATES]

    table = pd.DataFrame([
        {
            "학습률": run["learning_rate"],
            "평가": name,
            "처음 손실": round(float(run["history"][0]), 4),
            "마지막 손실": round(run["final_loss"], 4),
            "가장 높았던 손실": round(run["peak_loss"], 4),
            "손실이 오른 에포크 수": run["n_rises"],
            "찾은 w": round(run["final_w"], 4),
            "찾은 b": round(run["final_b"], 4),
            "정확도": round(run["accuracy"], 3),
        }
        for run, name in zip(runs, NAMES_KO)
    ])
    csv_path = RESULTS_DIR / "learning_rate_table.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_path = RESULTS_DIR / "learning_rate_curves.png"
    save_curve_plot(ko, runs, floor, hours, y, plot_path)

    # ------------------------------------------------------------------
    print("=== 실습 6.2 학습률 세 가지 비교 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 관측 자료 아님)")
    print(f"학생 {N_STUDENTS}명, 공부시간 0~10시간, "
          f"통과 확률 = 시그모이드({TRUE_W} x 공부시간 {TRUE_B:+g})로 통과·탈락을 뽑았다")
    print(f"통과한 학생: {int(y.sum())}명")
    if SHIFT_INPUT:
        print(f"입력 = 공부시간 - {SHIFT:g} (SHIFT_INPUT = True)")
    else:
        print("입력 = 공부시간 그대로 (SHIFT_INPUT = False)")
    print(f"출발점 w = {START_W}, b = {START_B}, 에포크 {N_EPOCHS}회")
    print("배치를 나누지 않는다. 에포크 1회 = 가중치 갱신 1번이다.")
    print()

    print("=== 목적지 ===")
    print(f"w와 b로 도달할 수 있는 가장 낮은 손실: {floor:.4f}")
    print(f"그때의 w = {w_best:.4f}, b = {b_best:.4f}, "
          f"정확도 = {accuracy(x, y, w_best, b_best):.3f}")
    print()

    print("=== 학습률별 결과 ===")
    print(table.to_string(index=False))
    print()

    print("=== 에포크별 손실 (일부만 뽑아 봄) ===")
    checkpoints = [0, 1, 2, 3, 5, 10, 20, 50, 100, N_EPOCHS]
    header = f"{'에포크':>6}" + "".join(f"{f'lr={lr:g}':>12}" for lr in LEARNING_RATES)
    print(header)
    for epoch in checkpoints:
        cells = "".join(f"{run['history'][epoch]:>12.4f}" for run in runs)
        print(f"{epoch:>6}{cells}")
    print()

    print("=== 저장된 결과 ===")
    print(f"손실 곡선: results/{plot_path.name}")
    print(f"비교표: results/{csv_path.name}")
    print()
    print("LEARNING_RATES, N_EPOCHS, SHIFT_INPUT을 바꿔 다시 실행하면 곡선 모양이 달라진다.")


if __name__ == "__main__":
    main()
