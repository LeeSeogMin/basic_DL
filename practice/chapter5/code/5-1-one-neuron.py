"""실습 5.1 - 뉴런 하나가 하는 계산을 눈으로 본다.

4장에서 로지스틱 회귀가 찾아 준 경계선을 이번에는 우리가 직접 만든다.
쓰는 데이터는 4장과 똑같은 학생 자료다. 점 하나가 학생 한 명이고,
가로축은 하루 공부 시간, 세로축은 하루 수면 시간이다.

뉴런 하나는 이렇게 계산한다.

    z = W1 * 공부시간 + W2 * 수면시간 + B
    출력 = 활성화함수(z)

z 가 0보다 크면 "통과"로 판정한다. 그래서 z = 0 인 자리가 경계선이 된다.
아래 W1, W2, B 세 값을 바꾸면 그 선이 움직인다.

주의: 이 데이터는 실제 학생 자료가 아니라 컴퓨터로 만든 합성 데이터다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _krfont

# ---------------------------------------------------------------------------
# 여기 세 값을 바꿔가며 실습한다.
#   W1  공부 시간에 붙는 가중치
#   W2  수면 시간에 붙는 가중치
#   B   편향
# 경계선 식:  수면시간 = -(W1/W2) * 공부시간 - B/W2
W1 = 0.5
W2 = 1.0
B = -8.0
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def make_synthetic_students(n_samples: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """합성 학생 데이터를 만든다. 4장 실습 4.1과 같은 데이터다."""
    rng = np.random.default_rng(RANDOM_SEED)
    study = rng.uniform(0.0, 10.0, n_samples)   # 하루 공부 시간
    sleep = rng.uniform(3.0, 9.0, n_samples)    # 하루 수면 시간

    hidden_score = 0.8 * study + 0.6 * sleep + rng.normal(0.0, 1.0, n_samples)
    passed = (hidden_score > 7.5).astype(int)

    x = np.column_stack([study, sleep])
    return x, passed


def neuron_sum(x: np.ndarray, w1: float, w2: float, b: float) -> np.ndarray:
    """뉴런의 가중합 z 를 구한다."""
    return w1 * x[:, 0] + w2 * x[:, 1] + b


def step(z):
    """계단 함수. 0보다 크면 1, 아니면 0."""
    return (z > 0).astype(float)


def sigmoid(z):
    """시그모이드 함수. 어떤 값이 들어와도 0과 1 사이로 눌러 준다."""
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    """ReLU 함수. 음수는 0으로 자르고, 양수는 그대로 통과시킨다."""
    return np.maximum(z, 0.0)


def boundary_line(w1: float, w2: float, b: float, grid_x: np.ndarray) -> np.ndarray:
    """z = 0 인 자리를 수면시간 = 기울기 * 공부시간 + 절편 형태로 바꾼다."""
    return -(w1 / w2) * grid_x - b / w2


def scatter_students(ax, x: np.ndarray, y: np.ndarray, ko: bool) -> None:
    L = _krfont.label
    for value, color, ko_name, en_name in [
        (0, BLUE, "미통과", "not passed"),
        (1, ORANGE, "통과", "passed"),
    ]:
        mask = y == value
        ax.scatter(
            x[mask, 0], x[mask, 1],
            c=color, s=34, edgecolor="white", linewidth=0.5,
            label=L(ko_name, en_name, ko),
        )


def save_neuron_plot(
    ko: bool, x: np.ndarray, y: np.ndarray, accuracy: float, out_path: Path
) -> None:
    """뉴런 하나의 경계선, 가중치를 바꿨을 때, 편향을 바꿨을 때를 나란히 그린다."""
    L = _krfont.label
    grid_x = np.linspace(0.0, 10.0, 200)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.2))

    # 1번 칸: 지금 값으로 그은 선
    ax = axes[0]
    scatter_students(ax, x, y, ko)
    ax.plot(grid_x, boundary_line(W1, W2, B, grid_x),
            color=GREEN, linewidth=2.8,
            label=L("지금 내 뉴런의 선", "my neuron now", ko))
    ax.set_title(
        L(f"지금 값: W1={W1}, W2={W2}, B={B}\n정확도 {accuracy:.3f}",
          f"now: W1={W1}, W2={W2}, B={B}\naccuracy {accuracy:.3f}", ko),
        fontsize=12, pad=10,
    )

    # 2번 칸: W1만 바꾼다 -> 선의 기울기가 바뀐다
    ax = axes[1]
    scatter_students(ax, x, y, ko)
    for w1_try, color in [(0.5, GRAY), (1.0, GREEN), (1.5, RED)]:
        ax.plot(grid_x, boundary_line(w1_try, W2, B, grid_x),
                color=color, linewidth=2.4,
                label=f"W1 = {w1_try}")
    ax.set_title(
        L(f"가중치 W1만 바꿈 (W2={W2}, B={B} 고정)\n선의 기울기가 바뀐다",
          f"changing W1 only (W2={W2}, B={B})\nthe slope changes", ko),
        fontsize=12, pad=10,
    )

    # 3번 칸: B만 바꾼다 -> 선이 평행하게 옮겨진다
    ax = axes[2]
    scatter_students(ax, x, y, ko)
    for b_try, color in [(-6.0, GRAY), (-8.0, GREEN), (-10.0, RED)]:
        ax.plot(grid_x, boundary_line(W1, W2, b_try, grid_x),
                color=color, linewidth=2.4,
                label=f"B = {b_try}")
    ax.set_title(
        L(f"편향 B만 바꿈 (W1={W1}, W2={W2} 고정)\n선이 기울기 그대로 옮겨진다",
          f"changing B only (W1={W1}, W2={W2})\nthe line shifts, slope unchanged", ko),
        fontsize=12, pad=10,
    )

    for ax in axes:
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(2.5, 9.5)
        ax.set_xlabel(L("하루 공부 시간(시간)", "study hours per day", ko), fontsize=11)
        ax.set_ylabel(L("하루 수면 시간(시간)", "sleep hours per day", ko), fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                  ncol=3, fontsize=9.5, frameon=False)

    fig.suptitle(
        L("뉴런 하나가 긋는 선 (합성 데이터)",
          "The line drawn by a single neuron (synthetic data)", ko),
        fontsize=14, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_activation_plot(ko: bool, out_path: Path) -> None:
    """계단, 시그모이드, ReLU 세 활성화 함수를 나란히 그린다."""
    L = _krfont.label
    z = np.linspace(-6.0, 6.0, 400)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))

    # 세 함수는 출력 범위가 달라서 세로축을 따로 잡는다.
    # 하나로 맞추면 계단과 시그모이드가 바닥에 눌려 모양이 안 보인다.
    panels = [
        (axes[0], step(z), BLUE, (-0.35, 1.35),
         L("계단 함수", "step", ko),
         L("0 아니면 1. 딱 잘라 판정하지만\n중간이 없어 학습에 쓰기 어렵다",
           "0 or 1. A hard verdict, but\nno middle ground to learn from", ko)),
        (axes[1], sigmoid(z), ORANGE, (-0.35, 1.35),
         L("시그모이드", "sigmoid", ko),
         L("0과 1 사이로 눌러 준다.\n확률처럼 읽을 수 있다",
           "squashes into 0 to 1.\nreads like a probability", ko)),
        (axes[2], relu(z), GREEN, (-1.5, 6.3),
         L("ReLU", "ReLU", ko),
         L("음수는 0으로 자르고\n양수는 그대로 통과시킨다",
           "clips negatives to 0,\npasses positives through", ko)),
    ]

    for ax, values, color, ylim, title, caption in panels:
        ax.plot(z, values, color=color, linewidth=2.8)
        ax.axhline(0.0, color=GRAY, linewidth=0.8)
        ax.axvline(0.0, color=GRAY, linewidth=0.8)
        ax.set_xlim(-6.0, 6.0)
        ax.set_ylim(*ylim)
        ax.set_xlabel("z", fontsize=11)
        ax.set_ylabel(L("출력", "output", ko), fontsize=11)
        ax.set_title(title, fontsize=13, pad=10)
        ax.text(0.5, -0.30, caption, ha="center", va="top",
                transform=ax.transAxes,
                fontsize=9.5, color="#333333", linespacing=1.5)
        ax.grid(alpha=0.25)

    fig.suptitle(
        L("활성화 함수 세 가지 — 가중합 z 를 무엇으로 바꾸는가",
          "Three activation functions - what they do to the weighted sum z", ko),
        fontsize=14, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    x, y = make_synthetic_students()
    z = neuron_sum(x, W1, W2, B)
    prediction = (z > 0).astype(int)
    accuracy = float((prediction == y).mean())

    plot_path = RESULTS_DIR / "one_neuron.png"
    save_neuron_plot(ko, x, y, accuracy, plot_path)
    activation_path = RESULTS_DIR / "activation_functions.png"
    save_activation_plot(ko, activation_path)

    # ------------------------------------------------------------------
    print("=== 실습 5.1 뉴런 하나가 하는 계산 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 학생 자료 아님), 4장 실습 4.1과 같은 데이터")
    print(f"학생 {len(x)}명 (통과 {int(y.sum())}명, 미통과 {int(len(y) - y.sum())}명)")
    print()

    print("=== 내 뉴런 ===")
    print(f"W1 = {W1}  (공부 시간에 붙는 가중치)")
    print(f"W2 = {W2}  (수면 시간에 붙는 가중치)")
    print(f"B  = {B}   (편향)")
    print(f"z = {W1} x 공부시간 + {W2} x 수면시간 + ({B})")
    print(f"경계선: 수면시간 = {-(W1 / W2):.2f} x 공부시간 + {-B / W2:.2f}")
    print(f"전체 정확도(z > 0 이면 통과로 판정): {accuracy:.3f}")
    print()

    print("=== 학생 5명의 계산 과정 ===")
    print(f"{'공부':>6}{'수면':>6}{'z':>8}{'시그모이드':>11}{'계단':>6}{'ReLU':>8}{'판정':>7}{'실제':>7}")
    for i in range(5):
        study, sleep = x[i]
        zi = z[i]
        verdict = "통과" if prediction[i] == 1 else "미통과"
        truth = "통과" if y[i] == 1 else "미통과"
        print(
            f"{study:>6.2f}{sleep:>6.2f}{zi:>8.2f}"
            f"{sigmoid(zi):>11.3f}{step(zi):>6.0f}{relu(zi):>8.2f}"
            f"{verdict:>7}{truth:>7}"
        )
    print()

    print("=== 가중치와 편향이 선을 어떻게 움직이는가 ===")
    print(f"{'설정':<22}{'기울기':>9}{'y절편':>9}{'정확도':>9}")
    settings = [
        (f"지금 값 W1={W1}", W1, W2, B),
        ("W1을 1.0으로", 1.0, W2, B),
        ("W1을 1.5로", 1.5, W2, B),
        ("B를 -6.0으로", W1, W2, -6.0),
        ("B를 -10.0으로", W1, W2, -10.0),
    ]
    for name, w1_try, w2_try, b_try in settings:
        pred_try = (neuron_sum(x, w1_try, w2_try, b_try) > 0).astype(int)
        acc_try = float((pred_try == y).mean())
        print(
            f"{name:<22}{-(w1_try / w2_try):>9.2f}"
            f"{-b_try / w2_try:>9.2f}{acc_try:>9.3f}"
        )
    print()

    print("=== 저장된 결과 ===")
    print(f"뉴런의 선: results/{plot_path.name}")
    print(f"활성화 함수: results/{activation_path.name}")
    print()
    print("W1, W2, B를 바꿔 다시 실행하면 선이 움직이고 정확도가 달라진다.")


if __name__ == "__main__":
    main()
