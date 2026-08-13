"""5주차 실습: XOR 문제 — 단층 퍼셉트론 실패와 다층 퍼셉트론 성공 비교.

직선 하나로 나눌 수 없는 XOR 문제를 통해,
은닉층이 왜 필요한지 시각적으로 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier

# ── 설정 ──
RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

# XOR 데이터: 4개의 점
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([0, 1, 1, 0])


def make_grid(x_data: np.ndarray, margin: float = 0.5, n: int = 200) -> tuple:
    """판단 경계 시각화용 격자를 만든다."""
    x_min, x_max = x_data[:, 0].min() - margin, x_data[:, 0].max() + margin
    y_min, y_max = x_data[:, 1].min() - margin, x_data[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n),
        np.linspace(y_min, y_max, n),
    )
    return xx, yy


def train_single_layer(x_data: np.ndarray, y_data: np.ndarray) -> MLPClassifier:
    """단층 퍼셉트론(은닉층 없음)으로 학습한다.

    hidden_layer_sizes=() 이면 은닉층이 없으므로
    입력이 바로 출력으로 연결된다. 선형 판단 경계만 만들 수 있다.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(),
        activation="logistic",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    model.fit(x_data, y_data)
    return model


def train_mlp(x_data: np.ndarray, y_data: np.ndarray) -> MLPClassifier:
    """은닉층이 있는 다층 퍼셉트론으로 학습한다.

    hidden_layer_sizes=(8,) 이면 뉴런 8개짜리 은닉층이 1개 있다.
    비선형 판단 경계를 만들 수 있어 XOR 문제를 풀 수 있다.
    tanh 활성화 함수를 사용하면 XOR처럼 작은 데이터에서 수렴이 안정적이다.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="tanh",
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    model.fit(x_data, y_data)
    return model


def save_comparison_plot(
    single: MLPClassifier,
    mlp: MLPClassifier,
    x_data: np.ndarray,
    y_data: np.ndarray,
    output_path: Path,
) -> None:
    """단층과 다층 퍼셉트론의 판단 경계를 나란히 비교하는 그림을 저장한다."""
    xx, yy = make_grid(x_data)
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = [single, mlp]
    titles = [
        "No hidden layer (single-layer perceptron)",
        "One hidden layer with 8 neurons (MLP)",
    ]

    for ax, model, title in zip(axes, models, titles):
        zz = model.predict(grid).reshape(xx.shape)
        ax.contourf(
            xx, yy, zz,
            alpha=0.25,
            levels=[-0.5, 0.5, 1.5],
            colors=["#4C78A8", "#F58518"],
        )
        # XOR 데이터 점 표시
        for label, marker, color in [(0, "o", "#4C78A8"), (1, "s", "#F58518")]:
            mask = y_data == label
            ax.scatter(
                x_data[mask, 0],
                x_data[mask, 1],
                c=color,
                marker=marker,
                s=180,
                edgecolor="black",
                linewidth=1.5,
                label=f"class {label}",
                zorder=5,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(loc="upper right")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

    fig.suptitle("XOR problem: single-layer vs multi-layer perceptron", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 라이브러리 버전 출력 ──
    print("=== 환경 정보 ===")
    print(f"Python numpy: {np.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"random seed: {RANDOM_SEED}")
    print()

    # ── XOR 데이터 ──
    print("=== XOR 데이터 ===")
    print("  x1  x2  |  정답")
    print("  --------+------")
    for xi, yi in zip(X, y):
        print(f"   {int(xi[0])}   {int(xi[1])}  |   {yi}")
    print()

    # ── 단층 퍼셉트론 ──
    single = train_single_layer(X, y)
    single_pred = single.predict(X)
    single_acc = np.mean(single_pred == y)

    print("=== 단층 퍼셉트론 (은닉층 없음) ===")
    print(f"hidden_layer_sizes: {single.hidden_layer_sizes}")
    print(f"activation: {single.activation}")
    print(f"예측: {single_pred.tolist()}")
    print(f"정답: {y.tolist()}")
    print(f"정확도: {single_acc:.3f}")
    if single_acc < 1.0:
        wrong = np.where(single_pred != y)[0]
        print(f"틀린 점: {len(wrong)}개")
        for idx in wrong:
            print(f"  ({int(X[idx, 0])}, {int(X[idx, 1])}) -> 정답 {y[idx]}, 예측 {single_pred[idx]}")
    print()

    # ── 다층 퍼셉트론 ──
    mlp = train_mlp(X, y)
    mlp_pred = mlp.predict(X)
    mlp_acc = np.mean(mlp_pred == y)

    print("=== 다층 퍼셉트론 (은닉층 8개 뉴런) ===")
    print(f"hidden_layer_sizes: {mlp.hidden_layer_sizes}")
    print(f"activation: {mlp.activation}")
    print(f"예측: {mlp_pred.tolist()}")
    print(f"정답: {y.tolist()}")
    print(f"정확도: {mlp_acc:.3f}")
    if mlp_acc < 1.0:
        wrong = np.where(mlp_pred != y)[0]
        print(f"틀린 점: {len(wrong)}개")
        for idx in wrong:
            print(f"  ({int(X[idx, 0])}, {int(X[idx, 1])}) -> 정답 {y[idx]}, 예측 {mlp_pred[idx]}")
    else:
        print("모든 XOR 데이터를 맞혔다.")
    print()

    # ── 비교 요약 ──
    print("=== 비교 ===")
    print(f"단층 퍼셉트론 정확도: {single_acc:.3f}")
    print(f"다층 퍼셉트론 정확도: {mlp_acc:.3f}")
    if single_acc < mlp_acc:
        print("은닉층을 추가하면 XOR 문제를 풀 수 있다.")
    print()

    # ── 판단 경계 그림 저장 ──
    boundary_path = RESULTS_DIR / "xor_decision_boundary.png"
    save_comparison_plot(single, mlp, X, y, boundary_path)

    print("=== 저장된 결과 ===")
    print(f"판단 경계 그림: {boundary_path.relative_to(CHAPTER_DIR)}")
    print()
    print("=== 해석 ===")
    print("단층 퍼셉트론은 직선 하나로 판단 경계를 만든다.")
    print("XOR 데이터는 직선 하나로 나눌 수 없으므로 단층 퍼셉트론은 실패한다.")
    print("은닉층을 추가하면 비선형 판단 경계를 만들 수 있어 XOR 문제를 풀 수 있다.")


if __name__ == "__main__":
    main()
