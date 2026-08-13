"""6주차 실습: 학습률을 바꿔가며 손실 곡선 비교하기.

scikit-learn의 make_classification으로 합성 데이터를 만들고,
numpy로 로지스틱 회귀(단층 신경망)를 직접 구현한다.
학습률 세 가지(10.0, 0.1, 0.0001)로 각각 학습하여
손실 곡선을 하나의 그래프에 그린다.

산출물:
  - results/loss_curves.png  학습률별 손실 곡선 비교 그래프
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ── 상수 ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

N_SAMPLES = 200
N_FEATURES = 2
N_EPOCHS = 100

LEARNING_RATES = [10.0, 0.1, 0.0001]
LR_LABELS = ["lr=10.0 (too large)", "lr=0.1 (good)", "lr=0.0001 (too small)"]


# ── 도우미 함수 ────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    """시그모이드 함수. 오버플로를 방지하기 위해 클리핑한다."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """이진 교차 엔트로피 손실. log(0)을 막기 위해 클리핑한다."""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return float(loss)


def compute_accuracy(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """예측 확률을 0.5 기준으로 이진 분류한 뒤 정확도를 구한다."""
    y_pred_label = (y_pred_prob >= 0.5).astype(int)
    return float(np.mean(y_true == y_pred_label))


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    lr: float,
    n_epochs: int,
) -> tuple[np.ndarray, float, list[float]]:
    """numpy로 로지스틱 회귀를 경사하강법으로 학습한다.

    반환값:
        weights: (n_features + 1,) 배열. 마지막 원소가 편향(bias).
        final_loss: 마지막 epoch의 손실.
        loss_history: epoch별 손실 기록.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_samples, n_features = x_train.shape

    # 가중치 초기화 (편향 포함)
    w = rng.standard_normal(n_features) * 0.01
    b = 0.0

    loss_history: list[float] = []

    for epoch in range(n_epochs):
        # 순전파
        z = x_train @ w + b
        y_pred = sigmoid(z)

        # 손실 계산
        loss = binary_cross_entropy(y_train, y_pred)
        loss_history.append(loss)

        # NaN이 발생하면 학습을 조기 중단한다
        if np.isnan(loss) or np.isinf(loss):
            print(f"  [!] lr={lr}: epoch {epoch + 1}에서 손실이 NaN/Inf — 학습 중단")
            # 나머지 epoch를 NaN으로 채워 그래프에 끊김을 표시한다
            loss_history.extend([float("nan")] * (n_epochs - epoch - 1))
            break

        # 역전파 (기울기 계산)
        error = y_pred - y_train  # (n_samples,)
        dw = (x_train.T @ error) / n_samples  # (n_features,)
        db = np.mean(error)

        # 파라미터 갱신
        w -= lr * dw
        b -= lr * db

    weights = np.append(w, b)
    return weights, loss_history[-1] if loss_history else float("nan"), loss_history


# ── 메인 ──────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 합성 데이터 생성
    x, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.05,
        random_state=RANDOM_SEED,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y,
    )

    print("=== 6주차 학습률 실험 ===")
    print(f"전체 데이터 수: {N_SAMPLES}")
    print(f"훈련 데이터 수: {len(x_train)}")
    print(f"테스트 데이터 수: {len(x_test)}")
    print(f"특성 수: {N_FEATURES}")
    print(f"epoch 수: {N_EPOCHS}")
    print(f"학습률 목록: {LEARNING_RATES}")
    print()

    # 2. 학습률별 학습 실행
    all_histories: list[list[float]] = []
    summary_rows: list[dict] = []

    for lr, label in zip(LEARNING_RATES, LR_LABELS):
        print(f"--- {label} ---")
        weights, final_loss, history = train_logistic_regression(
            x_train, y_train, lr, N_EPOCHS,
        )
        all_histories.append(history)

        # 테스트 정확도 계산
        w, b = weights[:-1], weights[-1]
        y_test_pred = sigmoid(x_test @ w + b)
        test_acc = compute_accuracy(y_test, y_test_pred)

        # epoch별 손실 출력 (처음 5, 마지막 5만)
        valid_losses = [(i, h) for i, h in enumerate(history) if not np.isnan(h)]
        if len(valid_losses) <= 10:
            for i, h in valid_losses:
                print(f"  epoch {i + 1:3d}: loss = {h:.4f}")
        else:
            for i, h in valid_losses[:5]:
                print(f"  epoch {i + 1:3d}: loss = {h:.4f}")
            print(f"  ... (중간 {len(valid_losses) - 10}개 생략) ...")
            for i, h in valid_losses[-5:]:
                print(f"  epoch {i + 1:3d}: loss = {h:.4f}")

        print(f"  최종 손실: {final_loss:.4f}")
        print(f"  테스트 정확도: {test_acc:.3f}")
        print()

        summary_rows.append({
            "학습률": lr,
            "최종 손실": round(final_loss, 4),
            "테스트 정확도": round(test_acc, 3),
            "수렴 여부": "발산" if np.isnan(final_loss) or final_loss > 2.0 else (
                "매우 느림" if final_loss > 0.5 else "수렴"
            ),
        })

    # 3. 비교표 출력
    print("=== 학습률 비교표 ===")
    print(f"{'학습률':>10s} | {'최종 손실':>10s} | {'테스트 정확도':>12s} | {'수렴 여부'}")
    print("-" * 55)
    for row in summary_rows:
        loss_str = f"{row['최종 손실']:.4f}" if not np.isnan(row["최종 손실"]) else "NaN"
        print(
            f"{row['학습률']:>10.4f} | {loss_str:>10s} | {row['테스트 정확도']:>12.3f} | {row['수렴 여부']}"
        )
    print()

    # 4. 손실 곡선 그래프 저장
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#E45756", "#4C78A8", "#F58518"]
    linestyles = ["--", "-", ":"]

    for history, label, color, ls in zip(all_histories, LR_LABELS, colors, linestyles):
        epochs = list(range(1, len(history) + 1))
        ax.plot(epochs, history, label=label, color=color, linestyle=ls, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (Binary Cross-Entropy)", fontsize=12)
    ax.set_title("Learning Rate Comparison: Loss Curves", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0, top=3.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    loss_curves_path = RESULTS_DIR / "loss_curves.png"
    fig.savefig(loss_curves_path, dpi=160)
    plt.close(fig)

    # 5. 결과 안내
    print("=== 저장된 결과 ===")
    print(f"손실 곡선 그림: {loss_curves_path.relative_to(CHAPTER_DIR)}")
    print()
    print("=== 해석 안내 ===")
    print("학습률이 너무 크면 손실이 줄어들지 않거나 발산한다.")
    print("학습률이 너무 작으면 손실이 매우 천천히 줄어든다.")
    print("적절한 학습률을 쓰면 손실이 빠르게 줄어들어 수렴한다.")
    print("이 세 가지를 손실 곡선 그래프에서 직접 비교할 수 있다.")


if __name__ == "__main__":
    main()
