from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


def make_student_friendly_data() -> tuple[np.ndarray, np.ndarray]:
    """눈으로 이해하기 쉬운 2차원 분류 데이터를 만든다."""
    x, y = make_blobs(
        n_samples=160,
        centers=[(-1.8, -1.2), (1.6, 1.4)],
        cluster_std=[1.15, 1.25],
        random_state=RANDOM_SEED,
    )

    # 두 그룹이 조금 겹치도록 일부 점을 섞는다. 그래야 틀린 사례가 생긴다.
    rng = np.random.default_rng(RANDOM_SEED)
    flip_indices = rng.choice(np.arange(len(y)), size=12, replace=False)
    y[flip_indices] = 1 - y[flip_indices]
    return x, y


def save_decision_boundary(
    model: LogisticRegression,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    x_min, x_max = x_train[:, 0].min() - 1.0, x_train[:, 0].max() + 1.0
    y_min, y_max = x_train[:, 1].min() - 1.0, x_train[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, alpha=0.18, levels=[-0.5, 0.5, 1.5], colors=["#4C78A8", "#F58518"])
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="coolwarm", s=42, edgecolor="white", label="train")
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap="coolwarm", s=72, marker="x", label="test")
    plt.title("Decision boundary for a simple classifier")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = make_student_friendly_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_pred = baseline.predict(x_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    model_acc = accuracy_score(y_test, pred)
    matrix = confusion_matrix(y_test, pred)

    mistakes = np.where(pred != y_test)[0]
    mistake_rows = []
    for rank, test_index in enumerate(mistakes[:5], start=1):
        mistake_rows.append(
            {
                "rank": rank,
                "feature_1": round(float(x_test[test_index, 0]), 3),
                "feature_2": round(float(x_test[test_index, 1]), 3),
                "true_label": int(y_test[test_index]),
                "predicted_label": int(pred[test_index]),
            }
        )
    mistakes_df = pd.DataFrame(mistake_rows)
    mistakes_path = RESULTS_DIR / "misclassified_examples.csv"
    mistakes_df.to_csv(mistakes_path, index=False)

    boundary_path = RESULTS_DIR / "decision_boundary.png"
    save_decision_boundary(model, x_train, y_train, x_test, y_test, boundary_path)

    confusion_path = RESULTS_DIR / "confusion_matrix.png"
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["A group", "B group"])
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=160)
    plt.close()

    print("=== 4주차 머신러닝 첫걸음 ===")
    print(f"전체 데이터 수: {len(x)}")
    print(f"훈련 데이터 수: {len(x_train)}")
    print(f"테스트 데이터 수: {len(x_test)}")
    print()
    print("=== 기준선 모델 ===")
    print("전략: 테스트 데이터의 모든 점을 훈련 데이터에서 더 많은 그룹으로 예측")
    print(f"기준선 정확도: {baseline_acc:.3f}")
    print()
    print("=== 로지스틱 회귀 분류기 ===")
    print(f"테스트 정확도: {model_acc:.3f}")
    print("혼동행렬:")
    print(matrix)
    print()
    print("=== 틀린 사례 일부 ===")
    if mistakes_df.empty:
        print("틀린 사례가 없습니다.")
    else:
        print(mistakes_df.to_string(index=False))
    print()
    print("=== 저장된 결과 ===")
    print(f"판단 경계 그림: {boundary_path.relative_to(CHAPTER_DIR)}")
    print(f"혼동행렬 그림: {confusion_path.relative_to(CHAPTER_DIR)}")
    print(f"틀린 사례 CSV: {mistakes_path.relative_to(CHAPTER_DIR)}")
    print()
    print("=== 해석 주의 ===")
    print("정확도는 맞힌 비율이다. 어떤 방향으로 틀렸는지는 혼동행렬과 틀린 사례를 함께 봐야 한다.")


if __name__ == "__main__":
    main()

