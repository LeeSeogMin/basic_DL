"""7주차 실습: 과적합 모델 vs 적절한 모델 비교.

scikit-learn의 make_classification으로 합성 데이터(100개)를 만들고,
결정 트리의 max_depth를 바꾸며 과적합을 관찰한다.

산출물:
  - 과적합 모델과 적절한 모델의 훈련/테스트 정확도 비교
  - max_depth별 훈련/테스트 정확도 그래프 (complexity_vs_accuracy.png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ── 설정 ──────────────────────────────────────────────
RANDOM_SEED = 42
N_SAMPLES = 100
TEST_SIZE = 0.3
MAX_DEPTH_RANGE = range(1, 21)

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


def make_small_dataset(
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """적은 수의 합성 분류 데이터를 만든다.

    데이터가 적으면 모델이 훈련 데이터를 외우기 쉬워 과적합이 잘 드러난다.
    """
    x, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=5,
        n_redundant=5,
        n_classes=2,
        random_state=seed,
    )
    return x, y


def train_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_depth: int | None,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """결정 트리를 학습하고 훈련/테스트 정확도를 반환한다."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    model.fit(x_train, y_train)
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    return train_acc, test_acc


def save_complexity_graph(
    depths: list[int],
    train_accs: list[float],
    test_accs: list[float],
    output_path: Path,
) -> None:
    """모델 복잡도(max_depth)별 훈련/테스트 정확도 그래프를 저장한다."""
    plt.figure(figsize=(9, 5))
    plt.plot(depths, train_accs, "o-", label="Train accuracy", color="#E45756")
    plt.plot(depths, test_accs, "s-", label="Test accuracy", color="#4C78A8")

    plt.xlabel("max_depth (model complexity)")
    plt.ylabel("Accuracy")
    plt.title("Model complexity vs accuracy: overfitting demo")
    plt.xticks(depths)
    plt.ylim(0.4, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 생성 ─────────────────────────────────
    x, y = make_small_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )

    print("=== 7주차 과적합과 일반화 ===")
    print(f"전체 데이터 수: {len(x)}")
    print(f"훈련 데이터 수: {len(x_train)}")
    print(f"테스트 데이터 수: {len(x_test)}")
    print(f"특성 수: {x.shape[1]}")
    print()

    # 2. 과적합 모델 vs 적절한 모델 ────────────────────
    overfit_train, overfit_test = train_and_evaluate(
        x_train, y_train, x_test, y_test, max_depth=None,
    )
    good_train, good_test = train_and_evaluate(
        x_train, y_train, x_test, y_test, max_depth=3,
    )

    print("=== 과적합 모델 (max_depth=None, 제한 없음) ===")
    print(f"  훈련 정확도: {overfit_train:.3f}")
    print(f"  테스트 정확도: {overfit_test:.3f}")
    print(f"  차이 (훈련 - 테스트): {overfit_train - overfit_test:.3f}")
    print()
    print("=== 적절한 모델 (max_depth=3) ===")
    print(f"  훈련 정확도: {good_train:.3f}")
    print(f"  테스트 정확도: {good_test:.3f}")
    print(f"  차이 (훈련 - 테스트): {good_train - good_test:.3f}")
    print()

    # 3. 복잡도별 훈련/테스트 정확도 ────────────────────
    depths = list(MAX_DEPTH_RANGE)
    train_accs: list[float] = []
    test_accs: list[float] = []

    for d in depths:
        tr, te = train_and_evaluate(x_train, y_train, x_test, y_test, max_depth=d)
        train_accs.append(tr)
        test_accs.append(te)

    print("=== max_depth별 정확도 ===")
    print(f"{'depth':>5}  {'train':>8}  {'test':>8}  {'gap':>8}")
    for d, tr, te in zip(depths, train_accs, test_accs):
        print(f"{d:>5}  {tr:>8.3f}  {te:>8.3f}  {tr - te:>8.3f}")
    print()

    # 4. 그래프 저장 ──────────────────────────────────
    graph_path = RESULTS_DIR / "complexity_vs_accuracy.png"
    save_complexity_graph(depths, train_accs, test_accs, graph_path)

    print("=== 저장된 결과 ===")
    print(f"복잡도-정확도 그래프: {graph_path.relative_to(CHAPTER_DIR)}")
    print()

    # 5. 과적합 진단 요약 ──────────────────────────────
    best_test_depth = depths[int(np.argmax(test_accs))]
    best_test_acc = max(test_accs)

    print("=== 과적합 진단 요약 ===")
    print(f"테스트 정확도가 가장 높은 max_depth: {best_test_depth}")
    print(f"그때의 테스트 정확도: {best_test_acc:.3f}")
    print(f"max_depth=None일 때 훈련 정확도: {overfit_train:.3f}")
    print(f"max_depth=None일 때 테스트 정확도: {overfit_test:.3f}")
    print()
    print("=== 해석 주의 ===")
    print("훈련 정확도가 높아도 테스트 정확도가 낮으면 과적합이다.")
    print("모델을 복잡하게 만들면 훈련 데이터를 외울 수 있지만,")
    print("새로운 데이터에서는 성능이 떨어질 수 있다.")


if __name__ == "__main__":
    main()
