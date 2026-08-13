"""9주차 실습: 이미지 분류기와 실패 갤러리.

sklearn의 load_digits 데이터(8x8 흑백 숫자 이미지, 1797장)를 사용한다.
MLP 분류기로 숫자를 분류하고, 틀린 이미지를 모아 실패 갤러리를 만든다.

CNN 원리는 강의에서 다루고, 실습은 '이미지를 숫자로 다루고 분류하는 경험'에 집중한다.
sklearn만 사용하므로 CPU에서 즉시 실행된다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


# ── 1. 데이터 불러오기 ──────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """load_digits 데이터를 불러온다.

    Returns:
        images: (N, 8, 8) 형태의 원본 이미지 배열
        x: (N, 64) 형태의 펼친 특성 배열
        y: (N,) 형태의 정답 배열 (0~9)
    """
    digits = load_digits()
    images = digits.images          # (1797, 8, 8)
    x = digits.data                 # (1797, 64) — 이미 펼쳐진 형태
    y = digits.target               # (1797,)
    return images, x, y


# ── 2. 샘플 이미지 시각화 ────────────────────────────────────────

def save_sample_digits(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    n_samples: int = 16,
) -> None:
    """숫자 이미지 샘플을 격자로 그려 저장한다."""
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(len(labels), size=n_samples, replace=False)
    indices = np.sort(indices)

    cols = 4
    rows = n_samples // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    for ax, idx in zip(axes.ravel(), indices):
        ax.imshow(images[idx], cmap="gray_r", interpolation="nearest")
        ax.set_title(f"label={labels[idx]}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Sample digits (8x8 pixels, values 0-16)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ── 3. 실패 갤러리 ──────────────────────────────────────────────

def save_failure_gallery(
    images_test: np.ndarray,
    y_test: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    max_show: int = 20,
) -> np.ndarray:
    """틀린 이미지를 모아 실패 갤러리 PNG로 저장한다.

    Returns:
        mistake_indices: 틀린 이미지의 테스트 인덱스 배열
    """
    mistake_indices = np.where(pred != y_test)[0]

    show = mistake_indices[:max_show]
    n = len(show)
    if n == 0:
        # 틀린 게 없으면 빈 그림은 저장하지 않는다
        return mistake_indices

    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.8 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, ax in enumerate(axes.ravel()):
        if i < n:
            idx = show[i]
            ax.imshow(images_test[idx], cmap="gray_r", interpolation="nearest")
            ax.set_title(
                f"true={y_test[idx]} pred={pred[idx]}",
                fontsize=9,
                color="red",
            )
        ax.axis("off")

    fig.suptitle("Failure gallery: misclassified digits", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return mistake_indices


# ── 4. 메인 ─────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 데이터 준비 ──
    images, x, y = load_data()

    print("=== 9주차 이미지 인식과 CNN ===")
    print(f"데이터셋: sklearn load_digits")
    print(f"이미지 크기: {images.shape[1]}x{images.shape[2]} 픽셀, 흑백 1채널")
    print(f"픽셀값 범위: {x.min():.1f} ~ {x.max():.1f}")
    print(f"전체 이미지 수: {len(y)}")
    print(f"클래스 수: {len(np.unique(y))} (숫자 0~9)")
    print()

    # 클래스별 개수
    unique, counts = np.unique(y, return_counts=True)
    print("클래스별 이미지 수:")
    for digit, count in zip(unique, counts):
        print(f"  숫자 {digit}: {count}장")
    print()

    # 샘플 이미지 저장
    sample_path = RESULTS_DIR / "sample_digits.png"
    save_sample_digits(images, y, sample_path)

    # ── 훈련/테스트 분할 ──
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    images_test = images[
        np.isin(np.arange(len(y)), train_test_split(
            np.arange(len(y)),
            test_size=0.3,
            random_state=RANDOM_SEED,
            stratify=y,
        )[1])
    ]

    print(f"훈련 데이터 수: {len(x_train)}")
    print(f"테스트 데이터 수: {len(x_test)}")
    print()

    # ── 기준선 모델 ──
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(x_test))

    print("=== 기준선 모델 ===")
    print("전략: 모든 이미지를 훈련 데이터에서 가장 많은 숫자로 예측")
    print(f"기준선 정확도: {baseline_acc:.3f}")
    print()

    # ── MLP 분류기 ──
    # 8x8=64 입력 → 은닉층(64, 32) → 10 출력
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.15,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    model_acc = accuracy_score(y_test, pred)
    matrix = confusion_matrix(y_test, pred)

    print("=== MLP 분류기 ===")
    print(f"구조: 입력 64 → 은닉층(64, 32) → 출력 10")
    print(f"학습 epoch 수: {model.n_iter_}")
    print(f"테스트 정확도: {model_acc:.3f}")
    print()
    print("혼동행렬:")
    print(matrix)
    print()

    # ── 틀린 사례 분석 ──
    mistake_indices = save_failure_gallery(
        images_test, y_test, pred,
        RESULTS_DIR / "failure_gallery.png",
    )

    n_mistakes = len(mistake_indices)
    print(f"=== 틀린 사례: {n_mistakes}개 ===")

    # 틀린 사례 CSV 저장
    mistake_rows = []
    for rank, idx in enumerate(mistake_indices, start=1):
        mistake_rows.append({
            "rank": rank,
            "test_index": int(idx),
            "true_label": int(y_test[idx]),
            "predicted_label": int(pred[idx]),
        })

    mistakes_df = pd.DataFrame(mistake_rows)
    mistakes_path = RESULTS_DIR / "misclassified_examples.csv"
    mistakes_df.to_csv(mistakes_path, index=False)

    if not mistakes_df.empty:
        print(mistakes_df.to_string(index=False))
    else:
        print("틀린 사례가 없습니다.")
    print()

    # 가장 자주 혼동하는 쌍
    if n_mistakes > 0:
        print("=== 자주 혼동하는 숫자 쌍 ===")
        pair_counts: dict[tuple[int, int], int] = {}
        for idx in mistake_indices:
            pair = (int(y_test[idx]), int(pred[idx]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  실제 {pair[0]} → 예측 {pair[1]}: {count}건")
        print()

    # ── 혼동행렬 그림 저장 ──
    confusion_path = RESULTS_DIR / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[str(d) for d in range(10)],
    )
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Confusion matrix (digits 0-9)")
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)

    # ── 결과 요약 ──
    print("=== 저장된 결과 ===")
    print(f"샘플 이미지: {sample_path.relative_to(CHAPTER_DIR)}")
    print(f"혼동행렬 그림: {confusion_path.relative_to(CHAPTER_DIR)}")
    print(f"실패 갤러리: {(RESULTS_DIR / 'failure_gallery.png').relative_to(CHAPTER_DIR)}")
    print(f"틀린 사례 CSV: {mistakes_path.relative_to(CHAPTER_DIR)}")
    print()
    print("=== 해석 주의 ===")
    print("이미지를 숫자 배열로 바꾸면 분류기에 넣을 수 있다.")
    print("정확도만 보지 말고 혼동행렬과 실패 갤러리를 함께 확인한다.")
    print("어떤 숫자를 어떤 숫자로 틀렸는지가 모델의 약점을 보여준다.")


if __name__ == "__main__":
    main()
