"""실습 4.1 - 내가 그은 선과 모델이 찾은 선을 비교한다.

시험을 통과한 학생과 통과하지 못한 학생의 자료를 점으로 찍는다.
점 하나가 학생 한 명이고, 가로축은 하루 공부 시간, 세로축은 하루 수면 시간이다.

먼저 사람이 직접 선을 하나 긋고(규칙 기반), 그다음 모델에게 선을 찾게 한다(머신러닝).
두 선의 정확도를 같은 테스트 데이터에서 비교한다.

주의: 이 데이터는 실제 학생 자료가 아니라 컴퓨터로 만든 합성 데이터다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.
# 내가 그을 선:  수면시간 = MY_SLOPE * 공부시간 + MY_INTERCEPT
# 선보다 위에 있는 점을 "통과"로 예측한다.
MY_SLOPE = -1.0
MY_INTERCEPT = 11.0
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


def make_synthetic_students(n_samples: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """합성 학생 데이터를 만든다.

    공부 시간과 수면 시간이 모두 넉넉하면 통과할 확률이 높아지도록 만들되,
    잡음을 섞어 경계 근처에서는 결과가 갈리게 한다. 그래야 틀린 사례가 생긴다.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    study = rng.uniform(0.0, 10.0, n_samples)   # 하루 공부 시간
    sleep = rng.uniform(3.0, 9.0, n_samples)    # 하루 수면 시간

    hidden_score = 0.8 * study + 0.6 * sleep + rng.normal(0.0, 1.0, n_samples)
    passed = (hidden_score > 7.5).astype(int)

    x = np.column_stack([study, sleep])
    return x, passed


def predict_with_my_line(x: np.ndarray) -> np.ndarray:
    """내가 그은 선으로 예측한다. 선보다 위에 있으면 1(통과)."""
    study = x[:, 0]
    sleep = x[:, 1]
    line_height = MY_SLOPE * study + MY_INTERCEPT
    return (sleep > line_height).astype(int)


def model_line(model: LogisticRegression) -> tuple[float, float]:
    """로지스틱 회귀가 찾은 경계선을 기울기와 절편으로 바꾼다.

    모델의 판단식은 w0*공부 + w1*수면 + b = 0 이다.
    이를 수면 = (-w0/w1)*공부 + (-b/w1) 형태로 정리한다.
    """
    w0, w1 = model.coef_[0]
    b = model.intercept_[0]
    slope = -w0 / w1
    intercept = -b / w1
    return float(slope), float(intercept)


def save_comparison_plot(
    ko: bool,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    learned_slope: float,
    learned_intercept: float,
    out_path: Path,
) -> None:
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(9.0, 6.4))

    for label_value, color, ko_name, en_name in [
        (0, "#4C78A8", "미통과", "not passed"),
        (1, "#F58518", "통과", "passed"),
    ]:
        mask_train = y_train == label_value
        ax.scatter(
            x_train[mask_train, 0], x_train[mask_train, 1],
            c=color, s=44, edgecolor="white", linewidth=0.6,
            label=L(f"훈련 데이터 - {ko_name}", f"train - {en_name}", ko),
        )
        mask_test = y_test == label_value
        ax.scatter(
            x_test[mask_test, 0], x_test[mask_test, 1],
            c=color, s=90, marker="X", edgecolor="black", linewidth=0.7,
            label=L(f"테스트 데이터 - {ko_name}", f"test - {en_name}", ko),
        )

    grid_x = np.linspace(0.0, 10.0, 200)
    ax.plot(
        grid_x, MY_SLOPE * grid_x + MY_INTERCEPT,
        linestyle="--", linewidth=2.4, color="#79706E",
        label=L("내가 그은 선", "my line", ko),
    )
    ax.plot(
        grid_x, learned_slope * grid_x + learned_intercept,
        linestyle="-", linewidth=2.8, color="#54A24B",
        label=L("모델이 찾은 선", "model line", ko),
    )

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(2.5, 9.5)
    ax.set_xlabel(L("하루 공부 시간(시간)", "study hours per day", ko), fontsize=12)
    ax.set_ylabel(L("하루 수면 시간(시간)", "sleep hours per day", ko), fontsize=12)
    ax.set_title(
        L("내가 그은 선과 모델이 찾은 선 (합성 데이터)",
          "My line vs the model's line (synthetic data)", ko),
        fontsize=14, pad=12,
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=3, fontsize=9.5, frameon=False,
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    x, y = make_synthetic_students()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y,
    )

    # 1) 기준선: 훈련 데이터에서 더 많았던 쪽으로만 찍는다
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(x_test))

    # 2) 내가 그은 선
    my_train_acc = accuracy_score(y_train, predict_with_my_line(x_train))
    my_test_acc = accuracy_score(y_test, predict_with_my_line(x_test))

    # 3) 모델이 찾은 선
    model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(x_train, y_train)
    model_pred = model.predict(x_test)
    model_train_acc = accuracy_score(y_train, model.predict(x_train))
    model_test_acc = accuracy_score(y_test, model_pred)
    learned_slope, learned_intercept = model_line(model)

    # 틀린 사례 모으기
    wrong_index = np.where(model_pred != y_test)[0]
    rows = []
    for rank, i in enumerate(wrong_index[:5], start=1):
        rows.append({
            "순위": rank,
            "공부시간": round(float(x_test[i, 0]), 2),
            "수면시간": round(float(x_test[i, 1]), 2),
            "실제": "통과" if y_test[i] == 1 else "미통과",
            "모델예측": "통과" if model_pred[i] == 1 else "미통과",
        })
    wrong_df = pd.DataFrame(rows)
    wrong_path = RESULTS_DIR / "misclassified_examples.csv"
    wrong_df.to_csv(wrong_path, index=False, encoding="utf-8-sig")

    plot_path = RESULTS_DIR / "decision_boundary.png"
    save_comparison_plot(
        ko, x_train, y_train, x_test, y_test,
        learned_slope, learned_intercept, plot_path,
    )

    # ------------------------------------------------------------------
    print("=== 실습 4.1 내가 그은 선 vs 모델이 찾은 선 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 학생 자료 아님)")
    print(f"전체 {len(x)}명 = 훈련 {len(x_train)}명 + 테스트 {len(x_test)}명")
    print(f"전체에서 통과한 학생: {int(y.sum())}명, 미통과: {int(len(y) - y.sum())}명")
    print()

    print("=== 내가 그은 선 ===")
    print(f"식: 수면시간 = {MY_SLOPE} x 공부시간 + {MY_INTERCEPT}")
    print(f"훈련 데이터 정확도: {my_train_acc:.3f}")
    print(f"테스트 데이터 정확도: {my_test_acc:.3f}")
    print()

    print("=== 모델이 찾은 선 ===")
    print(f"식: 수면시간 = {learned_slope:.2f} x 공부시간 + {learned_intercept:.2f}")
    print(f"훈련 데이터 정확도: {model_train_acc:.3f}")
    print(f"테스트 데이터 정확도: {model_test_acc:.3f}")
    print()

    print("=== 테스트 데이터에서 세 가지 비교 ===")
    print(f"{'방법':<22}{'테스트 정확도':>12}")
    print(f"{'기준선(무조건 한쪽)':<22}{baseline_acc:>12.3f}")
    print(f"{'내가 그은 선':<22}{my_test_acc:>12.3f}")
    print(f"{'모델이 찾은 선':<22}{model_test_acc:>12.3f}")
    print()

    print("=== 모델이 틀린 사례 ===")
    if wrong_df.empty:
        print("틀린 사례가 없습니다.")
    else:
        print(wrong_df.to_string(index=False))
    print()

    print("=== 저장된 결과 ===")
    print(f"비교 그림: results/{plot_path.name}")
    print(f"틀린 사례: results/{wrong_path.name}")
    print()
    print("MY_SLOPE와 MY_INTERCEPT를 바꿔 다시 실행하면 내 선의 정확도가 달라진다.")


if __name__ == "__main__":
    main()
