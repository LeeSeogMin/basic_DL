"""실습 7.1 - 일부러 과적합을 만든다.

잡음이 섞인 곡선 데이터를 만들고, 다항식 차수를 1, 4, 15로 올려 세 모델을 맞춘다.
차수가 높아질수록 곡선이 훈련 데이터의 점에 바싹 달라붙지만, 점과 점 사이에서는
위아래로 요동친다. 훈련 오차는 내려가고 테스트 오차는 올라간다.
N_SAMPLES를 차수보다 하나 큰 값(예: 16)으로 줄이면 곡선이 모든 점을 통과한다.

주의: 이 데이터는 실제 관측값이 아니라 컴퓨터로 만든 합성 데이터다.

차수 1, 4, 15라는 조합은 scikit-learn 공식 예제 Underfitting vs. Overfitting에서
쓰는 조합과 같다. 같은 그림을 우리 데이터로 다시 만들어 본다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.
DEGREES = [1, 4, 15]      # 맞춰 볼 다항식 차수 세 개
NOISE_LEVEL = 0.25        # 데이터에 섞는 잡음의 크기. 0으로 하면 잡음이 없다
N_SAMPLES = 30            # 훈련 데이터 점의 개수
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_TEST = 200              # 테스트 데이터 점의 개수(훈련에 쓰지 않는다)
Y_LIMIT = (-2.2, 2.2)     # 그림의 세로 범위

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def true_function(x: np.ndarray) -> np.ndarray:
    """데이터 뒤에 숨어 있는 진짜 곡선. 모델은 이 곡선을 모른다."""
    return np.cos(1.5 * np.pi * x)


def make_curve_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """진짜 곡선 위에 잡음을 섞어 합성 데이터를 만든다."""
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y = true_function(x) + rng.normal(0.0, NOISE_LEVEL, n)
    return x, y


def fit_polynomial(degree: int, x: np.ndarray, y: np.ndarray):
    """차수가 degree인 다항식을 데이터에 맞춘다.

    PolynomialFeatures가 x, x^2, ... , x^degree 열을 만들고
    LinearRegression이 각 열에 붙는 계수를 찾는다.
    StandardScaler는 높은 차수에서 숫자 크기가 벌어지는 것을 막는다.
    """
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )
    model.fit(x.reshape(-1, 1), y)
    return model


def rmse(model, x: np.ndarray, y: np.ndarray) -> float:
    """예측이 정답에서 평균 몇만큼 떨어져 있는지 잰다. 작을수록 잘 맞춘 것이다."""
    pred = model.predict(x.reshape(-1, 1))
    return float(np.sqrt(mean_squared_error(y, pred)))


def save_fit_plot(
    ko: bool,
    x_train: np.ndarray,
    y_train: np.ndarray,
    models: list,
    scores: list[tuple[int, float, float]],
    out_path: Path,
) -> None:
    L = _krfont.label
    grid = np.linspace(0.0, 1.0, 400)

    fig, axes = plt.subplots(1, len(DEGREES), figsize=(13.5, 4.8), sharey=True)
    if len(DEGREES) == 1:
        axes = [axes]

    for ax, model, (degree, train_err, test_err) in zip(axes, models, scores):
        curve = model.predict(grid.reshape(-1, 1))

        ax.plot(
            grid, true_function(grid),
            color=GRAY, linewidth=2.0, linestyle="--",
            label=L("진짜 곡선", "true curve", ko),
        )
        ax.plot(
            grid, curve,
            color=GREEN, linewidth=2.4,
            label=L("모델이 그린 곡선", "model curve", ko),
        )
        ax.scatter(
            x_train, y_train,
            c=BLUE, s=52, edgecolor="white", linewidth=0.8, zorder=3,
            label=L("훈련 데이터", "training data", ko),
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(*Y_LIMIT)
        ax.set_xlabel(L("입력 x", "input x", ko), fontsize=11)
        ax.set_title(
            L(f"차수 {degree}", f"degree {degree}", ko)
            + "\n"
            + L(f"훈련 오차 {train_err:.3f}  테스트 오차 {test_err:.3f}",
                f"train {train_err:.3f}  test {test_err:.3f}", ko),
            fontsize=12, pad=10,
        )
        ax.grid(alpha=0.25)

        # 곡선이 그림 밖으로 나가면 그 사실을 그림 안에 적는다.
        if curve.max() > Y_LIMIT[1] or curve.min() < Y_LIMIT[0]:
            ax.text(
                0.5, Y_LIMIT[0] + 0.16,
                L("곡선이 그림 밖까지 솟구친다", "curve runs off the chart", ko),
                ha="center", fontsize=10.5, color=RED,
            )

    axes[0].set_ylabel(L("출력 y", "output y", ko), fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=3, fontsize=11, frameon=False,
    )

    fig.suptitle(
        L("차수를 올릴수록 훈련 점에 달라붙고, 점 사이에서 요동친다 (합성 데이터)",
          "Higher degree hugs the training points and swings between them (synthetic data)", ko),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    rng = np.random.default_rng(RANDOM_SEED)
    x_train, y_train = make_curve_data(N_SAMPLES, rng)
    x_test, y_test = make_curve_data(N_TEST, rng)

    models = []
    scores = []
    for degree in DEGREES:
        model = fit_polynomial(degree, x_train, y_train)
        train_err = rmse(model, x_train, y_train)
        test_err = rmse(model, x_test, y_test)
        models.append(model)
        scores.append((degree, train_err, test_err))

    table = pd.DataFrame(
        [
            {
                "차수": degree,
                "훈련오차": round(train_err, 4),
                "테스트오차": round(test_err, 4),
                "차이": round(test_err - train_err, 4),
            }
            for degree, train_err, test_err in scores
        ]
    )
    table_path = RESULTS_DIR / "overfit_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    plot_path = RESULTS_DIR / "overfit_curves.png"
    save_fit_plot(ko, x_train, y_train, models, scores, plot_path)

    # ------------------------------------------------------------------
    print("=== 실습 7.1 일부러 과적합 만들기 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 관측값 아님)")
    print(f"훈련 데이터 {N_SAMPLES}개, 테스트 데이터 {N_TEST}개")
    print(f"잡음 크기 NOISE_LEVEL = {NOISE_LEVEL}")
    print(f"맞춰 본 차수: {DEGREES}")
    print()

    print("=== 차수별 오차 (RMSE, 작을수록 잘 맞춘 것) ===")
    print(f"{'차수':>4}{'훈련오차':>12}{'테스트오차':>13}{'차이':>11}")
    for degree, train_err, test_err in scores:
        print(f"{degree:>4}{train_err:>12.4f}{test_err:>13.4f}{test_err - train_err:>11.4f}")
    print()

    best_degree, _, best_test = min(scores, key=lambda row: row[2])
    print(f"테스트 오차가 가장 작은 차수: {best_degree} (테스트 오차 {best_test:.4f})")
    print()

    print("=== 훈련 점에 얼마나 붙었고, 점 사이에서 얼마나 솟구쳤는가 ===")
    print(f"{'차수':>4}{'훈련 점에서 벗어난 최대 거리':>22}{'곡선 최솟값':>12}{'곡선 최댓값':>12}")
    grid = np.linspace(0.0, 1.0, 400)
    for degree, model in zip(DEGREES, models):
        residual = np.abs(model.predict(x_train.reshape(-1, 1)) - y_train)
        curve = model.predict(grid.reshape(-1, 1))
        print(f"{degree:>4}{residual.max():>22.4f}{curve.min():>12.2f}{curve.max():>12.2f}")
    print("참고: 진짜 곡선은 -1.00 에서 1.00 사이만 움직인다")
    print()

    print("=== 저장된 결과 ===")
    print(f"비교 그림: results/{plot_path.name}")
    print(f"오차 표: results/{table_path.name}")
    print()
    print("DEGREES, NOISE_LEVEL, N_SAMPLES를 바꿔 다시 실행하면 그림과 표가 달라진다.")


if __name__ == "__main__":
    main()
