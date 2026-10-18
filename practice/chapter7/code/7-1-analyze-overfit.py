"""실습 7.1 - 과적합된 모델을 불러와 분석한다.

교수가 미리 만들어 둔 파일(practice/chapter7/data/)을 불러온다.
  train.csv, test.csv      훈련 데이터 30개, 테스트 데이터 200개
  model_degree01.joblib    차수 1  다항식 모델
  model_degree04.joblib    차수 4  다항식 모델
  model_degree15.joblib    차수 15 다항식 모델

불러온 모델로 훈련 오차와 테스트 오차를 계산하고, 두 오차의 차이와 곡선이 점 사이에서
어디까지 솟구치는지 확인한다. 차수가 높을수록 훈련 오차는 내려가고 테스트 오차는 올라간다.

주의: 이 데이터는 실제 관측값이 아니라 컴퓨터로 만든 합성 데이터다.
파일이 없거나 scikit-learn 버전 때문에 모델을 못 불러오면, 7-1-make-overfit-files.py를
한 번 실행해 파일을 다시 만든다. joblib 파일은 저장소에서 받은 것만 불러온다.

차수 1, 4, 15라는 조합은 scikit-learn 공식 예제 Underfitting vs. Overfitting에서
쓰는 조합과 같다.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.
DEGREES = [1, 4, 15]      # 불러올 모델의 차수. 1, 4, 15 중에서 고른다
X_RANGE = (0.0, 1.0)      # 곡선의 최솟값·최댓값을 확인할 x 구간. 훈련 점은 0과 1 사이에 있다
# ---------------------------------------------------------------------------

Y_LIMIT = (-2.2, 2.2)     # 그림의 세로 범위

CHAPTER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CHAPTER_DIR / "data"
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def true_function(x: np.ndarray) -> np.ndarray:
    """데이터 뒤에 숨어 있는 진짜 곡선. 모델은 이 곡선을 모른다."""
    return np.cos(1.5 * np.pi * x)


def load_files(degrees: list[int]):
    """훈련·테스트 CSV와 차수별 모델을 불러온다. 없으면 안내 문구와 함께 멈춘다."""
    needed = [DATA_DIR / "train.csv", DATA_DIR / "test.csv"]
    needed += [DATA_DIR / f"model_degree{d:02d}.joblib" for d in degrees]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise SystemExit(
            f"불러올 파일이 없다: {missing}\n"
            "code 폴더에서 python 7-1-make-overfit-files.py를 한 번 실행해 파일을 만든다."
        )

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    # joblib 파일은 불러올 때 코드를 실행할 수 있다. 저장소에서 받은 파일만 불러온다.
    models = [joblib.load(DATA_DIR / f"model_degree{d:02d}.joblib") for d in degrees]
    return train, test, models


def rmse(model, x: np.ndarray, y: np.ndarray) -> float:
    """예측이 정답에서 평균 몇만큼 떨어져 있는지 계산한다. 작을수록 잘 맞춘 것이다."""
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

    train, test, models = load_files(DEGREES)
    x_train, y_train = train["x"].to_numpy(), train["y"].to_numpy()
    x_test, y_test = test["x"].to_numpy(), test["y"].to_numpy()

    scores = []
    for degree, model in zip(DEGREES, models):
        scores.append((degree, rmse(model, x_train, y_train), rmse(model, x_test, y_test)))

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
    print("=== 실습 7.1 과적합된 모델 분석하기 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 관측값 아님)")
    print(f"불러온 파일: data/train.csv, data/test.csv, 모델 {len(models)}개 (data/model_degree*.joblib)")
    print(f"훈련 데이터 {len(x_train)}개, 테스트 데이터 {len(x_test)}개")
    print(f"불러온 모델의 차수: {DEGREES}")
    print()

    print("=== 차수별 오차 (RMSE, 작을수록 잘 맞춘 것) ===")
    print(f"{'차수':>4}{'훈련오차':>12}{'테스트오차':>13}{'차이':>11}")
    for degree, train_err, test_err in scores:
        print(f"{degree:>4}{train_err:>12.4f}{test_err:>13.4f}{test_err - train_err:>11.4f}")
    print()

    by_train = [row[0] for row in sorted(scores, key=lambda row: row[1])]
    by_test = [row[0] for row in sorted(scores, key=lambda row: row[2])]
    print("오차가 작은 차수부터 나열한 순위")
    print(f"  훈련 오차 순위: {' > '.join(str(d) for d in by_train)}")
    print(f"  테스트 오차 순위: {' > '.join(str(d) for d in by_test)}")
    print()

    best_degree, _, best_test = min(scores, key=lambda row: row[2])
    print(f"테스트 오차가 가장 작은 차수: {best_degree} (테스트 오차 {best_test:.4f})")
    print()

    print("=== 훈련 점에 얼마나 붙었고, 점 사이에서 얼마나 솟구쳤는가 ===")
    print(f"{'차수':>4}{'훈련 점에서 벗어난 최대 거리':>22}{'곡선 최솟값':>12}{'곡선 최댓값':>12}")
    grid = np.linspace(X_RANGE[0], X_RANGE[1], 400)
    for degree, model in zip(DEGREES, models):
        residual = np.abs(model.predict(x_train.reshape(-1, 1)) - y_train)
        curve = model.predict(grid.reshape(-1, 1))
        print(f"{degree:>4}{residual.max():>22.4f}{curve.min():>12.2f}{curve.max():>12.2f}")
    print(f"확인한 x 구간: {X_RANGE[0]} 에서 {X_RANGE[1]} 사이")
    print("참고: 진짜 곡선은 -1.00 에서 1.00 사이만 움직인다")
    print()

    print("=== 저장된 결과 ===")
    print(f"비교 그림: results/{plot_path.name}")
    print(f"오차 표: results/{table_path.name}")
    print()
    print("DEGREES와 X_RANGE를 바꿔 다시 실행하면 그림과 표가 달라진다.")


if __name__ == "__main__":
    main()
