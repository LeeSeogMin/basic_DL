"""실습 7.2 - 과적합을 진단하고 고친다.

두 장의 그림을 그린다.

  왼쪽 그림(복잡도 곡선): 가로축에 모델 복잡도(다항식 차수)를 놓고
      훈련 오차와 검증 오차를 함께 그린다. 두 곡선이 벌어지기 시작하는 지점이
      과적합이 시작되는 지점이다.
  오른쪽 그림(학습 곡선): 복잡한 모델을 그대로 두고 훈련 데이터만 늘리면서
      두 곡선의 간격이 어떻게 달라지는지 그린다.

두 그림은 두 가지 처방에 각각 대응한다.
  모델을 단순하게 한다 -> 왼쪽 그림에서 검증 오차가 가장 낮은 차수로 내려온다
  데이터를 늘린다      -> 오른쪽 그림에서 두 곡선의 간격이 좁아진다

훈련 데이터를 한 번만 뽑으면 어느 점이 뽑혔는지에 따라 오차가 들쭉날쭉하다.
그래서 같은 크기의 훈련 데이터를 N_REPEATS번 다르게 뽑아 재고, 그중 가운데
값(중앙값)을 쓴다. 곡선이 매끄러워져 두 곡선의 관계가 눈에 들어온다.

데이터는 훈련 · 검증 · 테스트 세 갈래로 나눈다. 테스트 데이터는 맨 마지막에
한 번만 쓴다.

주의: 이 데이터는 실제 관측값이 아니라 컴퓨터로 만든 합성 데이터다.
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
MAX_DEGREE = 10          # 왼쪽 그림에서 차수를 1부터 여기까지 올린다
FIT_SIZE = 25            # 왼쪽 그림에서 쓰는 훈련 데이터 개수
LEARNING_DEGREE = 10     # 오른쪽 그림에서 고정해 두는 차수(복잡한 모델)
NOISE_LEVEL = 0.25       # 데이터에 섞는 잡음의 크기
N_REPEATS = 9            # 같은 크기로 훈련 데이터를 몇 번 다르게 뽑아 볼 것인가

# 마지막에 테스트할 차수를 사람이 직접 정하고 싶을 때 숫자를 넣는다.
# None 이면 검증 오차가 가장 낮은 차수를 코드가 고른다.
# 검증 오차가 비슷한 차수가 여럿이면, 더 단순한 쪽을 직접 넣어 비교해 볼 수 있다.
FINAL_DEGREE = None
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_TOTAL = 1000           # 만들어 둘 전체 데이터 개수
N_POOL = 700             # 훈련 후보
N_VAL = 200              # 검증
LEARNING_SIZES = [20, 25, 30, 40, 60, 90, 140, 200, 300]
Y_TOP = 1.05             # 그림의 세로 상한

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
    x = rng.uniform(0.0, 1.0, n)
    y = true_function(x) + rng.normal(0.0, NOISE_LEVEL, n)
    return x, y


def fit_polynomial(degree: int, x: np.ndarray, y: np.ndarray):
    """차수가 degree인 다항식을 데이터에 맞춘다."""
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


def measure(
    degree: int, size: int,
    x_pool: np.ndarray, y_pool: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
) -> tuple[float, float, int]:
    """훈련 데이터를 여러 번 다르게 뽑아 훈련·검증 오차의 중앙값을 낸다.

    풀에서 size개씩 겹치지 않게 잘라 쓴다. 풀이 모자라면 뽑는 횟수를 줄인다.
    """
    repeats = max(1, min(N_REPEATS, len(x_pool) // size))
    train_scores = []
    val_scores = []

    for r in range(repeats):
        start = r * size
        x_fit = x_pool[start:start + size]
        y_fit = y_pool[start:start + size]
        model = fit_polynomial(degree, x_fit, y_fit)
        train_scores.append(rmse(model, x_fit, y_fit))
        val_scores.append(rmse(model, x_val, y_val))

    return float(np.median(train_scores)), float(np.median(val_scores)), repeats


def complexity_curve(
    x_pool: np.ndarray, y_pool: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
) -> pd.DataFrame:
    """차수를 1부터 MAX_DEGREE까지 올리며 훈련·검증 오차를 잰다."""
    rows = []
    for degree in range(1, MAX_DEGREE + 1):
        train_err, val_err, repeats = measure(
            degree, FIT_SIZE, x_pool, y_pool, x_val, y_val
        )
        rows.append({
            "차수": degree,
            "훈련오차": train_err,
            "검증오차": val_err,
            "뽑은횟수": repeats,
        })
    table = pd.DataFrame(rows)
    table["간격"] = table["검증오차"] - table["훈련오차"]
    return table


def learning_curve(
    x_pool: np.ndarray, y_pool: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
) -> pd.DataFrame:
    """차수를 LEARNING_DEGREE로 고정하고 훈련 데이터 개수만 늘린다."""
    rows = []
    for size in LEARNING_SIZES:
        train_err, val_err, repeats = measure(
            LEARNING_DEGREE, size, x_pool, y_pool, x_val, y_val
        )
        rows.append({
            "훈련데이터수": size,
            "훈련오차": train_err,
            "검증오차": val_err,
            "뽑은횟수": repeats,
        })
    table = pd.DataFrame(rows)
    table["간격"] = table["검증오차"] - table["훈련오차"]
    return table


def save_diagnose_plot(
    ko: bool,
    complexity: pd.DataFrame,
    learning: pd.DataFrame,
    best_degree: int,
    out_path: Path,
) -> None:
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # --- 왼쪽: 복잡도에 따른 두 곡선 ---
    ax = axes[0]
    ax.plot(
        complexity["차수"], complexity["훈련오차"],
        marker="o", color=BLUE, linewidth=2.4,
        label=L("훈련 오차", "training error", ko),
    )
    ax.plot(
        complexity["차수"], complexity["검증오차"],
        marker="s", color=ORANGE, linewidth=2.4,
        label=L("검증 오차", "validation error", ko),
    )
    ax.fill_between(
        complexity["차수"],
        complexity["훈련오차"], complexity["검증오차"],
        color=RED, alpha=0.12,
    )
    ax.axvline(best_degree, color=GREEN, linestyle="--", linewidth=2.0)
    ax.text(
        best_degree + 0.2, Y_TOP * 0.92,
        L(f"검증 오차가 가장 낮은 차수 {best_degree}",
          f"best degree {best_degree}", ko),
        color=GREEN, fontsize=11,
    )
    ax.set_xlim(0.5, MAX_DEGREE + 0.5)
    ax.set_ylim(0.0, Y_TOP)
    ax.set_xticks(list(complexity["차수"]))
    ax.set_xlabel(L("모델 복잡도(다항식 차수)", "model complexity (degree)", ko), fontsize=12)
    ax.set_ylabel(L("오차(RMSE)", "error (RMSE)", ko), fontsize=12)
    ax.set_title(
        L(f"복잡도를 올릴 때 (훈련 데이터 {FIT_SIZE}개)",
          f"raising complexity (train n={FIT_SIZE})", ko),
        fontsize=13, pad=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=2, fontsize=11, frameon=False,
    )

    # --- 오른쪽: 데이터를 늘릴 때 ---
    ax = axes[1]
    ax.plot(
        learning["훈련데이터수"], learning["훈련오차"],
        marker="o", color=BLUE, linewidth=2.4,
        label=L("훈련 오차", "training error", ko),
    )
    ax.plot(
        learning["훈련데이터수"], learning["검증오차"],
        marker="s", color=ORANGE, linewidth=2.4,
        label=L("검증 오차", "validation error", ko),
    )
    ax.fill_between(
        learning["훈련데이터수"],
        learning["훈련오차"], learning["검증오차"],
        color=RED, alpha=0.12,
    )
    ax.set_xlim(0, max(LEARNING_SIZES) + 10)
    ax.set_ylim(0.0, Y_TOP)
    ax.set_xlabel(L("훈련 데이터 개수", "training set size", ko), fontsize=12)
    ax.set_ylabel(L("오차(RMSE)", "error (RMSE)", ko), fontsize=12)
    ax.set_title(
        L(f"데이터를 늘릴 때 (차수 {LEARNING_DEGREE} 고정)",
          f"adding data (degree {LEARNING_DEGREE} fixed)", ko),
        fontsize=13, pad=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=2, fontsize=11, frameon=False,
    )

    fig.suptitle(
        L("붉은 띠의 두께가 과적합의 크기다 (합성 데이터)",
          "The thickness of the red band is the size of overfitting (synthetic data)", ko),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def print_table(title: str, table: pd.DataFrame, first_col: str) -> None:
    print(f"=== {title} ===")
    print(f"{first_col:>12}{'훈련오차':>12}{'검증오차':>12}{'간격':>10}{'뽑은횟수':>10}")
    for _, row in table.iterrows():
        print(
            f"{int(row[first_col]):>12}"
            f"{row['훈련오차']:>12.4f}"
            f"{row['검증오차']:>12.4f}"
            f"{row['간격']:>10.4f}"
            f"{int(row['뽑은횟수']):>10}"
        )
    print()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    rng = np.random.default_rng(RANDOM_SEED)
    x_all, y_all = make_curve_data(N_TOTAL, rng)

    # 세 갈래로 나눈다: 훈련 후보 700 / 검증 200 / 테스트 100
    x_pool, y_pool = x_all[:N_POOL], y_all[:N_POOL]
    x_val, y_val = x_all[N_POOL:N_POOL + N_VAL], y_all[N_POOL:N_POOL + N_VAL]
    x_test, y_test = x_all[N_POOL + N_VAL:], y_all[N_POOL + N_VAL:]

    complexity = complexity_curve(x_pool, y_pool, x_val, y_val)
    learning = learning_curve(x_pool, y_pool, x_val, y_val)

    lowest_row = complexity.loc[complexity["검증오차"].idxmin()]
    lowest_degree = int(lowest_row["차수"])

    if FINAL_DEGREE is None:
        best_degree = lowest_degree
        chosen_by = "검증 오차가 가장 낮은 차수를 코드가 골랐다"
    else:
        best_degree = int(FINAL_DEGREE)
        chosen_by = f"FINAL_DEGREE={FINAL_DEGREE} 로 사람이 직접 정했다"

    best_row = complexity.loc[complexity["차수"] == best_degree].iloc[0]

    complexity.round(4).to_csv(
        RESULTS_DIR / "complexity_curve.csv", index=False, encoding="utf-8-sig"
    )
    learning.round(4).to_csv(
        RESULTS_DIR / "learning_curve.csv", index=False, encoding="utf-8-sig"
    )

    plot_path = RESULTS_DIR / "diagnose_curves.png"
    save_diagnose_plot(ko, complexity, learning, best_degree, plot_path)

    # ------------------------------------------------------------------
    print("=== 실습 7.2 과적합 진단하고 고치기 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 관측값 아님)")
    print(f"전체 {N_TOTAL}개 = 훈련 후보 {len(x_pool)}개 + 검증 {len(x_val)}개 + 테스트 {len(x_test)}개")
    print(f"잡음 크기 NOISE_LEVEL = {NOISE_LEVEL}")
    print(f"같은 크기로 훈련 데이터를 최대 {N_REPEATS}번 다르게 뽑아 중앙값을 쓴다")
    print()

    print_table(
        f"진단 1. 복잡도를 올릴 때 (훈련 데이터 {FIT_SIZE}개 고정)",
        complexity, "차수",
    )
    print(f"검증 오차가 가장 낮은 차수: {best_degree} (검증 오차 {best_row['검증오차']:.4f})")
    first_gap = complexity.loc[complexity["차수"] == 1, "간격"].iloc[0]
    best_gap = float(best_row["간격"])
    last_gap = complexity.loc[complexity["차수"] == MAX_DEGREE, "간격"].iloc[0]
    print(f"차수 1에서 두 곡선의 간격: {first_gap:.4f}")
    print(f"차수 {best_degree}에서 두 곡선의 간격: {best_gap:.4f}")
    print(f"차수 {MAX_DEGREE}에서 두 곡선의 간격: {last_gap:.4f}")
    print()

    print_table(
        f"진단 2. 데이터를 늘릴 때 (차수 {LEARNING_DEGREE} 고정)",
        learning, "훈련데이터수",
    )
    small = learning.iloc[0]
    large = learning.iloc[-1]
    print(f"훈련 {int(small['훈련데이터수'])}개일 때 간격: {small['간격']:.4f}")
    print(f"훈련 {int(large['훈련데이터수'])}개일 때 간격: {large['간격']:.4f}")
    print()

    # ------------------------------------------------------------------
    # 처방 표는 위의 두 표에서 값을 그대로 꺼내 온다. 따로 다시 재지 않는다.
    print("=== 처방 두 개를 적용한 결과 ===")
    none_row = complexity.loc[complexity["차수"] == LEARNING_DEGREE].iloc[0]
    simple_row = complexity.loc[complexity["차수"] == best_degree].iloc[0]
    more_row = learning.iloc[-1]

    print(f"{'처방':<30}{'훈련오차':>10}{'검증오차':>10}{'간격':>10}")
    for name, row in [
        (f"없음(차수 {LEARNING_DEGREE}, 데이터 {FIT_SIZE}개)", none_row),
        (f"모델을 단순하게(차수 {best_degree}, 데이터 {FIT_SIZE}개)", simple_row),
        (f"데이터를 늘림(차수 {LEARNING_DEGREE}, {int(more_row['훈련데이터수'])}개)", more_row),
    ]:
        print(f"{name:<30}{row['훈련오차']:>10.4f}{row['검증오차']:>10.4f}{row['간격']:>10.4f}")
    print()

    # ------------------------------------------------------------------
    print("=== 마지막에 한 번만 쓰는 테스트 데이터 ===")
    final_model = fit_polynomial(best_degree, x_pool, y_pool)
    print(f"고른 모델: 차수 {best_degree}, 훈련 후보 {len(x_pool)}개 전부로 다시 학습")
    print(f"고른 방법: {chosen_by}")
    print(f"검증 오차: {rmse(final_model, x_val, y_val):.4f}")
    print(f"테스트 오차: {rmse(final_model, x_test, y_test):.4f}")
    print("검증 데이터로 차수를 골랐으므로, 최종 성적은 테스트 데이터에서 잰다.")
    print()

    print("=== 저장된 결과 ===")
    print(f"진단 그림: results/{plot_path.name}")
    print("복잡도 표: results/complexity_curve.csv")
    print("학습 곡선 표: results/learning_curve.csv")
    print()
    print("MAX_DEGREE, FIT_SIZE, LEARNING_DEGREE, NOISE_LEVEL, N_REPEATS를 바꿔 다시 실행한다.")


if __name__ == "__main__":
    main()
