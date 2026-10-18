"""실습 7.1 준비 - 과적합된 모델과 데이터 파일을 만들어 저장한다.

이 스크립트는 교수가 한 번 실행해 파일을 만들기만 한다. 학생은 이 스크립트를 돌리지
않고, 저장된 파일을 불러와 과적합을 분석한다(교재 실습 7.1, 7-1-analyze-overfit.py).

만드는 파일은 practice/chapter7/data/ 아래 다섯 개다.
  train.csv, test.csv           훈련 데이터 30개, 테스트 데이터 200개
  model_degree01.joblib         차수 1  다항식 모델 (과소적합)
  model_degree04.joblib         차수 4  다항식 모델 (알맞음)
  model_degree15.joblib         차수 15 다항식 모델 (과적합)

데이터: 컴퓨터로 만든 합성 데이터다. 실제 관측값이 아니다.
scikit-learn 버전이 다르면 저장된 모델을 불러올 때 경고가 뜨거나 실패할 수 있다.
그때는 이 스크립트를 다시 실행해 파일을 새로 만든다.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DEGREES = [1, 4, 15]      # 저장할 다항식 차수 세 개
NOISE_LEVEL = 0.25        # 데이터에 섞는 잡음의 크기
N_TRAIN = 30              # 훈련 데이터 점의 개수
N_TEST = 200              # 테스트 데이터 점의 개수(훈련에 쓰지 않는다)
RANDOM_SEED = 42

CHAPTER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CHAPTER_DIR / "data"


def true_function(x: np.ndarray) -> np.ndarray:
    """데이터 뒤에 숨어 있는 진짜 곡선. 모델은 이 곡선을 모른다."""
    return np.cos(1.5 * np.pi * x)


def make_curve_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """진짜 곡선 위에 잡음을 섞어 합성 데이터를 만든다."""
    x = np.sort(rng.uniform(0.0, 1.0, n))
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


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    x_train, y_train = make_curve_data(N_TRAIN, rng)
    x_test, y_test = make_curve_data(N_TEST, rng)

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    pd.DataFrame({"x": x_train, "y": y_train}).to_csv(train_path, index=False)
    pd.DataFrame({"x": x_test, "y": y_test}).to_csv(test_path, index=False)

    print("=== 실습 7.1 준비: 과적합 파일 만들기 ===")
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 관측값 아님)")
    print(f"잡음 크기 {NOISE_LEVEL}, 난수 씨앗 {RANDOM_SEED}, scikit-learn {sklearn.__version__}")
    print()
    print("저장한 파일:")
    print(f"  data/{train_path.name}  훈련 데이터 {len(x_train)}개")
    print(f"  data/{test_path.name}  테스트 데이터 {len(x_test)}개")

    for degree in DEGREES:
        model = fit_polynomial(degree, x_train, y_train)
        model_path = DATA_DIR / f"model_degree{degree:02d}.joblib"
        joblib.dump(model, model_path)
        print(f"  data/{model_path.name}  차수 {degree} 다항식 모델")

    print()
    print("학생은 이 스크립트를 실행하지 않는다. 7-1-analyze-overfit.py로 위 파일을 불러와 분석한다.")


if __name__ == "__main__":
    main()
