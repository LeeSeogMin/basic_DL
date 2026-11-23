"""실습 13.1과 13.2가 함께 쓰는 합성 데이터와 모델.

이 파일은 이름이 밑줄로 시작하므로 실행 증거 게이트(run_and_capture.py)가
직접 실행하지 않는다. 두 실습 코드에서 import 해서 쓴다.

데이터는 "하루 날씨 숫자 세 개를 보고 그날 공공자전거 대여가 많은 날인지 맞히는"
문제를 흉내 낸 합성 데이터다. 실제 대여 기록이 아니다. 숫자는 전부 아래 규칙으로
컴퓨터가 만든다.

  - 기온이 24도에 가까울수록 대여가 많다. 그보다 춥거나 더우면 줄어든다.
  - 비가 많이 올수록, 바람이 셀수록 대여가 줄어든다.

모델은 봄·가을 날씨(기온 3~26도)만 보고 배운다. 이 범위에서는 "더울수록 대여가
많다"만 보이므로, 모델은 26도보다 더운 날에 대여가 줄어든다는 것을 배우지 못한다.
실습 13.2가 이 점을 확인한다.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
N_DAYS = 1500                 # 모델이 배우는 봄·가을 날 수
TRAIN_TEMP = (3.0, 26.0)      # 모델이 배우는 기온 범위(도)

# 입력 세 개. (이름, 단위, 앱이 받는 최솟값, 앱이 받는 최댓값, 눈금 간격)
# "앱이 받는 범위"는 세상에 있을 수 있는 값의 범위다. 모델이 배운 범위보다 넓다.
FEATURES = [
    ("기온", "도", -20.0, 45.0, 0.5),
    ("강수량", "mm", 0.0, 100.0, 0.5),
    ("풍속", "m/s", 0.0, 30.0, 0.5),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
FEATURE_NAMES_EN = ["temperature", "rain", "wind"]
LABELS = ["대여가 적은 날", "대여가 많은 날"]


def make_days(n: int, temp_range: tuple[float, float], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """합성 날씨 n일치와 정답(대여가 많은 날이면 1)을 만든다."""
    rng = np.random.default_rng(seed)
    temp = rng.uniform(temp_range[0], temp_range[1], n)
    rain = np.where(rng.random(n) < 0.6, 0.0, rng.exponential(6.0, n))   # 열흘 중 엿새는 비가 안 온다
    rain = np.clip(rain, 0.0, 60.0)
    wind = np.clip(rng.gamma(3.0, 1.0, n), 0.0, 15.0)

    # 정답을 만드는 규칙. 기온은 24도에서 가장 높고 양쪽으로 내려간다.
    score = 2.4 - ((temp - 24.0) / 6.5) ** 2 - 0.30 * rain - 0.30 * (wind - 3.0)
    prob_many = 1.0 / (1.0 + np.exp(-1.5 * score))
    y = (rng.random(n) < prob_many).astype(int)

    x = np.column_stack([np.round(temp, 1), np.round(rain, 1), np.round(wind, 1)])
    return x, y


def train_model():
    """봄·가을 데이터를 만들어 훈련·테스트로 나누고 로지스틱 회귀를 학습시킨다."""
    x, y = make_days(N_DAYS, TRAIN_TEMP, RANDOM_SEED)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(x_train, y_train)
    return model, x_train, x_test, y_train, y_test


def learned_ranges(x_train: np.ndarray) -> list[tuple[float, float]]:
    """입력마다 모델이 훈련에서 본 최솟값과 최댓값을 돌려준다."""
    return [(float(x_train[:, i].min()), float(x_train[:, i].max())) for i in range(x_train.shape[1])]
