"""실습 5.2 - 직선 하나로 못 나누는 문제를 은닉층으로 푼다.

XOR는 두 입력이 서로 다를 때만 1이 되는 문제다.

    (0, 0) -> 0      (0, 1) -> 1
    (1, 0) -> 1      (1, 1) -> 0

정답 1인 두 점은 서로 마주 보는 자리에 있고, 정답 0인 두 점도 마주 본다.
그래서 종이에 자를 대고 아무리 그어 봐도 직선 하나로는 두 색을 못 가른다.

여기서는 같은 데이터에 두 모델을 올려 비교한다.
  A 은닉층 없는 모델 - 뉴런 하나. 4장에서 쓴 로지스틱 회귀와 같은 구조다.
  B 은닉층 있는 모델 - 은닉 뉴런 HIDDEN_UNITS 개를 거쳐 출력 뉴런으로 간다.

주의: 이 데이터는 컴퓨터로 만든 합성 데이터다. XOR 네 점 주변에
잡음을 뿌려 만들었고, 난수 seed를 고정해 실행할 때마다 같은 값이 나온다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import _krfont

# ---------------------------------------------------------------------------
# 여기 세 값을 바꿔가며 실습한다.
#   HIDDEN_UNITS   은닉층에 넣을 뉴런 개수. 1로 줄이면 은닉층이 있어도 실패한다.
#   NOISE          네 모서리 주변에 뿌리는 잡음의 크기. 키우면 두 덩어리가 섞인다.
#   N_PER_CORNER   모서리 하나마다 만들 점의 개수.
HIDDEN_UNITS = 4
NOISE = 0.16
N_PER_CORNER = 60
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"

# 판단 영역을 칠할 때 쓰는 옅은 색 (파랑=0, 주황=1)
REGION_CMAP = ListedColormap(["#D6E1EE", "#FBDFC4"])

# XOR 네 모서리와 정답
CORNERS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
CORNER_LABELS = np.array([0, 1, 1, 0])


def make_xor_data() -> tuple[np.ndarray, np.ndarray]:
    """XOR 네 모서리 주변에 잡음을 뿌린 합성 데이터를 만든다.

    네 점만 찍으면 그림이 밋밋하고 훈련/테스트로 나눌 수도 없다.
    그래서 모서리마다 N_PER_CORNER 개씩 점을 흩뿌린다.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    points = []
    labels = []
    for corner, answer in zip(CORNERS, CORNER_LABELS):
        cloud = corner + rng.normal(0.0, NOISE, size=(N_PER_CORNER, 2))
        points.append(cloud)
        labels.append(np.full(N_PER_CORNER, answer))
    return np.vstack(points), np.concatenate(labels)


def decision_grid(model, low: float = -0.6, high: float = 1.6, step: int = 300):
    """모델이 격자 위의 모든 자리를 어떻게 판정하는지 계산한다."""
    axis = np.linspace(low, high, step)
    gx, gy = np.meshgrid(axis, axis)
    flat = np.column_stack([gx.ravel(), gy.ravel()])
    zone = model.predict(flat).reshape(gx.shape)
    return gx, gy, zone


def scatter_points(ax, x: np.ndarray, y: np.ndarray, ko: bool) -> None:
    """데이터 점을 정답 색으로 찍는다."""
    L = _krfont.label
    for value, color, ko_name, en_name in [
        (0, BLUE, "정답 0", "label 0"),
        (1, ORANGE, "정답 1", "label 1"),
    ]:
        mask = y == value
        ax.scatter(
            x[mask, 0], x[mask, 1],
            c=color, s=26, edgecolor="white", linewidth=0.4, zorder=4,
            label=L(ko_name, en_name, ko),
        )


def save_comparison_plot(
    ko: bool,
    x: np.ndarray,
    y: np.ndarray,
    panels: list[tuple[str, object, float]],
    out_path: Path,
) -> None:
    """두 모델의 판단 영역을 한 장에 나란히 그린다."""
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6))

    for ax, (title, model, test_acc) in zip(axes, panels):
        gx, gy, zone = decision_grid(model)
        ax.contourf(gx, gy, zone, levels=[-0.5, 0.5, 1.5], cmap=REGION_CMAP)
        scatter_points(ax, x, y, ko)

        # XOR 네 모서리의 원래 자리를 검은 테두리로 표시한다.
        ax.scatter(
            CORNERS[:, 0], CORNERS[:, 1],
            facecolor="none", edgecolor="black", s=230, linewidth=1.4, zorder=6,
            label=L("XOR 네 모서리", "the four XOR corners", ko),
        )

        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel("x1", fontsize=11)
        ax.set_ylabel("x2", fontsize=11)
        ax.set_title(
            f"{title}\n" + L(f"테스트 정확도 {test_acc:.3f}",
                             f"test accuracy {test_acc:.3f}", ko),
            fontsize=12.5, pad=10,
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                  ncol=3, fontsize=9.5, frameon=False)

    fig.suptitle(
        L("같은 XOR 데이터, 은닉층이 없을 때와 있을 때 (합성 데이터)",
          "The same XOR data, without and with a hidden layer (synthetic)", ko),
        fontsize=14, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    x, y = make_xor_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    # A 은닉층 없는 모델: 입력 두 개가 곧바로 출력 뉴런 하나로 간다.
    flat_model = LogisticRegression()
    flat_model.fit(x_train, y_train)

    # B 은닉층 있는 모델: 입력 -> 은닉 뉴런 HIDDEN_UNITS 개 -> 출력 뉴런 하나.
    deep_model = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        activation="relu",
        solver="adam",
        max_iter=6000,
        random_state=RANDOM_SEED,
    )
    deep_model.fit(x_train, y_train)

    flat_train = accuracy_score(y_train, flat_model.predict(x_train))
    flat_test = accuracy_score(y_test, flat_model.predict(x_test))
    deep_train = accuracy_score(y_train, deep_model.predict(x_train))
    deep_test = accuracy_score(y_test, deep_model.predict(x_test))

    plot_path = RESULTS_DIR / "xor_comparison.png"
    save_comparison_plot(
        ko, x, y,
        [
            (_krfont.label("A 은닉층 없음 (뉴런 1개)",
                           "A no hidden layer (1 neuron)", ko),
             flat_model, flat_test),
            (_krfont.label(f"B 은닉층 있음 (은닉 뉴런 {HIDDEN_UNITS}개)",
                           f"B with hidden layer ({HIDDEN_UNITS} neurons)", ko),
             deep_model, deep_test),
        ],
        plot_path,
    )

    # ------------------------------------------------------------------
    print("=== 실습 5.2 XOR - 직선 하나로 못 나누는 문제 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 데이터(XOR 네 모서리 주변에 잡음을 뿌림)")
    print(f"설정: HIDDEN_UNITS={HIDDEN_UNITS}, NOISE={NOISE}, "
          f"N_PER_CORNER={N_PER_CORNER}, seed={RANDOM_SEED}")
    print(f"전체 {len(x)}개 = 훈련 {len(x_train)}개 + 테스트 {len(x_test)}개")
    print(f"정답 1: {int(y.sum())}개, 정답 0: {int(len(y) - y.sum())}개")
    print()

    print("=== XOR 네 모서리의 정답 ===")
    print(f"{'x1':>4}{'x2':>5}{'정답':>6}")
    for (x1, x2), answer in zip(CORNERS, CORNER_LABELS):
        print(f"{x1:>4.0f}{x2:>5.0f}{answer:>6d}")
    print("정답 1인 두 점이 서로 마주 본다. 그래서 직선 하나로는 못 가른다.")
    print()

    print("=== 두 모델 비교 ===")
    print(f"{'모델':<26}{'훈련 정확도':>12}{'테스트 정확도':>14}")
    print(f"{'A 은닉층 없음(뉴런 1개)':<26}{flat_train:>12.3f}{flat_test:>14.3f}")
    print(f"{'B 은닉 뉴런 ' + str(HIDDEN_UNITS) + '개':<26}"
          f"{deep_train:>12.3f}{deep_test:>14.3f}")
    print()

    print("=== 두 모델이 네 모서리를 어떻게 판정하는가 ===")
    flat_corner = flat_model.predict(CORNERS)
    deep_corner = deep_model.predict(CORNERS)
    print(f"{'x1':>4}{'x2':>5}{'정답':>6}{'A 판정':>8}{'B 판정':>8}")
    for i, (x1, x2) in enumerate(CORNERS):
        print(f"{x1:>4.0f}{x2:>5.0f}{CORNER_LABELS[i]:>6d}"
              f"{flat_corner[i]:>8d}{deep_corner[i]:>8d}")
    print()

    print("=== 은닉 뉴런 개수를 바꾸면 ===")
    print(f"{'은닉 뉴런 수':<14}{'훈련 정확도':>12}{'테스트 정확도':>14}")
    for units in [1, 2, 3, 4, 8]:
        trial = MLPClassifier(
            hidden_layer_sizes=(units,),
            activation="relu",
            solver="adam",
            max_iter=6000,
            random_state=RANDOM_SEED,
        )
        trial.fit(x_train, y_train)
        trial_train = accuracy_score(y_train, trial.predict(x_train))
        trial_test = accuracy_score(y_test, trial.predict(x_test))
        print(f"{units:<14d}{trial_train:>12.3f}{trial_test:>14.3f}")
    print()

    print("=== 저장된 결과 ===")
    print(f"두 모델 비교 그림: results/{plot_path.name}")
    print()
    print("HIDDEN_UNITS를 1로 바꿔 다시 실행하면 은닉층이 있어도 실패한다.")


if __name__ == "__main__":
    main()
