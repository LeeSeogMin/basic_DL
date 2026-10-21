"""7장 이론 강의용 개념 그림을 만든다.

만드는 그림은 다섯 장이다.
  fig7-1  과소적합 · 알맞음 · 과적합 세 가지를 나란히 놓은 도식
  fig7-2  복잡도를 올릴 때 훈련 오차와 검증 오차가 벌어지는 지점
  fig7-3  훈련 · 검증 · 테스트 세 갈래 분할(4장의 2분할을 셋으로 넓힌다)
  fig7-4  5조각 교차검증(검증 조각을 바꿔 가며 5번 학습한다)
  fig7-5  드롭아웃(학습 회차마다 다른 뉴런을 쉬게 하고 예측할 때 모두 쓴다)

이 그림들은 실험 결과가 아니라 개념 설명용 도식이다. 그림에 쓰는 점과 곡선은
컴퓨터로 만든 합성 데이터이며, 실제 관측값이 아니다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import _krfont

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

RANDOM_SEED = 7

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def box(ax, x, y, w, h, text, facecolor, fontsize=11, textcolor="white"):
    """모서리가 둥근 상자 하나를 그리고 가운데에 글자를 넣는다."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=facecolor,
            edgecolor="none",
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        linespacing=1.5,
    )


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color=color,
        )
    )


def blank_axes(ax, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


# ---------------------------------------------------------------------------
# 그림 7-1: 과소적합 · 알맞음 · 과적합
# ---------------------------------------------------------------------------

def figure_three_fits(ko: bool, out_path: Path) -> None:
    """세 가지 맞춤 상태를 같은 데이터 위에 나란히 그린다."""
    L = _krfont.label
    rng = np.random.default_rng(RANDOM_SEED)

    # 개념 도식용 합성 데이터. 점 10개, 뒤에 숨은 곡선은 완만한 언덕 하나다.
    x = np.linspace(0.05, 0.95, 10)
    truth = np.sin(np.pi * x)
    y = truth + rng.normal(0.0, 0.13, x.size)

    # 곡선은 데이터가 놓인 구간 안에서만 그린다. 구간 밖으로 나가면 높은 차수의
    # 곡선이 위아래로 치솟아 그림 밖으로 나가고, 정작 봐야 할 요동이 안 보인다.
    grid = np.linspace(x.min(), x.max(), 400)

    panels = [
        (1, GRAY,
         L("과소적합", "underfitting", ko),
         L("직선 하나로는 언덕을 못 따라간다", "a straight line cannot follow the hill", ko),
         L("훈련 오차 큼 · 테스트 오차 큼", "train error high / test error high", ko),
         RED),
        (3, GREEN,
         L("알맞음", "good fit", ko),
         L("점을 다 지나지 않지만 흐름을 따라간다", "misses points but follows the shape", ko),
         L("훈련 오차 작음 · 테스트 오차 작음", "train error low / test error low", ko),
         GREEN),
        (9, ORANGE,
         L("과적합", "overfitting", ko),
         L("모든 점을 지나느라 사이에서 요동친다", "passes every point, swings between them", ko),
         L("훈련 오차 0에 가까움 · 테스트 오차 큼", "train error near 0 / test error high", ko),
         RED),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=True)

    for ax, (degree, curve_color, title, note, verdict, verdict_color) in zip(axes, panels):
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        )
        model.fit(x.reshape(-1, 1), y)
        curve = model.predict(grid.reshape(-1, 1))

        ax.plot(
            np.linspace(0.0, 1.0, 400), np.sin(np.pi * np.linspace(0.0, 1.0, 400)),
            color=GRAY, linewidth=1.8, linestyle="--",
            label=L("데이터 뒤에 숨은 진짜 흐름", "the true shape behind the data", ko),
        )
        ax.plot(
            grid, curve,
            color=curve_color, linewidth=2.8,
            label=L("모델이 그린 선", "the line the model drew", ko),
        )
        ax.scatter(
            x, y,
            c=BLUE, s=70, edgecolor="white", linewidth=1.0, zorder=3,
            label=L("훈련 데이터", "training data", ko),
        )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.75, 1.75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=15, pad=10, color=curve_color)
        ax.text(0.5, -0.42, note, ha="center", fontsize=10.5, color="#333333")
        ax.text(0.5, -0.62, verdict, ha="center", fontsize=10.5, color=verdict_color)
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=3, fontsize=11, frameon=False,
    )
    fig.suptitle(
        L("같은 데이터, 세 가지 맞춤 (개념 도식 · 합성 데이터)",
          "Same data, three kinds of fit (schematic, synthetic data)", ko),
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 7-2: 두 곡선이 벌어지는 지점
# ---------------------------------------------------------------------------

def figure_generalization_gap(ko: bool, out_path: Path) -> None:
    """복잡도를 올릴 때 훈련 오차와 검증 오차가 어떻게 갈라지는지 그린다."""
    L = _krfont.label

    # 개념 도식용 곡선. 실험값이 아니라 모양만 보여주는 식이다.
    c = np.linspace(1.0, 10.0, 400)
    train = 0.15 + 0.75 * np.exp(-0.45 * (c - 1.0))
    gap = 0.05 + 0.02 * np.exp(0.32 * (c - 1.0))
    val = train + gap

    best_i = int(np.argmin(val))
    best_c = float(c[best_i])

    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    ax.axvspan(1.0, best_c, color=BLUE, alpha=0.06)
    ax.axvspan(best_c, 10.0, color=RED, alpha=0.06)

    ax.plot(c, train, color=BLUE, linewidth=2.8,
            label=L("훈련 오차", "training error", ko))
    ax.plot(c, val, color=ORANGE, linewidth=2.8,
            label=L("검증 오차", "validation error", ko))
    ax.fill_between(c, train, val, color=RED, alpha=0.10)

    ax.axvline(best_c, color=GREEN, linestyle="--", linewidth=2.0)
    ax.scatter([best_c], [val[best_i]], s=90, color=GREEN, zorder=5)
    ax.annotate(
        L("여기서 멈춘다\n검증 오차가 가장 낮은 지점",
          "stop here\nlowest validation error", ko),
        xy=(best_c, val[best_i]),
        xytext=(best_c - 2.6, val[best_i] + 0.30),
        fontsize=11, color=GREEN, linespacing=1.5,
        arrowprops=dict(arrowstyle="-|>", color=GREEN, linewidth=1.6),
    )

    # 오른쪽 끝에서 두 곡선의 간격을 화살표로 표시한다.
    x_mark = 9.4
    j = int(np.argmin(np.abs(c - x_mark)))
    ax.annotate(
        "", xy=(x_mark, train[j]), xytext=(x_mark, val[j]),
        arrowprops=dict(arrowstyle="<|-|>", color=RED, linewidth=1.8),
    )
    ax.text(
        x_mark - 0.18, (train[j] + val[j]) / 2,
        L("이 간격이\n과적합의 크기", "this gap is\nthe size of overfitting", ko),
        ha="right", va="center", fontsize=11, color=RED, linespacing=1.5,
    )

    ax.text((1.0 + best_c) / 2, 1.02,
            L("과소적합 구역 — 둘 다 높다", "underfitting zone - both are high", ko),
            ha="center", fontsize=11.5, color=BLUE)
    ax.text((best_c + 10.0) / 2, 1.02,
            L("과적합 구역 — 훈련만 내려간다", "overfitting zone - only training drops", ko),
            ha="center", fontsize=11.5, color=RED)

    ax.set_xlim(1.0, 10.0)
    ax.set_ylim(0.0, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        L("모델 복잡도 →  (층·뉴런·차수를 늘릴수록 오른쪽)",
          "model complexity  ->  (more layers / neurons / degree)", ko),
        fontsize=12,
    )
    ax.set_ylabel(L("오차 (작을수록 좋다)", "error (lower is better)", ko), fontsize=12)
    ax.set_title(
        L("두 곡선이 갈라지는 지점이 과적합의 시작이다 (개념 도식)",
          "Overfitting starts where the two curves split (schematic)", ko),
        fontsize=14, pad=12,
    )
    ax.grid(alpha=0.2)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.13),
        ncol=2, fontsize=12, frameon=False,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 7-3: 훈련 · 검증 · 테스트 세 갈래 분할
# ---------------------------------------------------------------------------

def figure_three_way_split(ko: bool, out_path: Path) -> None:
    """4장의 2분할과 7장의 3분할을 위아래로 나란히 놓는다."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 9))

    ax.set_title(
        L("4장에서는 둘로 나눴다. 7장에서는 셋으로 나눈다",
          "Chapter 4 split it in two. Chapter 7 splits it in three", ko),
        fontsize=14, pad=12, loc="left",
    )

    # --- 위: 4장의 2분할 ---
    ax.text(0.2, 8.05, L("4장 — 2분할", "Chapter 4 - two-way split", ko),
            fontsize=12.5, color="#333333")
    box(ax, 0.2, 6.6, 6.6, 1.1,
        L("훈련 데이터 — 모델이 보고 배운다",
          "Training - the model learns from this", ko), BLUE, 11)
    box(ax, 7.0, 6.6, 2.8, 1.1,
        L("테스트 데이터\n성적을 잰다", "Test\nscore it here", ko), ORANGE, 10.5)
    ax.text(5.0, 6.05,
            L("모델을 여러 개 만들어 테스트에서 고르면, 테스트가 훈련의 일부가 된다",
              "Picking a model on the test set turns the test set into training data", ko),
            ha="center", fontsize=10.5, color=RED)

    arrow(ax, 5.0, 5.75, 5.0, 5.05)

    # --- 아래: 7장의 3분할 ---
    ax.text(0.2, 4.55, L("7장 — 3분할", "Chapter 7 - three-way split", ko),
            fontsize=12.5, color="#333333")
    # 상자의 가로 길이를 6 : 2 : 2 비율에 맞춘다.
    box(ax, 0.2, 3.1, 5.52, 1.1,
        L("훈련 데이터 — 모델이 보고 배운다",
          "Training - the model learns from this", ko), BLUE, 11)
    box(ax, 5.92, 3.1, 1.84, 1.1,
        L("검증 데이터\n모델을 고른다", "Validation\npick a model", ko), GREEN, 10.5)
    box(ax, 7.96, 3.1, 1.84, 1.1,
        L("테스트 데이터\n마지막 한 번", "Test\nonce, at the end", ko), ORANGE, 10.5)

    ax.text(2.96, 2.55, L("여러 번 본다", "seen many times", ko),
            ha="center", fontsize=10.5, color=BLUE)
    ax.text(6.84, 2.55, L("여러 번 본다", "seen many times", ko),
            ha="center", fontsize=10.5, color=GREEN)
    ax.text(8.88, 2.55, L("딱 한 번 본다", "seen exactly once", ko),
            ha="center", fontsize=10.5, color=ORANGE)

    ax.text(5.0, 1.35,
            L("검증 데이터가 모델을 고르는 일을 떠맡는다.\n"
              "그래서 테스트 데이터는 마지막까지 처음 보는 상태로 남는다.",
              "The validation set takes over the job of choosing a model,\n"
              "so the test set stays unseen until the very end.", ko),
            ha="center", fontsize=11, color="#333333", linespacing=1.6)

    ax.text(5.0, 0.35,
            L("흔히 훈련 6 : 검증 2 : 테스트 2로 나눈다",
              "a common ratio is 6 : 2 : 2", ko),
            ha="center", fontsize=10.5, color=GRAY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 7-4: 5조각 교차검증
# ---------------------------------------------------------------------------

def figure_cross_validation(ko: bool, out_path: Path) -> None:
    """데이터를 5조각으로 나누고 검증 조각을 바꿔 가며 5번 학습하는 과정을 그린다."""
    L = _krfont.label
    n_fold = 5
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    blank_axes(ax, xlim=(0, 10.4), ylim=(0, 9))

    ax.set_title(
        L("검증 조각을 바꿔 가며 5번 학습한다 (5조각 교차검증)",
          "Rotate the validation piece and train five times (5-fold cross-validation)", ko),
        fontsize=14, pad=12, loc="left",
    )

    x0, cell_w, gap = 1.3, 1.3, 0.08
    row_h, row_step, top = 0.78, 1.0, 6.5

    # 조각 번호
    for j in range(n_fold):
        ax.text(x0 + j * cell_w + cell_w / 2, top + row_h + 0.35,
                L(f"조각 {j + 1}", f"piece {j + 1}", ko),
                ha="center", fontsize=11, color="#333333")
    ax.text(8.85, top + row_h + 0.35, L("검증 점수", "val. score", ko),
            ha="center", fontsize=11, color="#333333")

    for i in range(n_fold):
        y = top - i * row_step
        ax.text(0.15, y + row_h / 2, L(f"{i + 1}회차", f"round {i + 1}", ko),
                ha="left", va="center", fontsize=11.5, color="#333333")
        for j in range(n_fold):
            is_val = (i == j)
            box(ax, x0 + j * cell_w + gap, y, cell_w - 2 * gap, row_h,
                L("검증", "valid", ko) if is_val else L("학습", "train", ko),
                ORANGE if is_val else BLUE, 10.5)
        arrow(ax, x0 + n_fold * cell_w + 0.1, y + row_h / 2,
              x0 + n_fold * cell_w + 0.55, y + row_h / 2)
        box(ax, 8.2, y, 1.3, row_h, L(f"점수 {i + 1}", f"score {i + 1}", ko), GREEN, 10.5)

    # 평균
    y_last = top - (n_fold - 1) * row_step
    arrow(ax, 8.85, y_last - 0.12, 8.85, y_last - 0.7)
    box(ax, 7.6, y_last - 1.65, 2.5, 0.9,
        L("5개 점수의 평균", "mean of the 5 scores", ko), GREEN, 11)

    # 범례
    box(ax, 1.3, 1.2, 0.5, 0.4, "", BLUE)
    ax.text(1.95, 1.4, L("학습에 쓰는 조각 (4개, 80%)", "used for training (4 pieces)", ko),
            va="center", fontsize=10.5, color="#333333")
    box(ax, 1.3, 0.55, 0.5, 0.4, "", ORANGE)
    ax.text(1.95, 0.75, L("검증에 쓰는 조각 (1개, 20%)", "used for validation (1 piece)", ko),
            va="center", fontsize=10.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 7-5: 드롭아웃
# ---------------------------------------------------------------------------

def figure_dropout(ko: bool, out_path: Path) -> None:
    """학습 회차마다 쉬는 뉴런이 달라지고 예측할 때 모두 쓰는 모습을 그린다."""
    L = _krfont.label
    layer_x = [0.0, 1.25, 2.5, 3.75]
    layer_counts = [3, 4, 4, 2]
    layer_names = [
        L("입력", "input", ko),
        L("은닉층 1", "hidden 1", ko),
        L("은닉층 2", "hidden 2", ko),
        L("출력", "output", ko),
    ]
    panels = [
        (
            {(1, 1), (2, 3)},
            L("학습 1회차", "training step 1", ko),
            L("회색 X 뉴런은 이번 회차에 쉰다", "gray X neurons rest in this step", ko),
        ),
        (
            {(1, 3), (2, 0)},
            L("다음 학습 회차", "next training step", ko),
            L("다른 뉴런을 무작위로 쉰다", "a different random set rests", ko),
        ),
        (
            set(),
            L("예측할 때", "during prediction", ko),
            L("모든 뉴런을 사용한다", "all neurons are active", ko),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))

    for ax, (dropped, title, note) in zip(axes, panels):
        blank_axes(ax, xlim=(-0.45, 4.2), ylim=(0.0, 5.2))
        positions = {}
        for layer, (x, count) in enumerate(zip(layer_x, layer_counts)):
            for node, y in enumerate(np.linspace(1.15, 4.05, count)):
                positions[(layer, node)] = (x, float(y))

        for layer in range(len(layer_counts) - 1):
            for left in range(layer_counts[layer]):
                for right in range(layer_counts[layer + 1]):
                    start = (layer, left)
                    end = (layer + 1, right)
                    inactive = start in dropped or end in dropped
                    x1, y1 = positions[start]
                    x2, y2 = positions[end]
                    ax.plot(
                        [x1, x2], [y1, y2],
                        color="#D5D5D5" if inactive else GRAY,
                        linewidth=0.8 if inactive else 1.0,
                        linestyle=":" if inactive else "-",
                        alpha=0.45,
                        zorder=1,
                    )

        for layer, count in enumerate(layer_counts):
            for node in range(count):
                key = (layer, node)
                x, y = positions[key]
                if key in dropped:
                    ax.scatter(
                        [x], [y], s=410, facecolor="#F1F1F1",
                        edgecolor="#B8B8B8", linewidth=1.8, zorder=3,
                    )
                    ax.scatter(
                        [x], [y], s=150, marker="x", color=RED,
                        linewidth=2.2, zorder=4,
                    )
                else:
                    color = ORANGE if layer == len(layer_counts) - 1 else BLUE
                    ax.scatter(
                        [x], [y], s=410, facecolor=color,
                        edgecolor="white", linewidth=1.8, zorder=3,
                    )

        for x, name in zip(layer_x, layer_names):
            ax.text(x, 0.58, name, ha="center", fontsize=10.5, color="#333333")

        ax.set_title(title, fontsize=14, pad=10)
        ax.text(1.88, 4.72, note, ha="center", fontsize=10.5, color="#333333")

    fig.suptitle(
        L("학습할 때만 뉴런 일부를 쉬게 한다 (드롭아웃 개념 도식)",
          "Only training drops neurons (dropout schematic)", ko),
        fontsize=15,
    )
    fig.text(
        0.5, 0.035,
        L("연결선도 쉬는 뉴런과 함께 잠시 끊긴다. 다음 회차에는 다시 연결될 수 있다.",
          "Connections to resting neurons pause too, and can return in the next step.", ko),
        ha="center", fontsize=10.5, color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 7장 개념 그림 생성 ===")

    targets = [
        ("fig7-1-three-fits.png", figure_three_fits,
         "과소적합 · 알맞음 · 과적합 비교"),
        ("fig7-2-generalization-gap.png", figure_generalization_gap,
         "훈련 오차와 검증 오차가 벌어지는 지점"),
        ("fig7-3-three-way-split.png", figure_three_way_split,
         "훈련 · 검증 · 테스트 세 갈래 분할"),
        ("fig7-4-cross-validation.png", figure_cross_validation,
         "5조각 교차검증"),
        ("fig7-5-dropout.png", figure_dropout,
         "학습 회차와 예측 단계의 드롭아웃"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")


if __name__ == "__main__":
    main()
