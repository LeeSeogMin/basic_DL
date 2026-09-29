"""실습 4.1 - 방법을 바꿔 가며 판단 기준을 찾고 같은 자로 비교한다.

시험을 통과한 학생과 통과하지 못한 학생의 자료를 점으로 찍는다.
점 하나가 학생 한 명이고, 가로축은 하루 공부 시간, 세로축은 하루 수면 시간이다.

같은 훈련 데이터를 방법 셋에 각각 넘긴다. 기준선은 더 많았던 쪽으로만 찍고,
로지스틱 회귀는 직선 하나를 찾고, 결정 트리는 조건문 몇 줄을 찾는다.
세 방법의 정확도를 같은 테스트 데이터에서 비교한다.

주의: 이 데이터는 실제 학생 자료가 아니라 컴퓨터로 만든 합성 데이터다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.
# 결정 트리가 조건문을 몇 단까지 겹쳐 쓸지 정한다.
# None 으로 바꾸면 제한을 풀어, 훈련 데이터를 통째로 외운 트리를 볼 수 있다.
MY_TREE_DEPTH = 3
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


def box(ax, x, y, w, h, text, facecolor, fontsize=11, textcolor="white"):
    """모서리가 둥근 상자 하나를 그리고 가운데에 글자를 넣는다."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=facecolor, edgecolor="none", zorder=3,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, linespacing=1.4, zorder=4)


def tree_rules(tree: DecisionTreeClassifier, ko: bool) -> str:
    """결정 트리가 찾은 판단 기준을 사람이 읽는 조건문으로 바꾼다.

    1절에서 사람이 손으로 쓰던 `if 공부시간 >= 6: ...` 와 같은 모양이다.
    다른 점은 조건과 기준값을 사람이 아니라 모델이 데이터에서 골랐다는 것이다.
    """
    L = _krfont.label
    names = [L("공부시간", "study", ko), L("수면시간", "sleep", ko)]
    text = export_text(tree, feature_names=names, decimals=1)
    return text.replace("class: 0", L("판정: 미통과", "class: not passed", ko)).replace(
        "class: 1", L("판정: 통과", "class: passed", ko))


def save_tree_plot(tree: DecisionTreeClassifier, ko: bool, out_path: Path) -> None:
    """결정 트리가 찾은 조건문을 가지가 갈라지는 그림으로 그린다.

    글로 읽는 조건문은 들여쓰기를 눈으로 따라가야 하지만, 그림에서는 어느
    조건이 어느 조건 아래에 붙는지가 위치로 드러난다.
    """
    L = _krfont.label
    inner = tree.tree_
    names = [L("공부시간", "study", ko), L("수면시간", "sleep", ko)]
    class_names = [L("미통과", "not passed", ko), L("통과", "passed", ko)]
    blue, orange, gray = "#4C78A8", "#F58518", "#79706E"

    # 잎을 왼쪽부터 차례로 놓고, 부모는 두 자식의 가운데에 놓는다.
    # SPREAD 는 이웃한 잎 사이의 간격이다. 상자 너비보다 커야 서로 닿지 않는다.
    SPREAD = 1.6
    pos, slot = {}, [0]

    def place(node: int, depth: int) -> float:
        left, right = inner.children_left[node], inner.children_right[node]
        if left == -1:
            x = slot[0] * SPREAD
            slot[0] += 1
        else:
            x = (place(left, depth + 1) + place(right, depth + 1)) / 2
        pos[node] = (x, depth)
        return x

    place(0, 0)
    max_depth = max(d for _, d in pos.values())
    n_leaf = slot[0]

    box_w, box_h = 1.32, 0.52
    fig, ax = plt.subplots(figsize=(2.0 * n_leaf + 1.0, 1.55 * (max_depth + 1) + 1.0))
    ax.set_xlim(-box_w / 2 - 0.3, (n_leaf - 1) * SPREAD + box_w / 2 + 0.3)
    ax.set_ylim(-0.75, max_depth + 0.75)
    ax.axis("off")

    def y_of(depth: int) -> float:
        return max_depth - depth

    # 먼저 가지를 긋는다. 상자가 선을 덮도록 순서를 잡는다.
    for node, (x, depth) in pos.items():
        left, right = inner.children_left[node], inner.children_right[node]
        if left == -1:
            continue
        for child, answer in ((left, L("예", "yes", ko)), (right, L("아니오", "no", ko))):
            cx, cdepth = pos[child]
            ax.plot([x, cx], [y_of(depth) - box_h / 2, y_of(cdepth) + box_h / 2],
                    color=gray, linewidth=1.4, zorder=1)
            ax.text((x + cx) / 2, (y_of(depth) + y_of(cdepth)) / 2,
                    answer, ha="center", va="center", fontsize=9.5, color=gray,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none"), zorder=2)

    for node, (x, depth) in pos.items():
        n_here = int(inner.n_node_samples[node])
        if inner.children_left[node] == -1:
            verdict = int(inner.value[node][0].argmax())
            text = L(f"{class_names[verdict]}\n훈련 {n_here}명",
                     f"{class_names[verdict]}\n{n_here} rows", ko)
            color = orange if verdict == 1 else blue
        else:
            feature = names[inner.feature[node]]
            text = L(f"{feature} ≤ {inner.threshold[node]:.1f} ?\n훈련 {n_here}명",
                     f"{feature} <= {inner.threshold[node]:.1f} ?\n{n_here} rows", ko)
            color = gray
        box(ax, x - box_w / 2, y_of(depth) - box_h / 2, box_w, box_h,
            text, color, fontsize=9.5)

    ax.set_title(
        L("결정 트리가 찾은 판단 기준 — 조건이 맞으면 왼쪽, 아니면 오른쪽으로 내려간다",
          "The rule the decision tree found - go left if the test holds", ko),
        fontsize=12.5, pad=14, loc="left",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_comparison_plot(
    ko: bool,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    learned_slope: float,
    learned_intercept: float,
    tree: DecisionTreeClassifier,
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
        grid_x, learned_slope * grid_x + learned_intercept,
        linestyle="-", linewidth=2.8, color="#54A24B",
        label=L("로지스틱 회귀가 찾은 선", "logistic regression line", ko),
    )

    # 결정 트리의 경계는 직선이 아니라 축에 나란한 계단이다. 조건문이
    # "공부시간 <= 6.1" 처럼 한 축씩만 자르기 때문이다.
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(0.0, 10.0, 400), np.linspace(2.5, 9.5, 400))
    mesh_pred = tree.predict(
        np.column_stack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_x.shape)
    ax.contour(mesh_x, mesh_y, mesh_pred, levels=[0.5],
               colors="#B279A2", linewidths=2.6, linestyles=":")
    ax.plot([], [], linestyle=":", linewidth=2.6, color="#B279A2",
            label=L("결정 트리가 나눈 경계", "decision tree boundary", ko))

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(2.5, 9.5)
    ax.set_xlabel(L("하루 공부 시간(시간)", "study hours per day", ko), fontsize=12)
    ax.set_ylabel(L("하루 수면 시간(시간)", "sleep hours per day", ko), fontsize=12)
    ax.set_title(
        L("두 모델이 찾은 판단 기준 — 로지스틱 회귀 · 결정 트리 (합성 데이터)",
          "Two learned decision rules - logistic regression and decision tree "
          "(synthetic data)", ko),
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

    # 2) 로지스틱 회귀가 찾은 선
    model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(x_train, y_train)
    model_pred = model.predict(x_test)
    model_train_acc = accuracy_score(y_train, model.predict(x_train))
    model_test_acc = accuracy_score(y_test, model_pred)
    learned_slope, learned_intercept = model_line(model)

    # 3) 같은 데이터에서 다른 방법: 결정 트리는 선 대신 조건문을 찾는다
    tree = DecisionTreeClassifier(max_depth=MY_TREE_DEPTH, random_state=RANDOM_SEED)
    tree.fit(x_train, y_train)
    tree_train_acc = accuracy_score(y_train, tree.predict(x_train))
    tree_test_acc = accuracy_score(y_test, tree.predict(x_test))

    # 깊이를 바꾸면 훈련 성적과 테스트 성적이 어떻게 갈리는지 함께 잰다
    depth_rows = []
    for depth in (1, 2, 3, None):
        probe = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED)
        probe.fit(x_train, y_train)
        depth_rows.append({
            "깊이": "제한없음" if depth is None else depth,
            "훈련 정확도": round(float(accuracy_score(y_train, probe.predict(x_train))), 3),
            "테스트 정확도": round(float(accuracy_score(y_test, probe.predict(x_test))), 3),
        })
    depth_df = pd.DataFrame(depth_rows)

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

    tree_path = RESULTS_DIR / "tree_rules.png"
    save_tree_plot(tree, ko, tree_path)

    plot_path = RESULTS_DIR / "decision_boundary.png"
    save_comparison_plot(
        ko, x_train, y_train, x_test, y_test,
        learned_slope, learned_intercept, tree, plot_path,
    )

    # ------------------------------------------------------------------
    print("=== 실습 4.1 세 방법이 찾은 판단 기준 비교 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 학생 자료 아님)")
    print(f"전체 {len(x)}명 = 훈련 {len(x_train)}명 + 테스트 {len(x_test)}명")
    print(f"전체에서 통과한 학생: {int(y.sum())}명, 미통과: {int(len(y) - y.sum())}명")
    print()

    print("=== 로지스틱 회귀가 찾은 선 ===")
    print(f"식: 수면시간 = {learned_slope:.2f} x 공부시간 + {learned_intercept:.2f}")
    print(f"훈련 데이터 정확도: {model_train_acc:.3f}")
    print(f"테스트 데이터 정확도: {model_test_acc:.3f}")
    print()

    depth_label = "제한없음" if MY_TREE_DEPTH is None else MY_TREE_DEPTH
    print(f"=== 결정 트리가 찾은 규칙 (깊이 {depth_label}) ===")
    print(tree_rules(tree, ko).rstrip())
    print(f"훈련 데이터 정확도: {tree_train_acc:.3f}")
    print(f"테스트 데이터 정확도: {tree_test_acc:.3f}")
    print()

    print("=== 테스트 데이터에서 세 가지 비교 ===")
    print(f"{'방법':<26}{'테스트 정확도':>12}")
    print(f"{'기준선(무조건 한쪽)':<26}{baseline_acc:>12.3f}")
    print(f"{'로지스틱 회귀가 찾은 선':<26}{model_test_acc:>12.3f}")
    print(f"{'결정 트리가 찾은 규칙':<26}{tree_test_acc:>12.3f}")
    print()

    print("=== 트리 깊이를 바꾸면 ===")
    print(depth_df.to_string(index=False))
    print()

    print("=== 모델이 틀린 사례 ===")
    if wrong_df.empty:
        print("틀린 사례가 없습니다.")
    else:
        print(wrong_df.to_string(index=False))
    print()

    print("=== 저장된 결과 ===")
    print(f"트리 그림: results/{tree_path.name}")
    print(f"비교 그림: results/{plot_path.name}")
    print(f"틀린 사례: results/{wrong_path.name}")
    print()
    print("MY_TREE_DEPTH를 바꾸면 트리가 쓰는 조건문의 수와 두 정확도가 함께 달라진다.")


if __name__ == "__main__":
    main()
