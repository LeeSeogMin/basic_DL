"""실습 4.2 - 정확도 95% 모델의 정체를 밝힌다.

메일 1000통 중 스팸이 50통뿐인 자료를 만든다. 스팸은 전체의 5%다.
이런 자료에서는 "전부 정상"이라고만 답해도 정확도가 95%가 나온다.

세 가지 모델을 같은 테스트 데이터에서 비교한다.
  모델 A  무조건 정상이라고 답한다
  모델 B  로지스틱 회귀를 기본 설정으로 학습한다
  모델 C  로지스틱 회귀에 "스팸을 놓치면 더 큰 손해"라고 알려주고 학습한다

정확도만 보면 셋의 차이가 보이지 않는다. 혼동행렬을 봐야 보인다.

주의: 이 데이터는 실제 메일이 아니라 컴퓨터로 만든 합성 데이터다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import _krfont

RANDOM_SEED = 42
N_MAILS = 1000
N_SPAM = 50

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


def make_synthetic_mails() -> tuple[np.ndarray, np.ndarray]:
    """합성 메일 데이터를 만든다.

    특징은 두 개다.
      - 메일에 들어 있는 링크 개수
      - 제목에서 대문자가 차지하는 비율
    스팸은 링크가 많고 대문자 비율이 높은 편이지만, 정상 메일과 겹치는 구간이 있다.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_normal = N_MAILS - N_SPAM

    normal_links = np.clip(rng.normal(2.0, 1.5, n_normal), 0, None)
    normal_caps = np.clip(rng.normal(0.12, 0.06, n_normal), 0, 1)

    spam_links = np.clip(rng.normal(6.0, 3.0, N_SPAM), 0, None)
    spam_caps = np.clip(rng.normal(0.30, 0.12, N_SPAM), 0, 1)

    x = np.vstack([
        np.column_stack([normal_links, normal_caps]),
        np.column_stack([spam_links, spam_caps]),
    ])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(N_SPAM, dtype=int)])

    order = rng.permutation(len(y))
    return x[order], y[order]


def draw_confusion(ax, matrix: np.ndarray, title: str, ko: bool) -> None:
    """혼동행렬 한 장을 그린다. 맞힌 칸은 초록, 틀린 칸은 빨강으로 칠한다."""
    L = _krfont.label
    colors = np.array([["#54A24B", "#E45756"], ["#E45756", "#54A24B"]])

    for row in range(2):
        for col in range(2):
            ax.add_patch(plt.Rectangle((col, 1 - row), 1, 1,
                                       facecolor=colors[row, col], alpha=0.85))
            ax.text(col + 0.5, 1.5 - row, str(matrix[row, col]),
                    ha="center", va="center", fontsize=22, color="white")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([1.5, 0.5])
    ax.set_xticklabels([L("예측: 정상", "pred: normal", ko),
                        L("예측: 스팸", "pred: spam", ko)], fontsize=10)
    ax.set_yticklabels([L("정답: 정상", "true: normal", ko),
                        L("정답: 스팸", "true: spam", ko)], fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    L = _krfont.label

    x, y = make_synthetic_mails()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y,
    )

    models = [
        ("A 무조건 정상", DummyClassifier(strategy="most_frequent")),
        ("B 기본 모델", LogisticRegression(random_state=RANDOM_SEED)),
        ("C 스팸 중시 모델", LogisticRegression(
            random_state=RANDOM_SEED, class_weight="balanced")),
    ]

    results = []
    matrices = {}
    for name, model in models:
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        matrix = confusion_matrix(y_test, pred, labels=[0, 1])
        matrices[name] = matrix

        spam_total = int((y_test == 1).sum())
        spam_caught = int(matrix[1, 1])
        false_alarm = int(matrix[0, 1])

        results.append({
            "모델": name,
            "정확도": round(float(accuracy_score(y_test, pred)), 3),
            "스팸 잡음": f"{spam_caught}/{spam_total}",
            "정상을 스팸이라 함": false_alarm,
        })

    table = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (name, _model) in zip(axes, models):
        row = table[table["모델"] == name].iloc[0]
        title = L(
            f"{name}\n정확도 {row['정확도']:.3f} · 스팸 {row['스팸 잡음']}",
            f"{name}\nacc {row['정확도']:.3f} · spam {row['스팸 잡음']}",
            ko,
        )
        draw_confusion(ax, matrices[name], title, ko)

    fig.suptitle(
        L("정확도는 비슷한데 잡은 스팸 수는 다르다 (합성 데이터)",
          "Similar accuracy, very different spam catches (synthetic data)", ko),
        fontsize=14,
    )
    fig.tight_layout()
    plot_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    table_path = RESULTS_DIR / "accuracy_trap_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    print("=== 실습 4.2 정확도 95% 모델의 정체 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 메일 데이터(실제 메일 아님)")
    print(f"전체 {len(x)}통 중 스팸 {int(y.sum())}통 = {y.mean() * 100:.1f}%")
    print(f"훈련 {len(x_train)}통, 테스트 {len(x_test)}통"
          f"(테스트 안의 스팸 {int(y_test.sum())}통)")
    print()

    print("=== 세 모델 비교 ===")
    print(table.to_string(index=False))
    print()

    for name, matrix in matrices.items():
        print(f"--- {name} 혼동행렬 ---")
        print("               예측:정상  예측:스팸")
        print(f"  정답:정상  {matrix[0, 0]:>8}  {matrix[0, 1]:>8}")
        print(f"  정답:스팸  {matrix[1, 0]:>8}  {matrix[1, 1]:>8}")
        print()

    print("=== 저장된 결과 ===")
    print(f"혼동행렬 그림: results/{plot_path.name}")
    print(f"비교 표: results/{table_path.name}")
    print()
    print("정확도가 높다고 좋은 모델이 아니다. 무엇을 놓쳤는지는 혼동행렬에서만 보인다.")


if __name__ == "__main__":
    main()
