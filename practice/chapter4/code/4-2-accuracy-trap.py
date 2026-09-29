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

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

        # 정밀도의 분모는 "스팸이라고 부른 메일 수"다. 한 통도 부르지 않은
        # 모델(A)에서는 0으로 나누게 되므로 계산하지 않고 "-"로 적는다.
        called_spam = spam_caught + false_alarm
        precision = spam_caught / called_spam if called_spam else None
        recall = spam_caught / spam_total

        # F1은 정밀도가 정의되지 않는 A에서도 계산된다. 잡은 스팸이 0통이면
        # F1 = 2·TP / (2·TP + FP + FN) 의 분자가 0이 되어 값이 0으로 정해진다.
        f1 = float(f1_score(y_test, pred, zero_division=0))

        results.append({
            "모델": name,
            "정확도": round(float(accuracy_score(y_test, pred)), 3),
            "정밀도": "-" if precision is None else f"{precision:.3f}",
            "재현율": f"{recall:.3f}",
            "F1": f"{f1:.3f}",
            "스팸 잡음": f"{spam_caught}/{spam_total}",
            "정상을 스팸이라 함": false_alarm,
        })

    table = pd.DataFrame(results)

    table_path = RESULTS_DIR / "accuracy_trap_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    print("=== 실습 4.2 정확도 95% 모델의 정체 ===")
    print("데이터: 컴퓨터로 만든 합성 메일 데이터(실제 메일 아님)")
    print(f"전체 {len(x)}통 중 스팸 {int(y.sum())}통 = {y.mean() * 100:.1f}%")
    print(f"훈련 {len(x_train)}통, 테스트 {len(x_test)}통"
          f"(테스트 안의 스팸 {int(y_test.sum())}통)")
    print()

    print("=== 세 모델 비교 ===")
    print(table.to_string(index=False))
    print()

    # 칸 배치는 본문 표 4.6·4.7과 같게 스팸을 위·왼쪽에 둔다
    for name, matrix in matrices.items():
        print(f"--- {name} 혼동행렬 ---")
        print("               예측:스팸  예측:정상")
        print(f"  정답:스팸  {matrix[1, 1]:>8}  {matrix[1, 0]:>8}")
        print(f"  정답:정상  {matrix[0, 1]:>8}  {matrix[0, 0]:>8}")
        print()

    print("=== 저장된 결과 ===")
    print(f"비교 표: results/{table_path.name}")
    print()
    print("정확도가 높다고 좋은 모델이 아니다. 무엇을 놓쳤는지는 혼동행렬에서만 보인다.")


if __name__ == "__main__":
    main()
