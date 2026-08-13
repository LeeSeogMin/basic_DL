"""13주차 실습: 콘솔 기반 AI 데모 앱

scikit-learn의 숫자 데이터셋(digits)으로 분류 모델을 학습시키고,
사용자가 숫자 특징을 입력하면 예측 결과를 돌려주는 콘솔 데모 앱이다.

이 코드는 데모 모드로 실행된다. 미리 정해둔 입력으로 자동 실행해
결과를 보여주며, 모델의 한계 안내 문구를 함께 출력한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


def train_digit_model() -> tuple:
    """숫자 분류 모델을 학습시키고 관련 데이터를 반환한다."""
    digits = load_digits()
    x, y = digits.data, digits.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y,
    )

    # 기준선 모델
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(x_test))

    # 로지스틱 회귀 분류기
    model = LogisticRegression(
        max_iter=5000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    model_acc = accuracy_score(y_test, pred)
    matrix = confusion_matrix(y_test, pred)

    return model, x_train, x_test, y_train, y_test, baseline_acc, model_acc, matrix, digits


def predict_with_confidence(model, sample: np.ndarray) -> tuple[int, float, np.ndarray]:
    """하나의 샘플에 대해 예측 결과, 최대 확률, 전체 확률을 반환한다."""
    proba = model.predict_proba(sample.reshape(1, -1))[0]
    predicted_class = int(np.argmax(proba))
    confidence = float(proba[predicted_class])
    return predicted_class, confidence, proba


def format_limitation_notice(confidence: float) -> str:
    """모델 한계 안내 문구를 반환한다."""
    lines = [
        "[모델 한계 안내]",
        f"  이 예측의 신뢰도: {confidence:.1%}",
    ]
    if confidence < 0.5:
        lines.append("  주의: 신뢰도가 50% 미만이다. 이 예측은 불확실하다.")
        lines.append("  모델이 이 입력을 충분히 학습하지 못했을 수 있다.")
    elif confidence < 0.8:
        lines.append("  참고: 신뢰도가 80% 미만이다. 다른 숫자일 가능성이 있다.")
    else:
        lines.append("  신뢰도가 높지만, 모델이 틀릴 수 있다.")
    lines.append("  이 모델은 8x8 픽셀 손글씨 숫자 데이터로 학습했다.")
    lines.append("  실제 손글씨, 인쇄체, 다른 크기의 이미지에는 사용할 수 없다.")
    return "\n".join(lines)


def run_demo(model, x_test: np.ndarray, y_test: np.ndarray) -> list[dict]:
    """데모 모드: 미리 정해둔 테스트 샘플로 예측을 수행한다."""
    # 다양한 숫자를 보여주기 위해 각 숫자별로 하나씩 선택한다
    rng = np.random.default_rng(RANDOM_SEED)
    demo_indices = []
    for digit in range(10):
        candidates = np.where(y_test == digit)[0]
        if len(candidates) > 0:
            idx = rng.choice(candidates)
            demo_indices.append(idx)

    results = []
    for idx in demo_indices:
        sample = x_test[idx]
        true_label = int(y_test[idx])
        predicted, confidence, proba = predict_with_confidence(model, sample)
        correct = predicted == true_label

        results.append({
            "test_index": int(idx),
            "true_label": true_label,
            "predicted": predicted,
            "confidence": round(confidence, 4),
            "correct": correct,
        })

    return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 13주차 나만의 AI 앱 만들기 ===")
    print()

    # --- 모델 학습 ---
    print("--- 모델 학습 ---")
    model, x_train, x_test, y_train, y_test, baseline_acc, model_acc, matrix, digits = train_digit_model()

    print(f"데이터셋: scikit-learn digits (8x8 픽셀 손글씨 숫자, 0~9)")
    print(f"전체 데이터 수: {len(digits.data)}")
    print(f"훈련 데이터 수: {len(x_train)}")
    print(f"테스트 데이터 수: {len(x_test)}")
    print(f"특성 수: {x_train.shape[1]} (8x8 = 64 픽셀)")
    print(f"클래스 수: {len(np.unique(y_train))} (0~9)")
    print()
    print(f"기준선 정확도: {baseline_acc:.3f}")
    print(f"테스트 정확도: {model_acc:.3f}")
    print()
    print("혼동행렬:")
    print(matrix)
    print()

    # --- 데모 앱 실행 ---
    print("--- 데모 앱 (자동 실행 모드) ---")
    print("미리 정해둔 테스트 샘플 10개로 예측을 수행한다.")
    print()

    demo_results = run_demo(model, x_test, y_test)

    correct_count = 0
    wrong_examples = []

    for i, r in enumerate(demo_results, start=1):
        status = "맞힘" if r["correct"] else "틀림"
        if r["correct"]:
            correct_count += 1
        else:
            wrong_examples.append(r)

        print(f"[데모 {i}]")
        print(f"  실제 숫자: {r['true_label']}")
        print(f"  예측 숫자: {r['predicted']}")
        print(f"  결과: {status}")

        # 한계 안내 출력
        notice = format_limitation_notice(r["confidence"])
        print(notice)
        print()

    print(f"--- 데모 결과 요약 ---")
    print(f"총 {len(demo_results)}개 중 {correct_count}개 맞힘, {len(demo_results) - correct_count}개 틀림")
    print()

    # 틀린 사례 분석
    if wrong_examples:
        print("--- 틀린 사례 분석 ---")
        for w in wrong_examples:
            print(f"  실제 {w['true_label']}을 {w['predicted']}로 예측 (신뢰도 {w['confidence']:.1%})")
        print()

    # 결과 CSV 저장
    results_df = pd.DataFrame(demo_results)
    results_path = RESULTS_DIR / "demo_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"데모 결과 CSV 저장: {results_path.relative_to(CHAPTER_DIR)}")

    # --- 모델 한계 종합 안내 ---
    print()
    print("=== 모델 한계 종합 안내 ===")
    print("이 모델은 scikit-learn의 digits 데이터셋으로 학습했다.")
    print("digits 데이터셋은 8x8 픽셀 흑백 이미지이며, 약 1,800장이다.")
    print()
    print("이 모델을 실제 서비스에 쓸 수 없는 이유는 다음과 같다.")
    print("1. 8x8 픽셀은 해상도가 매우 낮다. 실제 손글씨나 인쇄체를 처리할 수 없다.")
    print("2. 데이터가 약 1,800장뿐이다. 다양한 필체를 충분히 학습하지 못했다.")
    print("3. 한 사람이 쓴 글씨체에만 맞춰져 있을 수 있다.")
    print("4. 회전, 기울기, 크기 변화에 대응하지 못한다.")
    print("5. 이 모델은 데모 목적이며, 사용자에게 이 한계를 반드시 알려야 한다.")
    print()
    print("=== 저장된 결과 ===")
    print(f"데모 결과 CSV: {results_path.relative_to(CHAPTER_DIR)}")


if __name__ == "__main__":
    main()
