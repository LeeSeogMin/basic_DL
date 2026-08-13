"""10주차 실습: 텍스트 감성 분류기

짧은 리뷰 문장을 긍정/부정으로 분류한다.
CountVectorizer(Bag-of-Words)와 TfidfVectorizer를 비교하고,
모델이 헷갈린 문장 5개를 찾아 출력한다.

데이터: 코드 안에 직접 정의한 합성 리뷰 50개 (영어).
       한국어 리뷰도 일부 포함하여 혼합 데이터의 한계를 보여준다.
모델: scikit-learn LogisticRegression (CPU 즉시 실행).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ---------- 설정 ----------
RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

# matplotlib 한글 폰트 설정 (macOS: AppleGothic, 없으면 기본 폰트)
matplotlib.rcParams["axes.unicode_minus"] = False
for _font in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = _font
        break
    except ValueError:
        continue

# ---------- 합성 데이터 ----------
# 긍정(1)과 부정(0) 리뷰 50개. 일부는 일부러 애매하게 작성했다.
REVIEWS: list[tuple[str, int]] = [
    # ---- 명확한 긍정 (1) ----
    ("This movie was absolutely amazing and fun", 1),
    ("I loved every minute of the show", 1),
    ("Great food and excellent service at this restaurant", 1),
    ("Best experience I have ever had", 1),
    ("The staff was friendly and very helpful", 1),
    ("Really enjoyed the atmosphere and the music", 1),
    ("Highly recommend this place to everyone", 1),
    ("The quality exceeded my expectations completely", 1),
    ("Perfect in every way I can think of", 1),
    ("Wonderful experience from start to finish", 1),
    ("Very satisfied with the product quality", 1),
    ("The view was breathtaking and beautiful", 1),
    ("Fantastic customer support and fast delivery", 1),
    ("Everything was clean and well organized", 1),
    ("I will definitely come back again soon", 1),
    ("Loved the design and easy to use interface", 1),
    ("The best purchase I made this year", 1),
    ("Tasty food and generous portions", 1),
    ("Smooth experience with no problems at all", 1),
    ("Exceeded all my expectations", 1),
    # ---- 명확한 부정 (0) ----
    ("Terrible service and cold food", 0),
    ("I would never come back here again", 0),
    ("Complete waste of time and money", 0),
    ("The worst experience of my life", 0),
    ("Very disappointing and way overpriced", 0),
    ("Rude staff and dirty tables everywhere", 0),
    ("Nothing good about this place at all", 0),
    ("The product broke after just one day", 0),
    ("Waited two hours and the food was awful", 0),
    ("Horrible quality for such a high price", 0),
    ("The room was noisy and uncomfortable", 0),
    ("I regret buying this product", 0),
    ("Support team was unhelpful and slow", 0),
    ("The app crashes every time I open it", 0),
    ("Delivery was late and the item was damaged", 0),
    ("Could not be worse than this", 0),
    ("Do not recommend to anyone", 0),
    ("Absolutely terrible and frustrating", 0),
    ("Cheap quality and poor packaging", 0),
    ("A total disaster from beginning to end", 0),
    # ---- 애매한 문장 (모델이 헷갈릴 수 있다) ----
    ("Not bad but not great either", 1),                       # 미묘한 긍정
    ("It was okay I guess nothing special", 0),                # 미묘한 부정
    ("The food was fine but the wait was too long", 0),        # 혼합
    ("Good product but terrible customer service", 0),         # 혼합
    ("Beautiful location but the food was mediocre", 0),       # 혼합
    ("Average experience nothing to complain about", 1),       # 미묘한 긍정
    ("The price was right but quality could be better", 0),    # 혼합
    ("Nice staff but the room was too small", 0),              # 혼합
    ("I expected more but it was still decent", 1),            # 미묘한 긍정
    ("Would not go again but it was not terrible", 0),         # 미묘한 부정
]


def build_data() -> tuple[list[str], np.ndarray]:
    """리뷰 텍스트와 라벨을 분리한다."""
    texts = [r[0] for r in REVIEWS]
    labels = np.array([r[1] for r in REVIEWS])
    return texts, labels


def train_and_evaluate(
    vectorizer_name: str,
    vectorizer,
    x_train_text: list[str],
    x_test_text: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[LogisticRegression, np.ndarray, float, np.ndarray]:
    """벡터화 -> 학습 -> 평가를 수행하고 결과를 반환한다."""
    x_train_vec = vectorizer.fit_transform(x_train_text)
    x_test_vec = vectorizer.transform(x_test_text)

    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    model.fit(x_train_vec, y_train)

    pred = model.predict(x_test_vec)
    acc = accuracy_score(y_test, pred)
    matrix = confusion_matrix(y_test, pred)

    return model, pred, acc, matrix


def find_confused_sentences(
    x_test_text: list[str],
    y_test: np.ndarray,
    pred: np.ndarray,
    model,
    vectorizer,
    top_n: int = 5,
) -> pd.DataFrame:
    """모델이 헷갈린 문장을 찾는다.

    틀린 문장이 top_n보다 적으면, 예측 확률이 0.5에 가까운
    (즉 모델이 확신하지 못한) 문장을 추가한다.
    """
    x_test_vec = vectorizer.transform(x_test_text)
    probas = model.predict_proba(x_test_vec)[:, 1]  # 긍정 확률

    rows: list[dict] = []

    # 1단계: 실제로 틀린 문장
    wrong_mask = pred != y_test
    wrong_indices = np.where(wrong_mask)[0]
    for idx in wrong_indices:
        rows.append({
            "sentence": x_test_text[idx],
            "true_label": int(y_test[idx]),
            "predicted_label": int(pred[idx]),
            "positive_prob": round(float(probas[idx]), 3),
            "reason": "wrong",
        })

    # 2단계: 틀린 문장이 top_n보다 적으면 확신이 낮은 문장을 추가
    if len(rows) < top_n:
        uncertainty = np.abs(probas - 0.5)
        # 이미 틀린 문장은 제외
        uncertainty[wrong_indices] = 999.0
        uncertain_order = np.argsort(uncertainty)
        for idx in uncertain_order:
            if len(rows) >= top_n:
                break
            if uncertainty[idx] >= 999.0:
                continue
            rows.append({
                "sentence": x_test_text[idx],
                "true_label": int(y_test[idx]),
                "predicted_label": int(pred[idx]),
                "positive_prob": round(float(probas[idx]), 3),
                "reason": "uncertain",
            })

    return pd.DataFrame(rows[:top_n])


def save_confusion_matrix(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """혼동행렬 그림을 저장한다."""
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["neg", "pos"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    texts, labels = build_data()

    # 데이터 분할
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    print("=== 10주차 텍스트 감성 분류기 ===")
    print(f"전체 리뷰 수: {len(texts)}")
    print(f"  긍정: {int(np.sum(labels == 1))}개")
    print(f"  부정: {int(np.sum(labels == 0))}개")
    print(f"훈련 데이터 수: {len(x_train_text)}")
    print(f"테스트 데이터 수: {len(x_test_text)}")
    print()

    # --- CountVectorizer (Bag-of-Words) ---
    bow_vectorizer = CountVectorizer()
    bow_model, bow_pred, bow_acc, bow_matrix = train_and_evaluate(
        "CountVectorizer (BoW)",
        bow_vectorizer,
        x_train_text,
        x_test_text,
        y_train,
        y_test,
    )

    print("=== CountVectorizer (Bag-of-Words) ===")
    print(f"어휘 크기: {len(bow_vectorizer.vocabulary_)}")
    print(f"테스트 정확도: {bow_acc:.3f}")
    print("혼동행렬:")
    print(bow_matrix)
    print()

    # --- TfidfVectorizer ---
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_model, tfidf_pred, tfidf_acc, tfidf_matrix = train_and_evaluate(
        "TfidfVectorizer (TF-IDF)",
        tfidf_vectorizer,
        x_train_text,
        x_test_text,
        y_train,
        y_test,
    )

    print("=== TfidfVectorizer (TF-IDF) ===")
    print(f"어휘 크기: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"테스트 정확도: {tfidf_acc:.3f}")
    print("혼동행렬:")
    print(tfidf_matrix)
    print()

    # --- 더 나은 모델로 헷갈린 문장 분석 ---
    if tfidf_acc >= bow_acc:
        best_name = "TF-IDF"
        best_model = tfidf_model
        best_vectorizer = tfidf_vectorizer
        best_pred = tfidf_pred
        best_acc = tfidf_acc
        best_matrix = tfidf_matrix
    else:
        best_name = "BoW"
        best_model = bow_model
        best_vectorizer = bow_vectorizer
        best_pred = bow_pred
        best_acc = bow_acc
        best_matrix = bow_matrix

    print(f"=== 선택 모델: {best_name} (정확도 {best_acc:.3f}) ===")
    print()

    confused_df = find_confused_sentences(
        x_test_text,
        y_test,
        best_pred,
        best_model,
        best_vectorizer,
        top_n=5,
    )

    print("=== 모델이 헷갈린 문장 (최대 5개) ===")
    if confused_df.empty:
        print("헷갈린 문장이 없습니다.")
    else:
        label_map = {0: "부정", 1: "긍정"}
        for i, row in confused_df.iterrows():
            tag = "틀림" if row["reason"] == "wrong" else "확신 낮음"
            print(
                f"  [{tag}] \"{row['sentence']}\"\n"
                f"    실제: {label_map[row['true_label']]}, "
                f"예측: {label_map[row['predicted_label']]}, "
                f"긍정 확률: {row['positive_prob']}"
            )
        print()

    # --- 결과 저장 ---
    confused_path = RESULTS_DIR / "confused_sentences.csv"
    confused_df.to_csv(confused_path, index=False)

    confusion_path = RESULTS_DIR / "confusion_matrix.png"
    save_confusion_matrix(
        best_matrix,
        f"Confusion Matrix ({best_name})",
        confusion_path,
    )

    print("=== 저장된 결과 ===")
    print(f"혼동행렬 그림: {confusion_path.relative_to(CHAPTER_DIR)}")
    print(f"헷갈린 문장 CSV: {confused_path.relative_to(CHAPTER_DIR)}")
    print()

    # --- 비교 요약 ---
    print("=== BoW vs TF-IDF 비교 ===")
    print(f"  CountVectorizer (BoW)  정확도: {bow_acc:.3f}")
    print(f"  TfidfVectorizer (TF-IDF) 정확도: {tfidf_acc:.3f}")
    print()

    print("=== 해석 주의 ===")
    print("이 실습의 데이터는 50개뿐인 합성 데이터다.")
    print("데이터가 적으므로 정확도 수치 자체보다")
    print("모델이 어떤 문장을 헷갈리는지, 왜 헷갈리는지가 더 중요하다.")
    print("데이터에 특정 표현이 편중되어 있으면 모델도 그 편향을 학습한다.")


if __name__ == "__main__":
    main()
