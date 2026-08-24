"""실습 10.2 - 리뷰 긍정/부정 분류기와 모델이 헷갈리는 문장.

다섯 단계로 진행한다.
  1단계  리뷰 64개를 훈련 44개 + 테스트 20개로 나눈다
  2단계  단어 가방 + 로지스틱 회귀로 학습하고 정확도와 혼동행렬을 낸다
  3단계  일부러 어렵게 만든 문장 8개를 넣어 예측 확률을 본다
  4단계  확률이 0.5 근처인 문장을 모아 그림으로 남긴다
  5단계  단어마다 붙은 가중치를 꺼내, 어떤 단어가 판단을 좌우하는지 확인한다

주의: 여기 쓰는 리뷰는 실제 쇼핑몰 리뷰가 아니라 수업용으로 직접 쓴 합성 문장이다.
인터넷에서 데이터나 모델을 내려받지 않는다. scikit-learn만 쓴다.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import _krfont

# ---------------------------------------------------------------------------
# 학생이 바꾸는 곳 1: 모델을 헷갈리게 만들 문장.
# (문장, 사람이 매긴 정답, 왜 어려운가) 순서로 적는다. 정답은 1이 긍정, 0이 부정이다.
# 자기 문장을 넣고 다시 실행하면 3단계 표와 그림이 바로 바뀐다.
# ---------------------------------------------------------------------------
MY_TRICKY_SENTENCES = [
    ("정말 좋겠다, 배송이 일주일이나 걸려서", 0, "반어"),
    ("나쁘지 않아요", 1, "부정어"),
    ("화면이 선명하지 않아요", 0, "부정어"),
    ("화면은 선명한데 배터리가 최악이에요", 0, "섞인 감정"),
    ("배송은 느렸지만 물건은 만족해요", 1, "섞인 감정"),
    ("가격이 비싸지만 성능이 좋아요", 1, "섞인 감정"),
    ("포장은 꼼꼼한데 배송이 너무 느려요", 0, "섞인 감정"),
    ("가격이 이 정도면 만족해요", 1, "에두른 칭찬"),
]

# ---------------------------------------------------------------------------
# 학생이 바꾸는 곳 2: 실행 설정.
# ---------------------------------------------------------------------------
TEST_RATIO = 0.3        # 테스트로 떼어 둘 비율
NEAR_BAND = 0.15        # 0.5 ± 이 값 안에 들면 "헷갈리는 구간"으로 본다
RANDOM_SEED = 42

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"

# ---------------------------------------------------------------------------
# 학습에 쓸 리뷰. 전부 수업용으로 직접 쓴 합성 문장이다.
# 라벨은 1이 긍정, 0이 부정이다. 긍정 32개, 부정 32개로 수를 맞춰 뒀다.
# 같은 물건 이야기를 긍정과 부정 양쪽에 넣어, 감정을 나타내는 단어에서만
# 두 무리가 갈리도록 만들었다.
# ---------------------------------------------------------------------------
REVIEWS: list[tuple[str, int]] = [
    ("배송이 정말 빨라요", 1),
    ("배송이 빨라서 좋아요", 1),
    ("포장이 꼼꼼해서 좋아요", 1),
    ("화면이 아주 선명해요", 1),
    ("화면이 밝고 선명해요", 1),
    ("소리가 크고 선명해요", 1),
    ("가격이 저렴해서 만족해요", 1),
    ("가격 대비 성능이 좋아요", 1),
    ("배터리가 오래가서 만족해요", 1),
    ("배터리가 하루 종일 넉넉해요", 1),
    ("재질이 튼튼해서 좋아요", 1),
    ("마감이 매끄러워서 만족해요", 1),
    ("사용법이 간단해서 편해요", 1),
    ("무게가 가벼워서 편해요", 1),
    ("손에 잡히는 느낌이 좋아요", 1),
    ("색상이 사진과 똑같아서 만족해요", 1),
    ("고객센터 응대가 친절해요", 1),
    ("설치가 간단해서 편해요", 1),
    ("매일 잘 쓰고 있어요", 1),
    ("다시 사고 싶어요", 1),
    ("친구에게 추천했어요", 1),
    ("값이 저렴해서 추천해요", 1),
    ("성능이 기대보다 좋아요", 1),
    ("속도가 빨라서 만족해요", 1),
    ("화질이 기대보다 선명해요", 1),
    ("소음이 적어서 좋아요", 1),
    ("크기가 딱 맞아서 만족해요", 1),
    ("배송이 하루 만에 왔어요", 1),
    ("잘 산 것 같아요", 1),
    ("두 번째 구매인데 만족해요", 1),
    ("오래 써도 튼튼해요", 1),
    ("디자인이 예뻐서 좋아요", 1),
    ("배송이 너무 느려요", 0),
    ("배송이 느려서 별로예요", 0),
    ("포장이 엉망이라 실망했어요", 0),
    ("화면이 어둡고 흐릿해요", 0),
    ("화면이 자꾸 꺼져요", 0),
    ("소리가 작고 흐릿해요", 0),
    ("가격이 너무 비싸요", 0),
    ("가격 대비 성능이 별로예요", 0),
    ("배터리가 금방 닳아요", 0),
    ("배터리가 반나절도 못 가요", 0),
    ("재질이 약해서 실망했어요", 0),
    ("마감이 거칠어서 최악이에요", 0),
    ("사용법이 복잡해서 불편해요", 0),
    ("무게가 무거워서 불편해요", 0),
    ("손에 잡히는 느낌이 별로예요", 0),
    ("색상이 사진과 달라서 실망했어요", 0),
    ("고객센터 응대가 불친절해요", 0),
    ("설치가 복잡해서 불편해요", 0),
    ("며칠 쓰다가 고장 났어요", 0),
    ("다시는 안 살 거예요", 0),
    ("친구에게 말리고 싶어요", 0),
    ("값이 비싸서 추천 안 해요", 0),
    ("성능이 기대보다 못해요", 0),
    ("속도가 느려서 별로예요", 0),
    ("화질이 기대보다 흐릿해요", 0),
    ("소음이 심해서 별로예요", 0),
    ("크기가 안 맞아서 실망했어요", 0),
    ("배송이 일주일이나 걸렸어요", 0),
    ("괜히 산 것 같아요", 0),
    ("두 번째 구매인데 실망했어요", 0),
    ("조금 쓰다가 부서졌어요", 0),
    ("디자인이 촌스러워서 별로예요", 0),
]


def tokenize(text: str) -> list[str]:
    """문장을 토큰으로 쪼갠다. 실습 10.1과 같은 규칙이다.

    문장부호를 공백으로 바꾸고 공백으로 자른다.
    한국어에서 이 규칙은 조사와 어미를 단어에 붙인 채로 남긴다.
    그래서 "화면이"와 "화면은"이 서로 다른 단어로 잡힌다.
    이 실습에서는 형태소 분석기를 설치하지 않는다. 붙어 있는 그대로 두고
    무슨 일이 생기는지 3단계에서 확인한다.
    """
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return [token for token in cleaned.split() if token]


def draw_confusion(ax, matrix: np.ndarray, title: str, ko: bool) -> None:
    """혼동행렬 한 장을 그린다. 맞힌 칸은 초록, 틀린 칸은 빨강으로 칠한다."""
    L = _krfont.label
    colors = np.array([[GREEN, RED], [RED, GREEN]])

    for row in range(2):
        for col in range(2):
            ax.add_patch(plt.Rectangle((col, 1 - row), 1, 1,
                                       facecolor=colors[row, col], alpha=0.85))
            ax.text(col + 0.5, 1.5 - row, str(matrix[row, col]),
                    ha="center", va="center", fontsize=26, color="white")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([1.5, 0.5])
    ax.set_xticklabels([L("예측: 부정", "pred: negative", ko),
                        L("예측: 긍정", "pred: positive", ko)], fontsize=11)
    ax.set_yticklabels([L("정답: 부정", "true: negative", ko),
                        L("정답: 긍정", "true: positive", ko)], fontsize=11)
    ax.set_title(title, fontsize=12.5, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def draw_diverging(ax, labels, probs, correct, ko: bool, title: str) -> None:
    """0.5를 기준으로 좌우로 뻗는 막대를 그린다.

    막대가 길수록 모델이 한쪽으로 기운 것이고, 짧을수록 0.5 근처다.
    사람이 매긴 정답과 예측이 같으면 초록, 다르면 빨강으로 칠한다.
    """
    L = _krfont.label
    y = np.arange(len(labels))

    ax.axvspan(0.5 - NEAR_BAND, 0.5 + NEAR_BAND, color="#F2F2F2", zorder=0)
    ax.axvline(0.5, color=GRAY, linewidth=1.4, zorder=1)

    for i, (p, ok) in enumerate(zip(probs, correct)):
        ax.barh(y[i], p - 0.5, left=0.5, height=0.62,
                color=GREEN if ok else RED, zorder=2)
        offset = 0.012 if p >= 0.5 else -0.012
        ax.text(p + offset, y[i], f"{p:.2f}",
                ha="left" if p >= 0.5 else "right", va="center",
                fontsize=9.5, color="#333333", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(L("모델이 매긴 긍정 확률", "model probability of positive", ko),
                  fontsize=10.5)
    ax.set_title(title, fontsize=12, pad=10, loc="left")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    L = _krfont.label

    sentences = [text for text, _ in REVIEWS]
    labels = np.array([label for _, label in REVIEWS])

    print("=== 실습 10.2 리뷰 긍정/부정 분류기 ===")
    _krfont.report(ko)
    print("데이터: 수업용으로 직접 쓴 합성 리뷰(실제 쇼핑몰 리뷰 아님)")
    print(f"리뷰 {len(sentences)}개 = 긍정 {int(labels.sum())}개 + "
          f"부정 {int((labels == 0).sum())}개")
    print()

    # -----------------------------------------------------------------
    # 1단계. 훈련과 테스트로 나눈다
    # -----------------------------------------------------------------
    train_text, test_text, y_train, y_test = train_test_split(
        sentences, labels, test_size=TEST_RATIO,
        random_state=RANDOM_SEED, stratify=labels,
    )

    print("=== 1단계. 훈련과 테스트로 나눈다 ===")
    print(f"훈련 {len(train_text)}개, 테스트 {len(test_text)}개"
          f"(테스트 안의 긍정 {int(y_test.sum())}개)")
    print()

    # -----------------------------------------------------------------
    # 2단계. 단어 가방 + 로지스틱 회귀
    # -----------------------------------------------------------------
    vectorizer = CountVectorizer(
        tokenizer=tokenize, token_pattern=None, lowercase=False, binary=True,
    )
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    vocab = vectorizer.get_feature_names_out()

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(x_train, y_train)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    test_prob = model.predict_proba(x_test)[:, 1]
    base_acc = accuracy_score(y_test, baseline.predict(x_test))
    matrix = confusion_matrix(y_test, test_pred, labels=[0, 1])

    print("=== 2단계. 단어 가방으로 학습한다 ===")
    print(f"훈련 문장 {len(train_text)}개에서 서로 다른 단어 {len(vocab)}개가 나왔다.")
    print(f"입력 표의 크기: {x_train.shape[0]}행 x {x_train.shape[1]}열")
    print()
    print(f"기준선(무조건 많은 쪽으로 찍기) 테스트 정확도: {base_acc:.3f}")
    print(f"훈련 데이터 정확도: {accuracy_score(y_train, train_pred):.3f}")
    print(f"테스트 데이터 정확도: {accuracy_score(y_test, test_pred):.3f}")
    print()
    print("--- 테스트 데이터 혼동행렬 ---")
    print("               예측:부정  예측:긍정")
    print(f"  정답:부정  {matrix[0, 0]:>8}  {matrix[0, 1]:>8}")
    print(f"  정답:긍정  {matrix[1, 0]:>8}  {matrix[1, 1]:>8}")
    print()

    result_table = pd.DataFrame({
        "문장": test_text,
        "정답": ["긍정" if v else "부정" for v in y_test],
        "모델예측": ["긍정" if v else "부정" for v in test_pred],
        "긍정확률": test_prob.round(2),
        "맞음": ["O" if a == b else "X" for a, b in zip(y_test, test_pred)],
    })
    print("--- 테스트 문장 20개 전체 결과 ---")
    print(result_table.to_string(index=False))
    print()

    wrong = result_table[result_table["맞음"] == "X"]
    print(f"테스트에서 틀린 문장: {len(wrong)}개")
    if len(wrong):
        print(wrong.to_string(index=False))
    print()

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    draw_confusion(
        ax, matrix,
        L(f"테스트 {len(test_text)}개 · 정확도 {accuracy_score(y_test, test_pred):.3f}",
          f"test n={len(test_text)} · acc {accuracy_score(y_test, test_pred):.3f}", ko),
        ko,
    )
    fig.suptitle(
        L("리뷰 감성 분류 혼동행렬 (합성 데이터)",
          "Sentiment classification confusion matrix (synthetic data)", ko),
        fontsize=13.5,
    )
    fig.tight_layout()
    confusion_path = RESULTS_DIR / "sentiment_confusion.png"
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)

    result_table.to_csv(RESULTS_DIR / "sentiment_results.csv",
                        index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------
    # 3단계. 일부러 어렵게 만든 문장
    # -----------------------------------------------------------------
    tricky_text = [t for t, _, _ in MY_TRICKY_SENTENCES]
    tricky_label = np.array([v for _, v, _ in MY_TRICKY_SENTENCES])
    tricky_kind = [k for _, _, k in MY_TRICKY_SENTENCES]

    x_tricky = vectorizer.transform(tricky_text)
    tricky_pred = model.predict(x_tricky)
    tricky_prob = model.predict_proba(x_tricky)[:, 1]

    known = set(vocab)
    unknown_lists = [[tok for tok in tokenize(t) if tok not in known]
                     for t in tricky_text]

    tricky_table = pd.DataFrame({
        "유형": tricky_kind,
        "문장": tricky_text,
        "사람정답": ["긍정" if v else "부정" for v in tricky_label],
        "모델예측": ["긍정" if v else "부정" for v in tricky_pred],
        "긍정확률": tricky_prob.round(2),
        "맞음": ["O" if a == b else "X" for a, b in zip(tricky_label, tricky_pred)],
        "모르는단어": [len(u) for u in unknown_lists],
    })

    print("=== 3단계. 모델이 헷갈리는 문장 ===")
    print("아래 8개는 학습에 쓰지 않은 문장이다. 사람이 정답을 직접 매겼다.")
    print(tricky_table.to_string(index=False))
    print()
    tricky_wrong = int((tricky_table["맞음"] == "X").sum())
    print(f"8개 중 틀린 것: {tricky_wrong}개 "
          f"(테스트 리뷰에서는 {len(test_text)}개 중 {len(wrong)}개 틀렸다)")
    print()

    print("--- 문장마다 모델이 아는 단어와 모르는 단어 ---")
    weights = model.coef_[0]
    weight_of = {word: float(w) for word, w in zip(vocab, weights)}
    for i, text in enumerate(tricky_text):
        tokens = tokenize(text)
        seen = [tok for tok in tokens if tok in known]
        unseen = unknown_lists[i]
        print(f"[{tricky_kind[i]}] {text}")
        if seen:
            parts = [f"{tok}({weight_of[tok]:+.2f})"
                     for tok in sorted(seen, key=lambda t: -abs(weight_of[t]))]
            print(f"    아는 단어 {len(seen)}개: {', '.join(parts)}")
        else:
            print("    아는 단어 0개")
        if unseen:
            print(f"    모르는 단어 {len(unseen)}개: {', '.join(unseen)}")
        print(f"    -> 긍정 확률 {tricky_prob[i]:.2f}"
              f"  (사람 정답 {'긍정' if tricky_label[i] else '부정'})")
        print()

    tricky_table.to_csv(RESULTS_DIR / "confusing_sentences.csv",
                        index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------
    # 4단계. 확률이 0.5 근처인 문장을 모은다
    # -----------------------------------------------------------------
    all_text = sentences
    all_prob = model.predict_proba(vectorizer.transform(all_text))[:, 1]
    near_idx = [i for i in range(len(all_text))
                if abs(all_prob[i] - 0.5) <= NEAR_BAND]
    near_idx.sort(key=lambda i: abs(all_prob[i] - 0.5))

    print("=== 4단계. 확률이 0.5 근처인 문장 ===")
    low, high = 0.5 - NEAR_BAND, 0.5 + NEAR_BAND
    print(f"긍정 확률이 {low:.2f}와 {high:.2f} 사이면 헷갈리는 구간으로 본다.")
    print(f"헷갈리게 만든 문장 8개 중 이 구간에 든 것: "
          f"{int((np.abs(tricky_prob - 0.5) <= NEAR_BAND).sum())}개")
    print(f"리뷰 {len(all_text)}개(훈련 + 테스트) 중 이 구간에 든 것: {len(near_idx)}개")
    if near_idx:
        near_table = pd.DataFrame({
            "문장": [all_text[i] for i in near_idx],
            "정답": ["긍정" if labels[i] else "부정" for i in near_idx],
            "긍정확률": [round(float(all_prob[i]), 2) for i in near_idx],
        })
        print(near_table.to_string(index=False))
    print()

    n_top = len(tricky_text)
    n_bottom = max(len(near_idx), 1)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(11.5, 3.0 + 0.42 * (n_top + n_bottom)),
        gridspec_kw={"height_ratios": [n_top, n_bottom]},
    )

    top_labels = ([f"[{k}] {t}" for k, t in zip(tricky_kind, tricky_text)]
                  if ko else [f"tricky #{i + 1}" for i in range(n_top)])
    draw_diverging(
        ax_top, top_labels, tricky_prob,
        [a == b for a, b in zip(tricky_label, tricky_pred)], ko,
        L("일부러 어렵게 만든 문장 8개",
          "eight deliberately hard sentences", ko),
    )

    if near_idx:
        bottom_labels = ([all_text[i] for i in near_idx]
                         if ko else [f"review #{i + 1}" for i in near_idx])
        draw_diverging(
            ax_bottom, bottom_labels,
            np.array([all_prob[i] for i in near_idx]),
            [(all_prob[i] >= 0.5) == bool(labels[i]) for i in near_idx], ko,
            L(f"리뷰 {len(all_text)}개 중 확률이 0.5 근처인 문장 {len(near_idx)}개",
              f"{len(near_idx)} of {len(all_text)} reviews with probability near 0.5", ko),
        )
    else:
        ax_bottom.axis("off")
        ax_bottom.text(
            0.5, 0.5,
            L("리뷰 중에는 0.5 근처인 문장이 없다.",
              "No review lands near 0.5.", ko),
            ha="center", va="center", fontsize=11.5, color=GRAY,
        )

    fig.suptitle(
        L("초록은 사람 정답과 같은 예측, 빨강은 다른 예측 · 회색 띠가 헷갈리는 구간 (합성 데이터)",
          "Green = matches the human label, red = does not; the grey band is the uncertain zone",
          ko),
        fontsize=12.5, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    confusing_path = RESULTS_DIR / "confusing_sentences.png"
    fig.savefig(confusing_path, dpi=160)
    plt.close(fig)

    # -----------------------------------------------------------------
    # 5단계. 단어마다 붙은 가중치
    # -----------------------------------------------------------------
    x_train_dense = x_train.toarray()
    pos_count = x_train_dense[y_train == 1].sum(axis=0)
    neg_count = x_train_dense[y_train == 0].sum(axis=0)
    one_sided = (pos_count == 0) | (neg_count == 0)

    weight_table = pd.DataFrame({
        "단어": vocab,
        "가중치": weights.round(3),
        "긍정문장수": pos_count.astype(int),
        "부정문장수": neg_count.astype(int),
        "한쪽에만": np.where(one_sided, "예", "아니오"),
    }).sort_values("가중치", ascending=False).reset_index(drop=True)

    print("=== 5단계. 단어마다 붙은 가중치 ===")
    print("가중치가 클수록 그 단어가 있으면 긍정 쪽으로, 작을수록 부정 쪽으로 민다.")
    print(f"훈련 문장에 나온 단어 {len(vocab)}개 중 "
          f"한쪽 라벨에만 나온 단어: {int(one_sided.sum())}개 "
          f"({one_sided.mean() * 100:.1f}%)")
    print()
    print("--- 긍정 쪽으로 가장 세게 미는 단어 10개 ---")
    print(weight_table.head(10).to_string(index=False))
    print()
    print("--- 부정 쪽으로 가장 세게 미는 단어 10개 ---")
    print(weight_table.tail(10).iloc[::-1].to_string(index=False))
    print()

    top20 = pd.concat([weight_table.head(10), weight_table.tail(10)])
    print(f"가중치 상위 10개와 하위 10개, 모두 20개 중 "
          f"한쪽 라벨에만 나온 단어: {int((top20['한쪽에만'] == '예').sum())}개")
    print()

    weight_table.to_csv(RESULTS_DIR / "word_weights.csv",
                        index=False, encoding="utf-8-sig")

    top_pos = weight_table.head(8).iloc[::-1]
    top_neg = weight_table.tail(8)
    picked = pd.concat([top_neg, top_pos])

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    y = np.arange(len(picked))
    for i, (_, row) in enumerate(picked.iterrows()):
        is_one_sided = row["한쪽에만"] == "예"
        color = GREEN if row["가중치"] >= 0 else RED
        ax.barh(y[i], row["가중치"], height=0.68,
                color=color if is_one_sided else "white",
                edgecolor=color, linewidth=1.6,
                hatch="" if is_one_sided else "//")
        offset = 0.03 if row["가중치"] >= 0 else -0.03
        ax.text(row["가중치"] + offset, y[i],
                f"{row['긍정문장수']} / {row['부정문장수']}",
                ha="left" if row["가중치"] >= 0 else "right",
                va="center", fontsize=9, color="#555555")

    ax.axvline(0, color=GRAY, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(
        list(picked["단어"]) if ko else [f"word {i + 1}" for i in range(len(picked))],
        fontsize=10.5,
    )
    ax.set_xlabel(L("가중치 (오른쪽이 긍정, 왼쪽이 부정)",
                    "weight (right = positive, left = negative)", ko), fontsize=10.5)
    ax.set_title(
        L("판단을 좌우하는 단어 16개 — 숫자는 그 단어가 나온 (긍정 문장 수 / 부정 문장 수)",
          "16 words that drive the decision - numbers are (positive / negative) counts", ko),
        fontsize=12.5, pad=12, loc="left",
    )
    ax.margins(x=0.18)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GRAY, edgecolor=GRAY),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRAY, hatch="//"),
    ]
    ax.legend(
        handles,
        [L("훈련 문장에서 한쪽 라벨에만 나온 단어",
           "appears in only one label", ko),
         L("양쪽 라벨에 모두 나온 단어",
           "appears in both labels", ko)],
        loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False,
    )
    fig.tight_layout()
    weights_path = RESULTS_DIR / "word_weights.png"
    fig.savefig(weights_path, dpi=160)
    plt.close(fig)

    # -----------------------------------------------------------------
    print("=== 저장된 결과 ===")
    print(f"혼동행렬 그림: results/{confusion_path.name}")
    print(f"헷갈리는 문장 그림: results/{confusing_path.name}")
    print(f"단어 가중치 그림: results/{weights_path.name}")
    print("감성 분석 결과표: results/sentiment_results.csv")
    print("헷갈리는 문장 표: results/confusing_sentences.csv")
    print("단어 가중치 표: results/word_weights.csv")
    print()
    print("=== 정리 ===")
    print("모델은 뜻을 읽지 않는다. 훈련 문장에서 어느 쪽에 나왔는지만 센다.")
    print("한쪽 라벨에만 나온 단어가 판단을 좌우하고, 못 본 단어는 판단에 끼지 못한다.")


if __name__ == "__main__":
    main()
