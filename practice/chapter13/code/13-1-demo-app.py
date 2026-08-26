"""실습 13.1 - 분류기를 남이 쓸 수 있는 데모 앱으로 감싼다.

4장에서 만든 것과 같은 종류의 스팸 분류기를 학습시킨다.
다른 점은 학습 뒤에 하는 일이다. 정확도 한 줄을 찍고 끝내지 않고,
사용자가 넣은 입력을 한 건씩 받아 다음 다섯 가지를 화면에 쓴다.

  1. 무엇을 입력했는지
  2. 입력이 받을 수 있는 값인지 (아니면 받지 않고 이유를 말한다)
  3. 예측 이름 (스팸 / 정상)
  4. 예측 확률
  5. 판단에 쓴 값과 그 값이 어느 쪽으로 얼마나 밀었는지

마지막에 "이 모델이 틀릴 수 있는 경우"를 화면에 적는다.

주의: 이 데이터는 실제 메일이 아니라 컴퓨터로 만든 합성 데이터다.
설치가 필요한 웹 프레임워크(gradio, streamlit)를 쓰지 않는다.
화면은 콘솔 출력이고, 같은 내용을 카드 그림으로도 남긴다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.
# 앱에 넣을 입력을 한 줄에 한 건씩 적는다.
#   (설명, 링크 개수, 대문자 비율, 느낌표 개수)
# 링크 개수와 느낌표 개수는 0 이상, 대문자 비율은 0.0 ~ 1.0 이어야 한다.
MY_INPUTS = [
    ("광고처럼 보이는 메일", 6, 0.32, 3),
    ("교수님이 보낸 공지", 1, 0.06, 0),
    ("링크는 많지만 조용한 메일", 7, 0.09, 0),
    ("애매한 메일", 4, 0.20, 1),
    ("동아리 모집 메일", 3, 0.28, 3),
    ("잘못 입력한 예", 2, 1.40, 1),   # 대문자 비율이 1을 넘는다 -> 받지 않는다
]

# 화면에 함께 적을 한계 안내문. 학생이 자기 앱에 맞게 고쳐 쓴다.
MODEL_LIMITS = [
    "합성 데이터로만 배웠다. 실제 받은 편지함에서 시험한 적이 없다.",
    "한국어 메일 본문을 보지 않는다. 링크 수, 대문자 비율, 느낌표 수만 본다.",
    "확률이 0.5 근처인 메일은 자주 틀린다.",
]
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_MAILS = 1200
SPAM_RATIO = 0.4

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

FEATURE_NAMES = ["링크 개수", "대문자 비율", "느낌표 개수"]
FEATURE_NAMES_EN = ["links", "caps ratio", "exclam."]
FEATURE_PADDED = ["링크 개수  ", "대문자 비율", "느낌표 개수"]

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def make_synthetic_mails() -> tuple[np.ndarray, np.ndarray]:
    """합성 메일 데이터를 만든다.

    특징은 세 개다. 링크 개수, 제목의 대문자 비율, 느낌표 개수.
    스팸이 세 값 모두 큰 편이지만 정상 메일과 겹치는 구간을 넉넉히 남겼다.
    겹치는 구간이 있어야 확률이 0.5 근처인 메일이 생긴다.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_spam = int(N_MAILS * SPAM_RATIO)
    n_normal = N_MAILS - n_spam

    normal = np.column_stack([
        np.clip(rng.normal(2.5, 2.0, n_normal), 0, None),
        np.clip(rng.normal(0.12, 0.09, n_normal), 0, 1),
        rng.poisson(0.8, n_normal).astype(float),
    ])
    spam = np.column_stack([
        np.clip(rng.normal(5.0, 2.6, n_spam), 0, None),
        np.clip(rng.normal(0.24, 0.11, n_spam), 0, 1),
        rng.poisson(2.0, n_spam).astype(float),
    ])

    x = np.vstack([normal, spam])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_spam, dtype=int)])

    order = rng.permutation(len(y))
    return x[order], y[order]


def check_input(links: float, caps: float, excl: float) -> str | None:
    """입력을 검사한다. 받을 수 있으면 None, 아니면 거절 이유를 돌려준다."""
    if links < 0:
        return "링크 개수는 0 이상이어야 한다"
    if not 0.0 <= caps <= 1.0:
        return "대문자 비율은 0.0과 1.0 사이여야 한다"
    if excl < 0:
        return "느낌표 개수는 0 이상이어야 한다"
    return None


def explain_one(model, values: list[float]) -> dict:
    """입력 한 건을 예측하고, 각 값이 어느 쪽으로 얼마나 밀었는지 계산한다.

    표준화한 값에 무게를 곱한 것이 그 값이 민 크기다.
    세 개를 다 더하고 절편을 더하면 로그오즈가 되고, 그것이 확률로 바뀐다.
    양수는 스팸 쪽, 음수는 정상 쪽으로 민 것이다.
    """
    scaler: StandardScaler = model[0]
    clf: LogisticRegression = model[1]

    x_one = np.array([values], dtype=float)
    z = scaler.transform(x_one)[0]
    push = clf.coef_[0] * z
    prob = float(model.predict_proba(x_one)[0, 1])

    return {"prob": prob, "push": push}


def draw_card(ax, item: dict, push_scale: float, ko: bool) -> None:
    """카드 한 장을 그린다. 화면 한 칸에 들어가는 내용과 같다."""
    L = _krfont.label
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    rejected = item["reason"] is not None
    if rejected:
        edge, fill, head = GRAY, "#F0EFEE", GRAY
    elif item["pred"] == 1:
        edge, fill, head = RED, "#FBEAEA", RED
    else:
        edge, fill, head = GREEN, "#EAF3E7", GREEN

    ax.add_patch(FancyBboxPatch(
        (0.2, 0.2), 9.6, 9.4,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        facecolor=fill, edgecolor=edge, linewidth=1.8,
    ))

    ax.text(0.7, 8.95, item["title"], ha="left", va="center",
            fontsize=11.5, color="#333333")

    # 머리띠 - 결론 한 줄
    ax.add_patch(FancyBboxPatch(
        (0.7, 7.55), 8.6, 1.0,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=head, edgecolor="none",
    ))
    if rejected:
        head_text = L("입력을 받지 않았다", "input rejected", ko)
    else:
        head_text = L(
            f"{item['pred_name_ko']} · 스팸일 확률 {item['prob']:.2f}",
            f"{item['pred_name_en']} · spam prob {item['prob']:.2f}", ko,
        )
    ax.text(5.0, 8.05, head_text, ha="center", va="center",
            fontsize=12, color="white")

    # 입력값 세 줄
    names = FEATURE_NAMES if ko else FEATURE_NAMES_EN
    ax.text(0.7, 6.95, L("넣은 값", "input values", ko),
            ha="left", va="center", fontsize=10, color=GRAY)
    for i, (name, value) in enumerate(zip(names, item["values"])):
        fmt = f"{value:.2f}" if i == 1 else f"{value:.0f}"
        ax.text(0.9, 6.3 - i * 0.6, f"{name}", ha="left", va="center",
                fontsize=10.5, color="#333333")
        ax.text(4.6, 6.3 - i * 0.6, fmt, ha="right", va="center",
                fontsize=10.5, color="#333333")

    if rejected:
        ax.text(5.0, 3.0, L(f"이유: {item['reason']}",
                            "reason: value out of allowed range", ko),
                ha="center", va="center", fontsize=10.5, color=GRAY,
                wrap=True)
        ax.text(5.0, 2.1, L("앱은 예측하지 않고 다시 입력을 받는다",
                            "the app asks for the input again", ko),
                ha="center", va="center", fontsize=10, color=GRAY)
        return

    # 판단 근거 - 각 값이 민 방향과 크기
    ax.text(5.3, 6.95, L("이 값이 민 방향", "how each value pushed", ko),
            ha="left", va="center", fontsize=10, color=GRAY)
    center = 7.4
    half = 1.9
    ax.plot([center, center], [3.9, 6.65], color=GRAY, linewidth=0.9, zorder=2)
    for i, push in enumerate(item["push"]):
        y = 6.3 - i * 0.6
        width = half * float(push) / push_scale
        color = RED if push > 0 else BLUE
        ax.add_patch(plt.Rectangle((center, y - 0.16), width, 0.32,
                                   facecolor=color, edgecolor="none", zorder=3))

    ax.text(center - half, 3.6, L("정상 쪽", "toward normal", ko),
            ha="center", va="center", fontsize=8.5, color=BLUE)
    ax.text(center + half, 3.6, L("스팸 쪽", "toward spam", ko),
            ha="center", va="center", fontsize=8.5, color=RED)

    # 확률 막대
    ax.text(0.7, 3.05, L("스팸일 확률", "spam probability", ko),
            ha="left", va="center", fontsize=10, color=GRAY)
    ax.add_patch(plt.Rectangle((0.7, 2.25), 8.6, 0.5,
                               facecolor="#DDDBDA", edgecolor="none"))
    ax.add_patch(plt.Rectangle((0.7, 2.25), 8.6 * item["prob"], 0.5,
                               facecolor=head, edgecolor="none"))
    ax.plot([0.7 + 4.3, 0.7 + 4.3], [2.15, 2.85], color="#333333",
            linewidth=1.2, linestyle="--")
    ax.text(5.0, 1.85, L("가운데 점선이 0.5", "dashed line = 0.5", ko),
            ha="center", va="center", fontsize=8.5, color=GRAY)

    ax.text(0.7, 1.05, L(f"가장 크게 민 값: {item['top_reason_ko']}",
                         f"largest push: {item['top_reason_en']}", ko),
            ha="left", va="center", fontsize=9.5, color="#333333")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    L = _krfont.label

    x, y = make_synthetic_mails()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=RANDOM_SEED),
    )
    model.fit(x_train, y_train)
    test_acc = float(accuracy_score(y_test, model.predict(x_test)))

    print("=== 실습 13.1 데모 앱 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 메일 데이터(실제 메일 아님)")
    print(f"전체 {len(x)}통 중 스팸 {int(y.sum())}통 = {y.mean() * 100:.1f}%")
    print(f"훈련 {len(x_train)}통, 테스트 {len(x_test)}통")
    print(f"테스트 정확도: {test_acc:.3f}")
    print()

    # 입력을 한 건씩 처리한다.
    items = []
    for title, links, caps, excl in MY_INPUTS:
        values = [float(links), float(caps), float(excl)]
        reason = check_input(*values)
        item = {
            "title": title,
            "values": values,
            "reason": reason,
            "pred": None,
            "prob": float("nan"),
            "push": np.zeros(3),
            "pred_name_ko": "",
            "pred_name_en": "",
            "top_reason_ko": "",
            "top_reason_en": "",
        }
        if reason is None:
            out = explain_one(model, values)
            item["prob"] = out["prob"]
            item["push"] = out["push"]
            item["pred"] = int(out["prob"] >= 0.5)
            item["pred_name_ko"] = "스팸" if item["pred"] == 1 else "정상"
            item["pred_name_en"] = "spam" if item["pred"] == 1 else "normal"
            top = int(np.argmax(np.abs(out["push"])))
            side_ko = "스팸 쪽" if out["push"][top] > 0 else "정상 쪽"
            side_en = "toward spam" if out["push"][top] > 0 else "toward normal"
            item["top_reason_ko"] = f"{FEATURE_NAMES[top]} ({side_ko})"
            item["top_reason_en"] = f"{FEATURE_NAMES_EN[top]} ({side_en})"
        items.append(item)

    # 콘솔 화면 - 이것이 앱의 출력이다.
    print("=== 입력 처리 결과 ===")
    rows = []
    for number, item in enumerate(items, start=1):
        print("-" * 58)
        print(f"입력 {number}  {item['title']}")
        for label_text, value, digits in zip(
            FEATURE_PADDED, item["values"], [0, 2, 0]
        ):
            print(f"  {label_text} : {value:.{digits}f}")

        if item["reason"] is not None:
            print("  판정         : 입력을 받지 않았다")
            print(f"  이유         : {item['reason']}")
            rows.append({
                "입력": item["title"],
                "링크 개수": item["values"][0],
                "대문자 비율": item["values"][1],
                "느낌표 개수": item["values"][2],
                "판정": "입력 거절",
                "스팸일 확률": "",
                "가장 크게 민 값": item["reason"],
            })
            continue

        print(f"  판정         : {item['pred_name_ko']}")
        print(f"  스팸일 확률  : {item['prob']:.3f}")
        print("  판단에 쓴 값이 민 크기 (양수는 스팸 쪽, 음수는 정상 쪽)")
        for name, push in zip(FEATURE_PADDED, item["push"]):
            print(f"    {name} : {push:+.2f}")
        rows.append({
            "입력": item["title"],
            "링크 개수": item["values"][0],
            "대문자 비율": item["values"][1],
            "느낌표 개수": item["values"][2],
            "판정": item["pred_name_ko"],
            "스팸일 확률": round(item["prob"], 3),
            "가장 크게 민 값": item["top_reason_ko"],
        })

    print("-" * 58)
    print()

    print("=== 이 모델이 틀릴 수 있는 경우 ===")
    for line in MODEL_LIMITS:
        print(f"  - {line}")
    print()

    # 카드 그림
    push_scale = max(
        float(np.max(np.abs(item["push"]))) for item in items
        if item["reason"] is None
    )
    push_scale = max(push_scale, 0.1)

    n_cards = len(items)
    n_cols = 3
    n_rows = (n_cards + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.0, 5.1 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, item in zip(axes, items):
        draw_card(ax, item, push_scale, ko)
    for ax in axes[n_cards:]:
        ax.axis("off")

    fig.suptitle(
        L(f"데모 앱 화면 — 입력 {n_cards}건의 결과 (합성 데이터, 테스트 정확도 {test_acc:.3f})",
          f"Demo app screen - {n_cards} inputs (synthetic data, test acc {test_acc:.3f})", ko),
        fontsize=15, y=0.985,
    )
    limits_text = L(
        "이 모델이 틀릴 수 있는 경우:  " + " / ".join(MODEL_LIMITS),
        "This model can be wrong: synthetic training data only; "
        "it reads only three numbers, not the mail text.",
        ko,
    )
    fig.text(0.5, 0.012, limits_text, ha="center", fontsize=10, color=GRAY)

    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    plot_path = RESULTS_DIR / "demo_app_cards.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    table = pd.DataFrame(rows)
    table_path = RESULTS_DIR / "demo_app_results.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    print("=== 저장된 결과 ===")
    print(f"카드 그림: results/{plot_path.name}")
    print(f"처리 표: results/{table_path.name}")
    print()
    print("예측 한 줄만 있으면 모델이고, 입력 검사와 확률과 근거와 한계가 붙어야 앱이다.")


if __name__ == "__main__":
    main()
