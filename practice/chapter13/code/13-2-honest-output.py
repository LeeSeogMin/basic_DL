"""실습 13.2 - 모델이 자신 없을 때 답을 미루게 만든다.

4장에서 "정확도만 보면 속는다"를 확인했다. 그때는 무엇을 놓쳤는지 세는 것으로 끝냈다.
이번에는 한 걸음 더 간다. 확률이 낮은 예측을 앱이 아예 단정하지 않게 만든다.

  확신 = max(스팸일 확률, 정상일 확률)
  확신이 CONFIDENCE_THRESHOLD 보다 낮으면 "판단 보류"라고 쓴다.

기준선을 여러 값으로 바꿔가며 세 숫자가 어떻게 맞바뀌는지 표와 그림으로 남긴다.
  - 단정한 답의 수
  - 단정했다가 틀린 답의 수
  - 판단 보류한 건수

기준선을 올리면 틀린 답이 줄지만 답하지 못하는 건수가 는다. 이것이 이 실습의 핵심이다.

주의: 이 데이터는 실제 메일이 아니라 컴퓨터로 만든 합성 데이터다.
실습 13.1과 같은 합성 데이터, 같은 모델을 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.
# 확신이 이 값보다 낮으면 앱이 답을 단정하지 않고 "판단 보류"로 쓴다.
# 0.50 이면 모든 건에 답한다. 1.00 에 가까울수록 답하는 건수가 줄어든다.
CONFIDENCE_THRESHOLD = 0.80

# 표와 그림에 함께 넣을 비교용 기준선들
THRESHOLD_SWEEP = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_MAILS = 1200
SPAM_RATIO = 0.4

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def make_synthetic_mails() -> tuple[np.ndarray, np.ndarray]:
    """합성 메일 데이터를 만든다. 실습 13.1과 같은 방법이다."""
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


def count_at(threshold: float, confidence: np.ndarray,
             pred: np.ndarray, truth: np.ndarray) -> dict:
    """기준선 하나에서 단정한 답, 틀린 답, 보류 건수를 센다."""
    answered = confidence >= threshold
    n_answered = int(answered.sum())
    n_wrong = int((answered & (pred != truth)).sum())
    n_right = n_answered - n_wrong
    n_hold = int(len(truth) - n_answered)

    return {
        "기준선": threshold,
        "단정한 답": n_answered,
        "그중 틀린 답": n_wrong,
        "판단 보류": n_hold,
        "단정한 답의 정확도": round(n_right / n_answered, 3) if n_answered else float("nan"),
    }


def screen_text(prob: float, threshold: float) -> list[str]:
    """앱이 화면에 쓸 문구를 만든다. 자신 없으면 답을 단정하지 않는다."""
    confidence = max(prob, 1.0 - prob)
    if confidence < threshold:
        return [
            "판단 보류",
            f"  스팸일 확률 {prob:.2f} — 확신 {confidence:.2f}이 기준선 {threshold:.2f}보다 낮다",
            "  이 메일은 자신 있게 답하지 못한다. 사람이 직접 확인하기 바란다.",
        ]
    name = "스팸" if prob >= 0.5 else "정상"
    return [
        f"{name}",
        f"  스팸일 확률 {prob:.2f} — 확신 {confidence:.2f}이 기준선 {threshold:.2f} 이상이다",
        "  그래도 틀릴 수 있다. 중요한 메일이면 직접 확인하기 바란다.",
    ]


def draw_tradeoff(table: pd.DataFrame, n_test: int, ko: bool, out_path: Path) -> None:
    """왼쪽에 맞바꿈 곡선, 오른쪽에 기준선별 건수 막대를 그린다."""
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))

    # 왼쪽 - 기준선을 올릴 때 두 숫자가 어떻게 움직이는가
    ax = axes[0]
    thresholds = table["기준선"].to_numpy()
    ax.plot(thresholds, table["그중 틀린 답"], color=RED, marker="o", linewidth=2.4,
            label=L("단정했다가 틀린 답", "wrong confident answers", ko))
    ax.plot(thresholds, table["판단 보류"], color=GRAY, marker="s", linestyle="--",
            linewidth=2.4, label=L("판단 보류한 건수", "answers withheld", ko))
    ax.axvline(CONFIDENCE_THRESHOLD, color=ORANGE, linewidth=1.6, linestyle=":")
    ax.text(CONFIDENCE_THRESHOLD, ax.get_ylim()[1] * 0.96,
            L(f" 지금 기준선 {CONFIDENCE_THRESHOLD:.2f}",
              f" current {CONFIDENCE_THRESHOLD:.2f}", ko),
            color=ORANGE, fontsize=10, ha="left", va="top")

    ax.set_xlabel(L("확신 기준선", "confidence threshold", ko), fontsize=11)
    ax.set_ylabel(L(f"테스트 {n_test}건 중 건수", f"count out of {n_test} test mails", ko),
                  fontsize=11)
    ax.set_title(L("기준선을 올리면 틀린 답이 줄고 보류가 는다",
                   "Raising the threshold trims errors and adds withheld cases", ko),
                 fontsize=12.5, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # 오른쪽 - 기준선별로 360건이 어떻게 갈리는가
    ax = axes[1]
    y_pos = np.arange(len(table))
    right = table["단정한 답"] - table["그중 틀린 답"]
    wrong = table["그중 틀린 답"]
    hold = table["판단 보류"]

    ax.barh(y_pos, right, color=GREEN, height=0.62,
            label=L("단정해서 맞힌 답", "correct confident answers", ko))
    ax.barh(y_pos, wrong, left=right, color=RED, height=0.62,
            label=L("단정했다가 틀린 답", "wrong confident answers", ko))
    ax.barh(y_pos, hold, left=right + wrong, color="#C8C6C4", height=0.62,
            label=L("판단 보류", "withheld", ko))

    for i in range(len(table)):
        # 맞힌 답은 초록 칸 안에, 틀린 답은 막대 오른쪽 바깥에 쓴다.
        # 보류는 회색 칸이 넉넉할 때만 칸 안에 쓴다. 그래야 숫자가 겹치지 않는다.
        ax.text(right.iloc[i] / 2, y_pos[i], f"{right.iloc[i]}",
                ha="center", va="center", fontsize=9.5, color="white")
        ax.text(n_test * 1.02, y_pos[i], f"{wrong.iloc[i]}",
                ha="left", va="center", fontsize=9.5, color=RED)
        if hold.iloc[i] >= 30:
            ax.text(n_test - hold.iloc[i] / 2, y_pos[i], f"{hold.iloc[i]}",
                    ha="center", va="center", fontsize=9.5, color="#333333")

    ax.text(n_test * 1.02, -0.72, L("틀린 답", "wrong", ko),
            ha="left", va="center", fontsize=9.5, color=RED)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{t:.2f}" for t in thresholds], fontsize=10)
    ax.set_ylim(len(table) - 0.45, -1.05)
    ax.set_xlim(0, n_test * 1.14)
    ax.set_xticks(list(range(0, n_test + 1, 60)))
    ax.set_xlabel(L(f"테스트 {n_test}건", f"{n_test} test mails", ko), fontsize=11)
    ax.set_ylabel(L("확신 기준선", "confidence threshold", ko), fontsize=11)
    ax.set_title(L("기준선별로 360건이 어떻게 갈리는가",
                   "How the test mails split at each threshold", ko).replace(
                       "360", str(n_test)),
                 fontsize=12.5, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        L("자신 없을 때 답을 미루면 무엇이 좋아지고 무엇이 나빠지는가 (합성 데이터)",
          "What improves and what worsens when the app withholds answers (synthetic data)",
          ko),
        fontsize=14.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    x, y = make_synthetic_mails()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=RANDOM_SEED),
    )
    model.fit(x_train, y_train)

    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    confidence = np.maximum(prob, 1.0 - prob)
    n_test = len(y_test)

    print("=== 실습 13.2 자신 없을 때 답을 미루는 앱 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 메일 데이터(실제 메일 아님)")
    print(f"전체 {len(x)}통 중 스팸 {int(y.sum())}통 = {y.mean() * 100:.1f}%")
    print(f"훈련 {len(x_train)}통, 테스트 {n_test}통")
    print(f"모든 건에 답했을 때의 정확도: {accuracy_score(y_test, pred):.3f}")
    print()

    # 지금 기준선에서의 결과
    now = count_at(CONFIDENCE_THRESHOLD, confidence, pred, y_test)
    print(f"=== 지금 기준선 CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD:.2f} ===")
    print(f"단정한 답: {now['단정한 답']}건")
    print(f"  그중 틀린 답: {now['그중 틀린 답']}건")
    print(f"  단정한 답의 정확도: {now['단정한 답의 정확도']:.3f}")
    print(f"판단 보류: {now['판단 보류']}건 "
          f"({now['판단 보류'] / n_test * 100:.1f}%)")
    print()

    # 화면 문구 예시 세 건
    print("=== 화면에 나가는 문구 예시 ===")
    picks = [
        ("가장 자신 있는 스팸", int(np.argmax(prob))),
        ("가장 자신 있는 정상", int(np.argmin(prob))),
        ("가장 애매한 메일", int(np.argmin(np.abs(prob - 0.5)))),
    ]
    for title, idx in picks:
        links, caps, excl = x_test[idx]
        truth = "스팸" if y_test[idx] == 1 else "정상"
        print(f"[{title}]  링크 {links:.0f}개, 대문자 {caps:.2f}, 느낌표 {excl:.0f}개"
              f"  (실제 정답: {truth})")
        for line in screen_text(float(prob[idx]), CONFIDENCE_THRESHOLD):
            print(f"  {line}")
        print()

    # 기준선을 바꿔가며 센다
    table = pd.DataFrame(
        [count_at(t, confidence, pred, y_test) for t in THRESHOLD_SWEEP]
    )
    print("=== 기준선을 바꾸면 무엇이 맞바뀌는가 ===")
    print(table.to_string(index=False))
    print()

    lowest, highest = table.iloc[0], table.iloc[-1]
    print(f"기준선 {lowest['기준선']:.2f} → {highest['기준선']:.2f} 로 올리면")
    print(f"  틀린 답: {int(lowest['그중 틀린 답'])}건 → {int(highest['그중 틀린 답'])}건")
    print(f"  판단 보류: {int(lowest['판단 보류'])}건 → {int(highest['판단 보류'])}건")
    print("  틀린 답이 준 만큼 답하지 못한 건수가 늘었다. 공짜로 좋아지지 않는다.")
    print()

    plot_path = RESULTS_DIR / "confidence_tradeoff.png"
    draw_tradeoff(table, n_test, ko, plot_path)

    table_path = RESULTS_DIR / "threshold_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    print("=== 저장된 결과 ===")
    print(f"맞바꿈 그림: results/{plot_path.name}")
    print(f"기준선 표: results/{table_path.name}")
    print()
    print("이 모델이 틀릴 수 있는 경우:")
    print("  - 합성 데이터로만 배웠다. 실제 받은 편지함에서 시험한 적이 없다.")
    print("  - 확신이 높다고 맞는 것은 아니다. 기준선 위에서도 틀린 답이 남는다.")
    print("  - 보류한 건수는 사람이 처리해야 한다. 사람이 감당할 양인지 먼저 확인한다.")


if __name__ == "__main__":
    main()
