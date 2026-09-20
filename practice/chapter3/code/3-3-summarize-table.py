"""3장 7~9절 - 정제한 표를 요약하고, 글자 열을 숫자로 바꾸고, 공개 데이터셋을 열어 본다.

세 부분으로 되어 있다.
  1. 실습 3.2에서 정제한 표(week03-cleaned.csv)를 요약 통계와
     그림(히스토그램·산점도)으로 줄인다.
  2. 글자 분류 열을 원-핫 인코딩으로 0/1 열들로 바꾼다.
  3. scikit-learn 내장 붓꽃(iris) 데이터를 열어, 공개 데이터셋도
     같은 표 구조라는 것을 확인한다. 내려받기(네트워크)는 필요 없다.

데이터: 1은 컴퓨터로 만든 합성 데이터, 3은 공개 데이터셋(iris)이다.
        실명·학번·연락처가 들어갈 자리가 없다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import _krfont

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"


def draw_summary(ko: bool, df: pd.DataFrame, out_path: Path) -> int:
    """히스토그램(한 열의 분포)과 산점도(두 열의 관계)를 나란히 그린다."""
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    ax = axes[0]
    ax.hist(df["공부시간"], bins=12, color=BLUE, edgecolor="white")
    ax.set_xlabel(L("공부시간(시간)", "study hours", ko), fontsize=11)
    ax.set_ylabel(L("학생 수(명)", "students", ko), fontsize=11)
    ax.set_title(L("히스토그램 — 한 열의 분포", "histogram - one column", ko),
                 fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    has_label = df["시험점수"].notna()
    ax.scatter(df.loc[has_label, "공부시간"], df.loc[has_label, "시험점수"],
               s=46, color=BLUE, alpha=0.75, edgecolors="white", linewidths=0.6)
    ax.set_xlabel(L("공부시간(시간)", "study hours", ko), fontsize=11)
    ax.set_ylabel(L("시험점수(점)", "exam score", ko), fontsize=11)
    ax.set_title(L("산점도 — 두 열의 관계", "scatter - two columns", ko),
                 fontsize=13, pad=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return int(has_label.sum())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    csv_path = RESULTS_DIR / "week03-cleaned.csv"
    if not csv_path.exists():
        print("week03-cleaned.csv가 없다. 먼저 3-2-data-quality.py를 실행한다.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # 1. 요약 통계와 그림 ---------------------------------------------------
    print("=== 1. 정제한 표를 숫자로 요약한다 ===")
    _krfont.report(ko)
    print(f"읽은 표: results/{csv_path.name} ({df.shape[0]}행 x {df.shape[1]}열)")
    print()

    numeric_cols = ["공부시간", "수면시간", "시험점수"]
    summary = df[numeric_cols].describe().loc[
        ["count", "mean", "50%", "std", "min", "max"]
    ]
    summary.index = ["개수", "평균", "중앙값", "표준편차", "최솟값", "최댓값"]
    print(summary.round(2).to_string())
    print()

    corr = df["공부시간"].corr(df["시험점수"])
    print(f"공부시간과 시험점수의 상관계수: {corr:.2f}")
    print("(1에 가까우면 함께 커지는 관계, 0이면 관계 없음, -1에 가까우면 반대 방향)")
    print()

    plot_path = RESULTS_DIR / "table-summary.png"
    n_plotted = draw_summary(ko, df, plot_path)
    print(f"그림 저장: results/{plot_path.name}")
    print(f"산점도에 그린 학생: 라벨(시험점수)이 있는 {n_plotted}명")
    print()

    # 2. 원-핫 인코딩 -------------------------------------------------------
    print("=== 2. 글자 분류 열을 숫자로 바꾼다 (원-핫 인코딩) ===")
    mini = pd.DataFrame({
        "항목": ["떡볶이", "기생충", "농구"],
        "분류": ["음식", "영화", "운동"],
        "점수": [9, 8, 7],
    })
    print("바꾸기 전")
    print(mini.to_string(index=False))
    print()
    onehot = pd.get_dummies(mini, columns=["분류"], dtype=int)
    print("바꾼 후")
    print(onehot.to_string(index=False))
    print()
    print("분류 열 하나가 0과 1만 담는 세 열이 됐다. 행마다 1은 한 칸뿐이다.")
    print()

    # 3. 공개 데이터셋 ------------------------------------------------------
    print("=== 3. 공개 데이터셋도 같은 표다 ===")
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True).frame
    print(f"scikit-learn 내장 붓꽃(iris) 데이터: {iris.shape[0]}행 x {iris.shape[1]}열")
    print(f"열 이름: {list(iris.columns)}")
    print()
    print("표 앞부분 3행")
    print(iris.head(3).to_string(index=False))
    print()
    print("행 하나가 꽃 한 송이, 앞 네 열이 특성(꽃잎·꽃받침 치수), target 열이 품종 라벨이다.")
    print("내가 만든 관심사 표와 구조가 같다. 크기와 내용만 다르다.")


if __name__ == "__main__":
    main()
