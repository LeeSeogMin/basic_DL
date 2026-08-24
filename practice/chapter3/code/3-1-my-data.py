"""실습 3.1 - 내 관심사를 표로 만들고 CSV로 저장한 뒤 그림으로 그린다.

리스트와 딕셔너리로 적은 내용을 pandas 표로 옮기고, CSV 파일로 저장하고,
저장한 파일을 다시 읽어 원본과 같은지 확인한 다음, 막대그래프로 그린다.

아래 세 개의 리스트를 자기 취향으로 바꾸면 표와 그림이 함께 바뀐다.

주의: 이 표에 실명, 학번, 전화번호, 이메일, SNS 계정을 쓰지 않는다.
      제출한 파일은 다른 사람이 열어 볼 수 있다. 별명만 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import _krfont

# ---------------------------------------------------------------------------
# 여기를 자기 취향으로 바꾼다. ("이름", 점수) 형태이고 점수는 0~10이다.
# 항목 수는 몇 개를 넣어도 되지만, 한 분류에 3~5개가 보기 좋다.
NICKNAME = "student01"          # 실명 대신 쓰는 별명. 학번이나 이메일을 쓰지 않는다.

FAVORITE_FOODS = [
    ("떡볶이", 9),
    ("김치찌개", 7),
    ("김밥", 6),
    ("샐러드", 4),
]
FAVORITE_MOVIES = [
    ("인터스텔라", 10),
    ("기생충", 8),
    ("어벤져스", 6),
]
FAVORITE_SPORTS = [
    ("배드민턴", 7),
    ("농구", 5),
    ("등산", 3),
]

# 점수가 이 값 이상이면 "좋아함"으로 본다. 이 값이 라벨을 만드는 기준이다.
LIKE_THRESHOLD = 7

# 표를 훈련용과 테스트용으로 나눌 때 테스트 쪽 비율
TEST_RATIO = 0.3
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
GRAY = "#79706E"

CATEGORY_COLORS = {"음식": BLUE, "영화": ORANGE, "운동": GREEN}


def build_table() -> pd.DataFrame:
    """세 개의 리스트를 하나의 표로 합친다.

    리스트 -> 딕셔너리 -> 표 순서로 옮긴다. 세 그릇이 같은 내용을 담는다.
    """
    groups = {
        "음식": FAVORITE_FOODS,
        "영화": FAVORITE_MOVIES,
        "운동": FAVORITE_SPORTS,
    }

    records = []
    for category, pairs in groups.items():
        for name, score in pairs:
            # 행 하나를 딕셔너리로 만든다. 키가 열 이름이 된다.
            records.append({
                "별명": NICKNAME,
                "분류": category,
                "항목": name,
                "점수": int(score),
                "이름글자수": len(name),
            })

    df = pd.DataFrame(records)
    # 라벨 열을 만든다. 점수라는 숫자를 "좋아함/보통"이라는 이름으로 바꾼다.
    df["좋아함"] = (df["점수"] >= LIKE_THRESHOLD).map({True: "좋아함", False: "보통"})
    return df


def save_plot(ko: bool, df: pd.DataFrame, out_path: Path) -> None:
    """왼쪽에는 항목별 점수 막대, 오른쪽에는 분류별 평균 점수 막대를 그린다."""
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})

    # --- 왼쪽: 항목별 점수 ---
    ax = axes[0]
    ordered = df.sort_values(["분류", "점수"], ascending=[True, True])
    positions = range(len(ordered))
    colors = [CATEGORY_COLORS[c] for c in ordered["분류"]]
    ax.barh(list(positions), ordered["점수"], color=colors, height=0.68)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(ordered["항목"], fontsize=11)
    ax.set_xlim(0, 10.8)
    ax.set_xlabel(L("점수 (0~10)", "score (0-10)", ko), fontsize=12)
    ax.set_title(L("항목별 점수", "score by item", ko), fontsize=13, pad=10)
    ax.axvline(LIKE_THRESHOLD, color=GRAY, linestyle="--", linewidth=1.6)
    ax.text(LIKE_THRESHOLD + 0.15, len(ordered) - 0.45,
            L(f"{LIKE_THRESHOLD}점 이상이면 '좋아함'",
              f"score >= {LIKE_THRESHOLD} -> 'like'", ko),
            fontsize=10, color=GRAY, va="center")
    for y, score in zip(positions, ordered["점수"]):
        ax.text(score + 0.2, y, str(score), va="center", fontsize=10, color="#333333")
    ax.grid(axis="x", alpha=0.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for color in CATEGORY_COLORS.values()]
    names = [L(k, {"음식": "food", "영화": "movie", "운동": "sport"}[k], ko)
             for k in CATEGORY_COLORS]
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, fontsize=10, frameon=False)

    # --- 오른쪽: 분류별 평균 ---
    ax = axes[1]
    means = df.groupby("분류")["점수"].mean().reindex(list(CATEGORY_COLORS))
    counts = df.groupby("분류")["점수"].size().reindex(list(CATEGORY_COLORS))
    labels = [L(k, {"음식": "food", "영화": "movie", "운동": "sport"}[k], ko)
              for k in means.index]
    ax.bar(labels, means.values,
           color=[CATEGORY_COLORS[k] for k in means.index], width=0.6)
    ax.set_ylim(0, 10.8)
    ax.set_ylabel(L("평균 점수", "mean score", ko), fontsize=12, labelpad=10)
    ax.set_title(L("분류별 평균 점수", "mean score by category", ko),
                 fontsize=13, pad=10)
    for i, (mean, count) in enumerate(zip(means.values, counts.values)):
        ax.text(i, mean + 0.25, f"{mean:.1f}", ha="center",
                fontsize=11, color="#333333")
        ax.text(i, 0.35, L(f"{count}개", f"n={count}", ko), ha="center",
                fontsize=10, color="white")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        L(f"{NICKNAME}의 관심사 표 (학생이 직접 채운 값)",
          f"Interest table of {NICKNAME} (filled in by the student)", ko),
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    print("=== 실습 3.1 내 관심사를 표로 만든다 ===")
    _krfont.report(ko)
    print("이 표에는 실명·학번·연락처를 넣지 않는다. 별명만 쓴다.")
    print(f"별명: {NICKNAME}")
    print()

    # 1) 리스트에 적은 내용을 확인한다
    print("=== 1단계. 리스트에 적은 내용 ===")
    print(f"좋아하는 음식 {len(FAVORITE_FOODS)}개: {[name for name, _ in FAVORITE_FOODS]}")
    print(f"좋아하는 영화 {len(FAVORITE_MOVIES)}개: {[name for name, _ in FAVORITE_MOVIES]}")
    print(f"좋아하는 운동 {len(FAVORITE_SPORTS)}개: {[name for name, _ in FAVORITE_SPORTS]}")
    print()

    # 2) 딕셔너리로 옮긴 뒤 표로 만든다
    df = build_table()
    print("=== 2단계. 표로 만든 결과 ===")
    print(f"표의 크기: {df.shape[0]}행 x {df.shape[1]}열")
    print(f"열 이름: {list(df.columns)}")
    print()
    print(df.to_string(index=False))
    print()

    print("=== 열마다 어떤 자료형인가 ===")
    print(df.dtypes.to_string())
    print()

    # 3) CSV로 저장하고 다시 읽어 확인한다
    csv_path = RESULTS_DIR / "week03-my-data.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    reloaded = pd.read_csv(csv_path)

    print("=== 3단계. CSV로 저장하고 다시 읽기 ===")
    print(f"저장한 파일: results/{csv_path.name}")
    print(f"파일 크기: {csv_path.stat().st_size}바이트")
    print(f"다시 읽은 표의 크기: {reloaded.shape[0]}행 x {reloaded.shape[1]}열")
    print(f"저장 전과 후의 내용이 같은가: {df.equals(reloaded)}")
    print()

    # 4) 특성과 라벨을 정한다
    feature_columns = ["분류", "점수", "이름글자수"]
    label_column = "좋아함"
    print("=== 4단계. 특성과 라벨 정하기 ===")
    print(f'질문: "이 항목을 좋아할까?"')
    print(f"특성(feature) 열: {feature_columns}")
    print(f"라벨(label) 열: {label_column}")
    print(df[label_column].value_counts().to_string())
    print()

    # 5) 훈련 데이터와 테스트 데이터로 나눈다
    train_df, test_df = train_test_split(
        df, test_size=TEST_RATIO, random_state=RANDOM_SEED,
    )
    print("=== 5단계. 훈련 데이터와 테스트 데이터로 나누기 ===")
    print(f"전체 {len(df)}행 = 훈련 {len(train_df)}행 + 테스트 {len(test_df)}행")
    print(f"테스트 쪽으로 간 항목: {list(test_df['항목'])}")
    print("테스트 쪽 행은 모델을 만들 때 보지 않는다. 4장에서 이 규칙을 다시 쓴다.")
    print()

    # 6) 그림으로 그린다
    plot_path = RESULTS_DIR / "my-interests.png"
    save_plot(ko, df, plot_path)

    print("=== 저장된 결과 ===")
    print(f"표 파일: results/{csv_path.name}")
    print(f"그림 파일: results/{plot_path.name}")
    print()
    print("파일 맨 위의 리스트를 바꿔 다시 실행하면 표와 그림이 함께 바뀐다.")


if __name__ == "__main__":
    main()
