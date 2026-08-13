"""3주차 실습: 학생 관심사 데이터 만들기, 시각화, 규칙 기반 추천.

합성 데이터로 CSV를 만들고, 막대그래프를 저장하고, 간단한 규칙 기반
추천을 실행한다. 모든 이름은 가명이며 개인정보를 포함하지 않는다.

산출물:
  - results/week03-my-data.csv
  - results/data_overview.png
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── 설정 ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

# ── 합성 데이터 생성 ─────────────────────────────────────────────

def make_synthetic_data(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """가명 학생 10명의 관심사 점수(1-5)를 합성한다.

    열 구성: name, movie, food, music, sports, game
    모든 이름은 가명이며 실제 인물과 관계가 없다.
    """
    rng = np.random.default_rng(seed)

    names = [
        "Student_A", "Student_B", "Student_C", "Student_D", "Student_E",
        "Student_F", "Student_G", "Student_H", "Student_I", "Student_J",
    ]
    categories = ["movie", "food", "music", "sports", "game"]

    rows: list[dict[str, object]] = []
    for name in names:
        scores = rng.integers(1, 6, size=len(categories))  # 1~5
        row: dict[str, object] = {"name": name}
        for cat, score in zip(categories, scores):
            row[cat] = int(score)
        rows.append(row)

    return pd.DataFrame(rows)


# ── 규칙 기반 추천 ───────────────────────────────────────────────

def rule_based_recommend(df: pd.DataFrame) -> pd.DataFrame:
    """간단한 if 규칙으로 추천 라벨을 붙인다.

    규칙:
      - movie >= 4 이고 game >= 4  → 'SF fan'
      - sports >= 4 이고 food >= 4 → 'Active foodie'
      - music >= 4                 → 'Music lover'
      - 그 외                      → 'Explorer'
    """
    labels: list[str] = []
    for _, row in df.iterrows():
        if row["movie"] >= 4 and row["game"] >= 4:
            labels.append("SF fan")
        elif row["sports"] >= 4 and row["food"] >= 4:
            labels.append("Active foodie")
        elif row["music"] >= 4:
            labels.append("Music lover")
        else:
            labels.append("Explorer")
    df = df.copy()
    df["recommendation"] = labels
    return df


# ── 시각화 ────────────────────────────────────────────────────────

def plot_overview(df: pd.DataFrame, output_path: Path) -> None:
    """학생별 관심사 점수를 가로 막대그래프로 그린다."""
    categories = ["movie", "food", "music", "sports", "game"]
    n_students = len(df)
    n_cats = len(categories)

    fig, ax = plt.subplots(figsize=(9, 5))

    bar_height = 0.15
    y_positions = np.arange(n_students)

    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]

    for i, cat in enumerate(categories):
        offsets = y_positions + i * bar_height
        ax.barh(offsets, df[cat], height=bar_height, label=cat, color=colors[i])

    ax.set_yticks(y_positions + bar_height * (n_cats - 1) / 2)
    ax.set_yticklabels(df["name"])
    ax.set_xlabel("Score (1-5)")
    ax.set_title("Student Interest Scores (synthetic data)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ── 데이터 품질 체크 ─────────────────────────────────────────────

def quality_check(df: pd.DataFrame) -> list[str]:
    """데이터 품질 체크리스트를 반환한다."""
    score_cols = ["movie", "food", "music", "sports", "game"]
    checks: list[str] = []

    # 1. 결측값
    missing = df[score_cols].isnull().sum().sum()
    checks.append(f"결측값 수: {missing}")

    # 2. 범위 확인 (1-5 범위)
    out_of_range = 0
    for col in score_cols:
        out_of_range += ((df[col] < 1) | (df[col] > 5)).sum()
    checks.append(f"범위 밖(1-5) 값 수: {out_of_range}")

    # 3. 중복 행
    duplicates = df.duplicated(subset=score_cols).sum()
    checks.append(f"중복 행 수: {duplicates}")

    # 4. 각 카테고리 평균
    for col in score_cols:
        checks.append(f"{col} 평균: {df[col].mean():.1f}")

    # 5. 행 수
    checks.append(f"전체 데이터 수: {len(df)}행 x {len(df.columns)}열")

    return checks


# ── 특성과 라벨 설명 ─────────────────────────────────────────────

def explain_features_and_labels(df: pd.DataFrame) -> None:
    """특성과 라벨을 구분해 출력한다."""
    feature_cols = ["movie", "food", "music", "sports", "game"]
    label_col = "recommendation"

    print("=== 특성(feature)과 라벨(label) ===")
    print(f"특성 열(입력): {feature_cols}")
    print(f"  -> 모델이 판단 재료로 사용하는 열")
    print(f"라벨 열(정답): {label_col}")
    print(f"  -> 모델이 맞혀야 하는 정답 열")
    print()

    print("첫 3행 예시:")
    print(df[feature_cols + [label_col]].head(3).to_string(index=False))


# ── 메인 ─────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 환경 정보
    print("=== 3주차 Python과 데이터 ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"matplotlib: {plt.matplotlib.__version__}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    # 1. 합성 데이터 생성
    df = make_synthetic_data()
    print("=== 합성 데이터 (synthetic data) ===")
    print("주의: 이 데이터는 합성 데이터이며, 실제 학생 정보가 아닙니다.")
    print()
    print(df.to_string(index=False))
    print()

    # 2. 규칙 기반 추천
    df_with_label = rule_based_recommend(df)
    print("=== 규칙 기반 추천 결과 ===")
    print("규칙:")
    print("  - movie >= 4 AND game >= 4  -> SF fan")
    print("  - sports >= 4 AND food >= 4 -> Active foodie")
    print("  - music >= 4               -> Music lover")
    print("  - otherwise                -> Explorer")
    print()
    print(df_with_label[["name", "recommendation"]].to_string(index=False))
    print()

    # 3. 추천 분포
    rec_counts = df_with_label["recommendation"].value_counts()
    print("=== 추천 분포 ===")
    for label, count in rec_counts.items():
        print(f"  {label}: {count}명")
    print()

    # 4. 특성과 라벨 설명
    explain_features_and_labels(df_with_label)
    print()

    # 5. 데이터 품질 체크
    checks = quality_check(df)
    print("=== 데이터 품질 체크리스트 ===")
    for check in checks:
        print(f"  {check}")
    print()

    # 6. CSV 저장
    csv_path = RESULTS_DIR / "week03-my-data.csv"
    df_with_label.to_csv(csv_path, index=False)
    print(f"CSV 저장: {csv_path.relative_to(CHAPTER_DIR)}")

    # 7. 시각화 저장
    plot_path = RESULTS_DIR / "data_overview.png"
    plot_overview(df, plot_path)
    print(f"그래프 저장: {plot_path.relative_to(CHAPTER_DIR)}")
    print()

    # 8. 안전 점검
    print("=== 안전 점검 ===")
    print("- 데이터에 실제 개인정보(이름, 학번, 연락처)가 포함되지 않았다.")
    print("- 합성 데이터임을 출력에 표시했다.")
    print(f"- 이 코드가 읽는 파일: 없음 (자체 생성)")
    print(f"- 이 코드가 쓰는 파일: {csv_path.name}, {plot_path.name}")
    print("- 외부 네트워크 전송: 없음")


if __name__ == "__main__":
    main()
