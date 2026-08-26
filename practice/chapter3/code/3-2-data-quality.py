"""실습 3.2 - 일부러 흠집 낸 표를 하나씩 고친다.

컴퓨터로 만든 합성 설문 표에 네 가지 흠집을 일부러 낸다.
  ① 중복 행          같은 응답이 두 번 들어왔다
  ② 문자열로 들어온 숫자  "82점" 처럼 단위가 붙어 숫자로 읽히지 않는다
  ③ 범위를 벗어난 이상치  하루에 99시간을 잤다고 적혀 있다
  ④ 결측값            칸이 비어 있다

그다음 흠집을 하나씩 찾아 세고, 고치고, 남은 것을 센다.
찾은 건수 = 고친 건수 + 남은 건수 가 맞아떨어지는지 코드가 직접 확인한다.

결측을 0으로 채우면 평균이 어떻게 어긋나는지도 숫자로 보여준다.
흠집을 내기 전 원본 값을 우리가 알고 있으므로 참값과 대조할 수 있다.

데이터: 컴퓨터로 만든 합성 데이터다. 실제 설문 응답이 아니고,
        실명·학번·연락처가 들어갈 자리가 없다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _krfont

# ---------------------------------------------------------------------------
# 여기 숫자를 바꾸면 흠집의 개수가 바뀌고, 아래 총계와 그림이 함께 바뀐다.
N_STUDENTS = 60          # 흠집 내기 전 원본 응답 수

N_DUPLICATE_ROWS = 4     # ① 통째로 두 번 들어간 행의 수
N_TEXT_SCORE = 6         # ② "82점" 처럼 단위가 붙은 시험점수 칸의 수
N_OUTLIER_SLEEP = 3      # ③ 범위를 벗어난 수면시간 칸의 수
N_MISSING_STUDY = 10     # ④ 비어 있는 공부시간 칸의 수 (특성 열)
N_MISSING_SCORE = 3      # ④ 비어 있는 시험점수 칸의 수 (라벨 열)

# 결측을 무엇으로 채울지 정한다. "median" 또는 "zero".
# "zero" 로 바꿔 다시 실행하면 정제 후 그림에서 점들이 왼쪽 벽에 달라붙는다.
FILL_STRATEGY = "median"
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# 있을 수 있는 값의 범위. 이 밖의 값을 이상치로 본다.
VALID_RANGE = {
    "공부시간": (0.0, 16.0),
    "수면시간": (3.0, 14.0),
    "시험점수": (0.0, 100.0),
}

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def make_clean_table(rng: np.random.Generator) -> pd.DataFrame:
    """흠집 내기 전의 원본 표를 만든다. 이 표가 참값 기준이 된다."""
    study = np.clip(rng.normal(4.0, 1.5, N_STUDENTS), 0.5, 9.0).round(1)
    sleep = np.clip(rng.normal(7.0, 1.2, N_STUDENTS), 4.5, 9.5).round(1)
    score = np.clip(
        38 + 6.0 * study + 2.0 * sleep + rng.normal(0, 6.0, N_STUDENTS), 0, 100
    ).round(0).astype(int)

    return pd.DataFrame({
        "응답번호": np.arange(1, N_STUDENTS + 1),
        "별명": [f"student{i:02d}" for i in range(1, N_STUDENTS + 1)],
        "공부시간": study,
        "수면시간": sleep,
        "시험점수": score,
    })


def make_dirty_table(clean: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """원본 표에 네 가지 흠집을 낸다.

    흠집이 서로 겹치지 않도록, 섞은 행 번호를 잘라 나눠 쓴다.
    같은 칸에 두 가지 흠집을 내면 나중에 건수를 두 번 세게 된다.
    """
    dirty = clean.copy()
    dirty["시험점수"] = dirty["시험점수"].astype(object)

    pool = rng.permutation(N_STUDENTS)
    take = 0

    def slice_off(n: int) -> np.ndarray:
        nonlocal take
        chosen = pool[take:take + n]
        take += n
        return chosen

    idx_missing_study = slice_off(N_MISSING_STUDY)
    idx_outlier_sleep = slice_off(N_OUTLIER_SLEEP)
    idx_missing_score = slice_off(N_MISSING_SCORE)
    idx_text_score = slice_off(N_TEXT_SCORE)
    idx_duplicate = slice_off(N_DUPLICATE_ROWS)

    # ④ 결측 - 공부시간 칸을 비운다
    dirty.loc[idx_missing_study, "공부시간"] = np.nan
    # ③ 이상치 - 있을 수 없는 수면시간을 넣는다
    absurd = [99.0, 99.0, -2.0]
    for k, row in enumerate(idx_outlier_sleep):
        dirty.loc[row, "수면시간"] = absurd[k % len(absurd)]
    # ④ 결측 - 시험점수(라벨) 칸을 비운다
    dirty.loc[idx_missing_score, "시험점수"] = np.nan
    # ② 문자열 - 점수 뒤에 단위를 붙인다
    for row in idx_text_score:
        dirty.loc[row, "시험점수"] = f"{int(dirty.loc[row, '시험점수'])}점"
    # ① 중복 - 흠집이 없는 행을 골라 통째로 한 번 더 붙인다
    duplicated_rows = dirty.loc[idx_duplicate].copy()

    return pd.concat([dirty, duplicated_rows], ignore_index=True)


def fix_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """① 완전히 같은 행을 찾아 하나만 남긴다."""
    found = int(df.duplicated().sum())
    fixed_df = df.drop_duplicates().reset_index(drop=True)
    return fixed_df, found, found


def fix_text_numbers(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int, int]:
    """② 숫자로 읽히지 않는 칸을 찾아 숫자로 바꾼다."""
    raw = df[column]
    blank_before = raw.isna()
    # 비어 있지도 않은데 숫자로 못 바꾸는 칸이 문자열 흠집이다
    found = int(((~blank_before) & pd.to_numeric(raw, errors="coerce").isna()).sum())

    stripped = raw.astype("string").str.replace("점", "", regex=False).str.strip()
    converted = pd.to_numeric(stripped, errors="coerce")
    still_broken = int((converted.isna() & ~blank_before).sum())

    out = df.copy()
    out[column] = converted
    return out, found, found - still_broken


def fix_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, dict]:
    """③ 정해 둔 범위를 벗어난 값을 찾아 그 열의 중앙값으로 바꾼다."""
    out = df.copy()
    found = 0
    fixed = 0
    detail = {}

    for column, (low, high) in VALID_RANGE.items():
        values = out[column]
        bad = values.notna() & ((values < low) | (values > high))
        n_bad = int(bad.sum())
        if n_bad == 0:
            detail[column] = 0
            continue

        replacement = round(float(values[~bad].median()), 1)
        out.loc[bad, column] = replacement
        found += n_bad
        fixed += n_bad
        detail[column] = n_bad

    return out, found, fixed, detail


def fix_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """④ 빈 칸을 센다. 특성 열은 채우고, 라벨 열은 채우지 않는다.

    라벨은 맞혀야 하는 정답이다. 정답을 지어내면 그 행은 정답이 아니라
    내가 만든 값을 정답이라고 우기는 행이 된다. 그래서 채우지 않고 남긴다.
    """
    out = df.copy()
    report = {}

    n_missing_feature = int(out["공부시간"].isna().sum())
    if FILL_STRATEGY == "zero":
        fill_value = 0.0
    else:
        fill_value = round(float(out["공부시간"].median()), 1)
    out["공부시간채움"] = out["공부시간"].isna()
    out["공부시간"] = out["공부시간"].fillna(fill_value)

    n_missing_label = int(out["시험점수"].isna().sum())

    report["특성결측_찾음"] = n_missing_feature
    report["특성결측_고침"] = n_missing_feature
    report["채운값"] = fill_value
    report["라벨결측_찾음"] = n_missing_label
    report["라벨결측_고침"] = 0
    return out, report


def mean_comparison(truth: pd.DataFrame, before_fill: pd.DataFrame) -> pd.DataFrame:
    """공부시간 평균을 네 가지 방법으로 구해 나란히 놓는다."""
    column = before_fill["공부시간"]
    median_value = round(float(column.median()), 1)

    rows = [
        ("흠집 내기 전 원본", float(truth["공부시간"].mean())),
        ("결측을 0으로 채움", float(column.fillna(0.0).mean())),
        ("결측인 행을 뺌", float(column.dropna().mean())),
        (f"결측을 중앙값({median_value})으로 채움", float(column.fillna(median_value).mean())),
    ]
    table = pd.DataFrame(rows, columns=["방법", "평균 공부시간"])
    truth_mean = table.loc[0, "평균 공부시간"]
    table["참값과의 차이"] = table["평균 공부시간"] - truth_mean
    table["어긋난 정도(%)"] = table["참값과의 차이"] / truth_mean * 100
    return table


def draw_before_after(ko: bool, dirty: pd.DataFrame, cleaned: pd.DataFrame,
                      counts: pd.DataFrame, means: pd.DataFrame,
                      out_path: Path) -> None:
    """정제 전과 정제 후를 위아래 네 칸으로 나란히 그린다."""
    L = _krfont.label
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4))

    # --- (1) 정제 전 산점도 ---
    ax = axes[0, 0]
    sleep = pd.to_numeric(dirty["수면시간"], errors="coerce")
    study = pd.to_numeric(dirty["공부시간"], errors="coerce")
    low, high = VALID_RANGE["수면시간"]
    bad = (sleep < low) | (sleep > high)
    ax.scatter(study[~bad], sleep[~bad], s=46, color=BLUE, alpha=0.75,
               edgecolors="white", linewidths=0.6,
               label=L("범위 안의 값", "in range", ko))
    ax.scatter(study[bad], sleep[bad], s=150, color=RED, marker="X",
               label=L("범위를 벗어난 값", "out of range", ko))
    ax.set_xlabel(L("공부시간(시간)", "study hours", ko), fontsize=11)
    ax.set_ylabel(L("수면시간(시간)", "sleep hours", ko), fontsize=11, labelpad=8)
    ax.set_title(
        L(f"정제 전 — {len(dirty)}행", f"before cleaning - {len(dirty)} rows", ko),
        fontsize=13, pad=10, color=RED,
    )
    ax.text(0.03, 0.55,
            L(f"수면시간 이상치 때문에\n세로축이 99까지 늘어난다.\n"
              f"나머지 점이 아래에 뭉친다.\n\n"
              f"공부시간이 빈 {int(study.isna().sum())}건은\n점으로 그릴 수도 없다.",
              f"Outliers stretch the y axis to 99.\n"
              f"{int(study.isna().sum())} rows cannot be plotted at all.", ko),
            transform=ax.transAxes, fontsize=10, color="#333333",
            va="center", linespacing=1.7)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              fontsize=10, frameon=False)

    # --- (2) 정제 후 산점도 ---
    ax = axes[0, 1]
    filled = cleaned["공부시간채움"]
    ax.scatter(cleaned.loc[~filled, "공부시간"], cleaned.loc[~filled, "수면시간"],
               s=46, color=BLUE, alpha=0.75, edgecolors="white", linewidths=0.6,
               label=L("원래 값", "original value", ko))
    ax.scatter(cleaned.loc[filled, "공부시간"], cleaned.loc[filled, "수면시간"],
               s=90, color=ORANGE, marker="s", edgecolors="white", linewidths=0.6,
               label=L("빈 칸을 채운 값", "filled value", ko))
    ax.set_xlabel(L("공부시간(시간)", "study hours", ko), fontsize=11)
    ax.set_ylabel(L("수면시간(시간)", "sleep hours", ko), fontsize=11, labelpad=8)
    ax.set_title(
        L(f"정제 후 — {len(cleaned)}행", f"after cleaning - {len(cleaned)} rows", ko),
        fontsize=13, pad=10, color=GREEN,
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              fontsize=10, frameon=False)

    # --- (3) 문제 유형별 찾음 / 고침 / 남음 ---
    ax = axes[1, 0]
    names = [
        L("①중복 행", "1. dup", ko),
        L("②문자열\n숫자", "2. text", ko),
        L("③범위\n벗어남", "3. range", ko),
        L("④특성 결측\n(공부시간)", "4. missing\nfeature", ko),
        L("⑤라벨 결측\n(시험점수)", "5. missing\nlabel", ko),
    ]
    x = np.arange(len(names))
    width = 0.27
    series = [
        (L("찾음", "found", ko), counts["찾음"], GRAY),
        (L("고침", "fixed", ko), counts["고침"], GREEN),
        (L("남음", "left", ko), counts["남음"], RED),
    ]
    for k, (name, values, color) in enumerate(series):
        bars = ax.bar(x + (k - 1) * width, values, width, color=color, label=name)
        for rect, value in zip(bars, values):
            if value > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, value + 0.25,
                        str(int(value)), ha="center", fontsize=9.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, float(counts["찾음"].max()) + 2.4)
    ax.set_ylabel(L("건수", "cases", ko), fontsize=11, labelpad=8)
    ax.set_title(L("흠집을 몇 건 찾아 몇 건 고쳤는가",
                   "found, fixed, left by problem type", ko),
                 fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              fontsize=10, frameon=False)

    # --- (4) 결측 처리 방법에 따른 평균 ---
    ax = axes[1, 1]
    method_labels = [
        L("원본\n(참값)", "truth", ko),
        L("0으로\n채움", "fill 0", ko),
        L("결측 행\n제외", "drop", ko),
        L("중앙값으로\n채움", "fill median", ko),
    ]
    values = means["평균 공부시간"].to_numpy()
    colors = [GRAY, RED, BLUE, GREEN]
    bars = ax.bar(method_labels, values, color=colors, width=0.62)
    ax.axhline(values[0], color=GRAY, linestyle="--", linewidth=1.5)
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 0.22,
                f"{value:.2f}", ha="center", fontsize=11, color="#333333")
    ax.set_ylim(0, max(values) * 1.34)
    ax.set_ylabel(L("평균 공부시간", "mean study hours", ko),
                  fontsize=11, labelpad=10)
    ax.set_title(L("결측을 0으로 채우면 평균이 내려간다",
                   "filling missing with 0 drags the mean down", ko),
                 fontsize=13, pad=10)
    gap = values[1] - values[0]
    ax.text(1, values[1] / 2,
            L(f"{gap:+.2f}시간", f"{gap:+.2f} h", ko),
            ha="center", fontsize=12, color="white")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        L("흠집 난 표를 고치기 전과 고친 후 (합성 데이터)",
          "Before and after cleaning (synthetic data)", ko),
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.subplots_adjust(hspace=0.52, wspace=0.30)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    rng = np.random.default_rng(RANDOM_SEED)

    print("=== 실습 3.2 흠집 난 표를 고친다 ===")
    _krfont.report(ko)
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 설문 응답 아님)")
    print(f"결측을 채우는 방법: {FILL_STRATEGY}")
    print()

    # 0단계 --------------------------------------------------------------
    truth = make_clean_table(rng)
    dirty = make_dirty_table(truth, rng)

    dirty_path = RESULTS_DIR / "week03-dirty.csv"
    dirty.to_csv(dirty_path, index=False, encoding="utf-8-sig")
    loaded = pd.read_csv(dirty_path)

    print("=== 0단계. 흠집 난 표 만들기 ===")
    print(f"원본 {N_STUDENTS}행에 낸 흠집:")
    print(f"  중복 행 {N_DUPLICATE_ROWS}건 / 문자열 점수 {N_TEXT_SCORE}건 / "
          f"수면시간 이상치 {N_OUTLIER_SLEEP}건 / "
          f"공부시간 결측 {N_MISSING_STUDY}건 / 시험점수 결측 {N_MISSING_SCORE}건")
    print(f"저장: results/{dirty_path.name}")
    print(f"다시 읽은 표: {loaded.shape[0]}행 x {loaded.shape[1]}열")
    print()
    print("열마다 어떤 자료형으로 읽혔는가")
    print(loaded.dtypes.to_string())
    print("시험점수 열이 object다. 숫자가 아니라 글자 덩어리로 읽혔다.")
    print()
    print("표 앞부분 8행")
    print(loaded.head(8).to_string(index=False))
    print()

    ledger = []

    # 1단계 --------------------------------------------------------------
    step1, found, fixed = fix_duplicates(loaded)
    ledger.append(("① 중복 행", found, fixed))
    print("=== 1단계. 중복 행 ===")
    print(f"찾음 {found}건 / 고침 {fixed}건 / 남음 {found - fixed}건")
    print(f"{len(loaded)}행 -> {len(step1)}행")
    print("한 사람을 두 번 세면 그 사람의 응답이 두 배로 반영된다.")
    print()

    # 2단계 --------------------------------------------------------------
    step2, found, fixed = fix_text_numbers(step1, "시험점수")
    ledger.append(("② 문자열로 들어온 숫자", found, fixed))
    print("=== 2단계. 문자열로 들어온 숫자 ===")
    print(f"찾음 {found}건 / 고침 {fixed}건 / 남음 {found - fixed}건")
    print(f'"82점" 처럼 단위가 붙은 칸에서 단위를 떼고 숫자로 바꿨다.')
    print(f"시험점수 열의 자료형: object -> {step2['시험점수'].dtype}")
    print()

    # 3단계 --------------------------------------------------------------
    step3, found, fixed, detail = fix_outliers(step2)
    ledger.append(("③ 범위를 벗어난 값", found, fixed))
    print("=== 3단계. 범위를 벗어난 이상치 ===")
    print("있을 수 있는 값의 범위")
    for column, (low, high) in VALID_RANGE.items():
        print(f"  {column}: {low} ~ {high}  (벗어난 칸 {detail.get(column, 0)}건)")
    print(f"찾음 {found}건 / 고침 {fixed}건 / 남음 {found - fixed}건")
    print("벗어난 값은 그 열에서 범위 안에 있는 값의 중앙값으로 바꿨다.")
    print()

    # 4단계 --------------------------------------------------------------
    before_fill = step3.copy()
    cleaned, missing_report = fix_missing(step3)
    ledger.append(("④ 특성 결측(공부시간)",
                   missing_report["특성결측_찾음"], missing_report["특성결측_고침"]))
    ledger.append(("⑤ 라벨 결측(시험점수)",
                   missing_report["라벨결측_찾음"], missing_report["라벨결측_고침"]))
    print("=== 4단계. 결측값 ===")
    print(f"공부시간(특성) 찾음 {missing_report['특성결측_찾음']}건 / "
          f"고침 {missing_report['특성결측_고침']}건 / 남음 0건")
    print(f"  채운 값: {missing_report['채운값']}")
    print(f"시험점수(라벨) 찾음 {missing_report['라벨결측_찾음']}건 / 고침 0건 / "
          f"남음 {missing_report['라벨결측_찾음']}건")
    print("  라벨은 맞혀야 하는 정답이다. 정답을 지어내지 않는다. 그래서 남긴다.")
    print()

    # 총계 ---------------------------------------------------------------
    counts = pd.DataFrame(ledger, columns=["문제 유형", "찾음", "고침"])
    counts["남음"] = counts["찾음"] - counts["고침"]
    total_found = int(counts["찾음"].sum())
    total_fixed = int(counts["고침"].sum())
    total_left = int(counts["남음"].sum())

    print("=== 총계 ===")
    total_row = pd.DataFrame(
        [["합계", total_found, total_fixed, total_left]], columns=counts.columns
    )
    print(pd.concat([counts, total_row], ignore_index=True).to_string(index=False))
    print()
    print(f"찾은 {total_found}건 = 고친 {total_fixed}건 + 남은 {total_left}건")
    assert total_found == total_fixed + total_left, "찾음, 고침, 남음의 합이 맞지 않는다"
    print("합이 맞는다.")
    print()

    usable = cleaned["시험점수"].notna().sum()
    print(f"정제 후 표: {len(cleaned)}행")
    print(f"그중 라벨이 있어 학습에 쓸 수 있는 행: {usable}행")
    print(f"라벨이 없어 학습에 쓸 수 없는 행: {len(cleaned) - usable}행")
    print()

    # 결측을 0으로 채우면 -------------------------------------------------
    means = mean_comparison(truth, before_fill)
    print("=== 결측을 0으로 채우면 생기는 일 ===")
    print(f"공부시간 열: 값이 있는 칸 {int(before_fill['공부시간'].notna().sum())}개, "
          f"빈 칸 {int(before_fill['공부시간'].isna().sum())}개")
    print()
    formatted = means.copy()
    formatted["평균 공부시간"] = formatted["평균 공부시간"].map(lambda v: f"{v:.2f}")
    formatted["참값과의 차이"] = formatted["참값과의 차이"].map(lambda v: f"{v:+.2f}")
    formatted["어긋난 정도(%)"] = formatted["어긋난 정도(%)"].map(lambda v: f"{v:+.1f}")
    print(formatted.to_string(index=False))
    print()
    print("0은 '값이 없다'는 뜻이 아니라 '0시간 공부했다'는 뜻이다.")
    print("빈 칸을 0으로 채우면 공부를 한 시간도 안 한 학생을 새로 만들어 낸 셈이 된다.")
    print("이 실습은 빈 칸을 무작위로 만들었기 때문에 결측 행을 뺀 평균이 참값에 가깝다.")
    print("특정 집단만 답을 하지 않았다면 결측 행을 빼도 평균이 어긋난다.")
    print()

    # 저장 ---------------------------------------------------------------
    clean_path = RESULTS_DIR / "week03-cleaned.csv"
    cleaned.drop(columns=["공부시간채움"]).to_csv(
        clean_path, index=False, encoding="utf-8-sig"
    )
    counts_path = RESULTS_DIR / "week03-quality-report.csv"
    pd.concat([counts, total_row], ignore_index=True).to_csv(
        counts_path, index=False, encoding="utf-8-sig"
    )

    plot_path = RESULTS_DIR / "data-cleaning-before-after.png"
    draw_before_after(ko, loaded, cleaned, counts, means, plot_path)

    print("=== 저장된 결과 ===")
    print(f"흠집 난 표: results/{dirty_path.name}")
    print(f"정제한 표: results/{clean_path.name}")
    print(f"점검 결과 표: results/{counts_path.name}")
    print(f"그림 파일: results/{plot_path.name}")
    print()
    print('파일 맨 위의 FILL_STRATEGY 를 "zero" 로 바꿔 다시 실행하면')
    print("정제 후 그림에서 채운 점들이 왼쪽 벽에 달라붙는 것을 볼 수 있다.")


if __name__ == "__main__":
    main()
