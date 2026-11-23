"""실습 13.2 - 앱이 화면에 적는 확률을 실제로 맞은 비율과 대조한다.

실습 13.1의 앱은 "대여가 많은 날일 확률 0.966" 같은 숫자를 화면에 적는다.
사용자는 이 숫자를 보고 답을 얼마나 믿을지 정한다. 그래서 확인할 것이 하나 있다.
앱이 0.9라고 적은 답은 실제로 열 번 중 아홉 번쯤 맞는가.

이 코드는 그 질문을 두 묶음의 데이터로 확인한다.
  - 배운 범위 안: 봄·가을 테스트 데이터 (모델이 배운 것과 같은 기온 범위)
  - 새 데이터:    여름 날씨 (기온이 배운 범위보다 높은 날이 섞여 있다)

[입력]  합성 날씨 데이터(_bikedata.py 가 만든다. 실제 대여 기록이 아니다).
[출력]  1. results/confidence_table.csv  확신 구간별 건수와 맞은 비율.
        2. results/confidence_check.png  같은 결과를 그린 그림.
[하면 안 되는 일]
        - 새 데이터로 모델을 다시 학습시키지 않는다. 이미 만든 앱을 점검하는 것이 목적이다.
[실패 시 동작]
        - 새 데이터가 한 건도 없으면 오류를 내고 멈춘다.

용어: 확신 = 앱이 고른 답 쪽의 확률. 확률이 0.2면 "적은 날"이라고 답하고 확신은 0.8이다.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bikedata
import _krfont

# ===================== 학생이 바꿔 볼 값 =====================
NEW_TEMP = (22.0, 38.0)   # 새 데이터의 기온 범위(도). (3.0, 26.0)으로 바꾸면 배운 범위와 같아진다
NEW_DAYS = 600            # 새 데이터 날 수
NEW_SEED = 7              # 새 데이터를 만드는 난수 seed
HIGH_CONFIDENCE = 0.9     # 이 값 이상이면 "확신이 높다"고 센다
# ============================================================

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BIN_EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BIN_NAMES = ["0.5~0.6", "0.6~0.7", "0.7~0.8", "0.8~0.9", "0.9~1.0"]

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
GRAY = "#79706E"


# ---------------------------------------------------------------- 표 출력 도우미
def _width(text: str) -> int:
    """한글은 두 칸, 영문·숫자는 한 칸으로 세어 글자 폭을 구한다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def print_table(headers: list[str], rows: list[list[str]], right: set[int] | None = None) -> None:
    """칸을 맞춰 표를 출력한다. right 에 든 열 번호는 오른쪽 정렬한다."""
    right = right or set()
    widths = [max(_width(headers[i]), *(_width(r[i]) for r in rows)) for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            pad = " " * (widths[i] - _width(cell))
            out.append(pad + cell if i in right else cell + pad)
        return "  ".join(out).rstrip()

    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))


# ---------------------------------------------------------------- 계산
def answer_frame(model, x: np.ndarray, y: np.ndarray, ranges: list[tuple[float, float]]) -> pd.DataFrame:
    """날마다 앱의 답, 확신, 맞았는지, 범위 밖 경고가 붙는지를 표로 만든다."""
    prob = model.predict_proba(x)[:, 1]
    answer = (prob >= 0.5).astype(int)
    outside = np.zeros(len(x), dtype=bool)
    for i, (low, high) in enumerate(ranges):
        outside |= (x[:, i] < low) | (x[:, i] > high)
    return pd.DataFrame({
        "확신": np.maximum(prob, 1.0 - prob),
        "맞음": answer == y,
        "경고": outside,
    })


def by_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    """확신 구간별로 건수, 평균 확신, 맞은 비율을 센다."""
    bins = pd.cut(frame["확신"], BIN_EDGES, labels=BIN_NAMES, right=False, include_lowest=True)
    bins = bins.fillna(BIN_NAMES[-1])          # 확신이 정확히 1.0인 답은 마지막 구간에 넣는다
    rows = []
    for name in BIN_NAMES:
        part = frame[bins == name]
        rows.append({
            "확신 구간": name,
            "건수": len(part),
            "평균 확신": part["확신"].mean() if len(part) else np.nan,
            "맞은 비율": part["맞음"].mean() if len(part) else np.nan,
        })
    return pd.DataFrame(rows)


def show_bins(table: pd.DataFrame) -> None:
    def fmt(value: float) -> str:
        return "-" if np.isnan(value) else f"{value:.3f}"

    print_table(
        ["확신 구간", "건수", "평균 확신", "맞은 비율"],
        [[r["확신 구간"], str(r["건수"]), fmt(r["평균 확신"]), fmt(r["맞은 비율"])]
         for _, r in table.iterrows()],
        right={1, 2, 3},
    )


def summary_row(name: str, frame: pd.DataFrame) -> dict:
    high = frame[frame["확신"] >= HIGH_CONFIDENCE]
    return {
        "묶음": name,
        "건수": len(frame),
        "평균 확신": frame["확신"].mean() if len(frame) else np.nan,
        "맞은 비율": frame["맞음"].mean() if len(frame) else np.nan,
        "확신 높은 답": len(high),
        "그중 틀린 답": int((~high["맞음"]).sum()),
    }


# ---------------------------------------------------------------- 그림
def draw(inside: pd.DataFrame, outside: pd.DataFrame, groups: pd.DataFrame,
         ko: bool, out_path: Path) -> None:
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))

    ax = axes[0]
    ax.plot([0.5, 1.0], [0.5, 1.0], color="#999999", linewidth=1, linestyle="--")
    ax.text(0.52, 0.56, L("표시한 확신 = 맞은 비율", "confidence = accuracy", ko),
            fontsize=9.5, color="#666666", rotation=38)
    for table, color, name in [
        (inside, BLUE, L("배운 범위 안 (테스트)", "inside learned range (test)", ko)),
        (outside, ORANGE, L("새 데이터 (여름)", "new data (summer)", ko)),
    ]:
        part = table[table["건수"] > 0]
        ax.plot(part["평균 확신"], part["맞은 비율"], marker="o", color=color, linewidth=2, label=name)
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel(L("앱이 화면에 표시한 확신", "confidence shown by the app", ko), fontsize=11)
    ax.set_ylabel(L("실제로 맞은 비율", "share actually correct", ko), fontsize=11)
    ax.set_title(L("표시한 확신만큼 실제로 맞았는가", "does shown confidence match accuracy?", ko),
                 fontsize=13, pad=10)
    ax.legend(fontsize=10, frameon=False, loc="lower left")

    ax = axes[1]
    english = ["test\n(inside range)", "new data\nno warning", "new data\nwith warning"]
    names = [L(row["묶음"].replace(" / ", "\n"), english[i], ko) for i, row in groups.iterrows()]
    x_pos = np.arange(len(groups))
    ax.bar(x_pos - 0.18, groups["평균 확신"], width=0.36, color=GRAY,
           label=L("평균 확신", "mean confidence", ko))
    ax.bar(x_pos + 0.18, groups["맞은 비율"], width=0.36, color=GREEN,
           label=L("맞은 비율", "share correct", ko))
    for x, (_, row) in zip(x_pos, groups.iterrows()):
        if row["건수"] == 0:
            continue
        ax.text(x - 0.18, row["평균 확신"] + 0.015, f'{row["평균 확신"]:.3f}', ha="center", fontsize=10)
        ax.text(x + 0.18, row["맞은 비율"] + 0.015, f'{row["맞은 비율"]:.3f}', ha="center", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylim(0, 1.18)
    ax.set_title(L("범위 밖 경고가 붙은 답과 안 붙은 답", "answers with and without the range warning", ko),
                 fontsize=13, pad=10)
    ax.legend(fontsize=10, frameon=False, loc="upper center", ncol=2)

    for ax in axes:
        ax.tick_params(labelsize=10, length=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(L("화면의 확신과 실제 맞은 비율 (합성 데이터)",
                   "shown confidence vs. accuracy (synthetic data)", ko), fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    model, x_train, x_test, y_train, y_test = _bikedata.train_model()
    ranges = _bikedata.learned_ranges(x_train)
    x_new, y_new = _bikedata.make_days(NEW_DAYS, NEW_TEMP, NEW_SEED)
    if len(y_new) == 0:
        raise RuntimeError("새 데이터가 없다. NEW_DAYS 를 1 이상으로 정한다.")

    print("=== 1. 데이터 두 묶음 ===")
    print("데이터: 컴퓨터로 만든 합성 날씨 데이터(실제 대여 기록 아님)")
    print(f"모델이 배운 기온 범위: {ranges[0][0]:g}~{ranges[0][1]:g}도 (훈련 {len(y_train)}일)")
    print(f"배운 범위 안: 테스트 {len(y_test)}일")
    print(f"새 데이터: 기온 {NEW_TEMP[0]:g}~{NEW_TEMP[1]:g}도인 {len(y_new)}일 (seed {NEW_SEED})")
    print("새 데이터로 모델을 다시 학습시키지 않았다.")
    print()

    test = answer_frame(model, x_test, y_test, ranges)
    new = answer_frame(model, x_new, y_new, ranges)

    inside_bins = by_confidence(test)
    print(f"=== 2. 배운 범위 안: 테스트 {len(test)}일 ===")
    show_bins(inside_bins)
    print()

    new_bins = by_confidence(new)
    print(f"=== 3. 새 데이터: {len(new)}일 ===")
    show_bins(new_bins)
    print()

    groups = pd.DataFrame([
        summary_row("테스트 / 배운 범위 안", test),
        summary_row("새 데이터 / 경고 없음", new[~new["경고"]]),
        summary_row("새 데이터 / 경고 있음", new[new["경고"]]),
    ])
    print(f"=== 4. 범위 밖 경고가 붙은 답과 안 붙은 답 (확신 높은 답 = 확신 {HIGH_CONFIDENCE:g} 이상) ===")
    print_table(
        ["묶음", "건수", "평균 확신", "맞은 비율", "확신 높은 답", "그중 틀린 답"],
        [[r["묶음"], str(r["건수"]),
          "-" if r["건수"] == 0 else f'{r["평균 확신"]:.3f}',
          "-" if r["건수"] == 0 else f'{r["맞은 비율"]:.3f}',
          str(r["확신 높은 답"]), str(r["그중 틀린 답"])] for _, r in groups.iterrows()],
        right={1, 2, 3, 4, 5},
    )
    print()

    print("=== 5. 집계 ===")
    print(f"새 데이터 전체: 맞은 비율 {new['맞음'].mean():.3f}, 평균 확신 {new['확신'].mean():.3f}")
    print(f"새 데이터 {len(new)}일 중 범위 밖 경고가 붙는 날: {int(new['경고'].sum())}일")
    wrong = new[~new["맞음"]]
    print(f"새 데이터에서 틀린 답 {len(wrong)}건 중 경고가 붙은 답: {int(wrong['경고'].sum())}건")
    print()

    table = pd.concat([inside_bins.assign(묶음="배운 범위 안"), new_bins.assign(묶음="새 데이터")])
    csv_path = RESULTS_DIR / "confidence_table.csv"
    table[["묶음", "확신 구간", "건수", "평균 확신", "맞은 비율"]].round(3).to_csv(
        csv_path, index=False, encoding="utf-8-sig")
    png_path = RESULTS_DIR / "confidence_check.png"
    draw(inside_bins, new_bins, groups, ko, png_path)
    print(f"저장: results/{csv_path.name}")
    print(f"저장: results/{png_path.name}")


if __name__ == "__main__":
    main()
