"""실습 12.2 - 에이전트의 보고와 실제 변경을 대조하고, 저장 지점으로 되돌린다.

코딩 에이전트는 일을 마치면 "무엇을 했다"는 보고를 글로 쓴다.
보고는 에이전트가 쓴 글이고, 실제 변경은 파일에 남는다. 둘이 같다는 보장은 없다.
이 코드는 둘을 따로 구해 한 줄씩 대조한다.

[입력]  practice/chapter12/sample_folder/ (합성 예제 폴더, 파일 11개)
        AGENT_ACTIONS  에이전트가 했다고 가정한 작업 목록(합성)
        AGENT_REPORT   에이전트가 썼다고 가정한 최종 보고(합성)
        실제 AI를 호출하지 않는다. 두 목록 모두 사람이 미리 적어 둔 것이다.
[출력]  1. 실제 변경과 보고를 대조한 표를 화면에 출력한다.
        2. 같은 표를 results/change_table.csv 로 저장한다.
        3. 건수를 results/report_vs_record.png 막대그래프로 그린다.
[하면 안 되는 일]
        - 원본 sample_folder 를 고치지 않는다. 임시 폴더에 복사한 뒤 복사본만 바꾼다.
        - 인터넷에 접속하지 않는다.
[실패 시 동작]
        - 원본 폴더가 없으면 FileNotFoundError 를 내고 멈춘다.
        - 작업 목록에 없는 파일을 지우거나 고치라고 적혀 있으면 FileNotFoundError 를 낸다.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

import _krfont

# ===================== 학생이 바꿔 볼 값 =====================
USE_SAVEPOINT = True          # False 로 바꾸면 저장 지점 없이 작업을 시작한다

# 에이전트가 복사본 폴더에서 한 작업(합성 예제). (종류, 경로) 순서다.
AGENT_ACTIONS = [
    ("생성", "summary.md"),
    ("수정", "data/scores.csv"),
    ("수정", "config.json"),
    ("삭제", "logs/train_run1.log"),
    ("삭제", "notes/todo.md"),
]

# 에이전트가 일을 마치고 쓴 최종 보고(합성 예제). (종류, 경로) 순서다.
AGENT_REPORT = [
    ("생성", "summary.md"),
    ("수정", "data/scores.csv"),
    ("삭제", "logs/train_run1.log"),
    ("수정", "README.txt"),
]
# ============================================================

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"
SOURCE_FOLDER = CHAPTER_DIR / "sample_folder"

KINDS = ["생성", "수정", "삭제"]
MATCH, MISSING, ONLY_REPORT = "일치", "보고에서 빠짐", "보고에만 있음"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


# ---------------------------------------------------------------- 표 출력 도우미
def _width(text: str) -> int:
    """한글은 두 칸, 영문·숫자는 한 칸으로 세어 글자 폭을 구한다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def print_table(headers: list[str], rows: list[list[str]], right: set[int] | None = None) -> None:
    """칸을 맞춰 표를 출력한다. right 에 든 열 번호는 오른쪽 정렬한다."""
    right = right or set()
    widths = [max(_width(headers[i]), *(_width(r[i]) for r in rows)) if rows
              else _width(headers[i]) for i in range(len(headers))]

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


# ---------------------------------------------------------------- 기록 만들기
def snapshot(folder: Path) -> dict[str, str]:
    """폴더 안 모든 파일의 내용 해시를 적는다. 해시는 내용이 한 글자만 달라도 바뀌는 값이다."""
    state = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            state[path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return state


def apply_actions(folder: Path) -> None:
    """합성 작업 목록을 복사본 폴더에 그대로 적용한다."""
    for kind, name in AGENT_ACTIONS:
        path = folder / name
        if kind == "생성":
            path.write_text("# 요약\n에이전트가 만든 파일(합성 예제)\n", encoding="utf-8")
        elif kind == "수정":
            if not path.is_file():
                raise FileNotFoundError(f"수정할 파일이 없습니다: {name}")
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n# 에이전트가 덧붙인 줄(합성 예제)\n")
        elif kind == "삭제":
            if not path.is_file():
                raise FileNotFoundError(f"지울 파일이 없습니다: {name}")
            path.unlink()                # 복사본 안의 파일만 지운다
        else:
            raise ValueError(f"모르는 작업 종류입니다: {kind}")


def diff(before: dict[str, str], after: dict[str, str]) -> list[tuple[str, str]]:
    """두 기록을 비교해 실제로 바뀐 파일 목록을 만든다."""
    changes = []
    for name in sorted(set(before) | set(after)):
        if name not in before:
            changes.append(("생성", name))
        elif name not in after:
            changes.append(("삭제", name))
        elif before[name] != after[name]:
            changes.append(("수정", name))
    return changes


def compare(record: list[tuple[str, str]], report: list[tuple[str, str]]) -> pd.DataFrame:
    """실제 변경(기록)과 보고를 파일 하나씩 대조한다."""
    recorded = {name: kind for kind, name in record}
    reported = {name: kind for kind, name in report}
    rows = []
    for name in sorted(set(recorded) | set(reported)):
        real, said = recorded.get(name, "없음"), reported.get(name, "없음")
        if real == said:
            verdict = MATCH
        elif said == "없음":
            verdict = MISSING
        elif real == "없음":
            verdict = ONLY_REPORT
        else:
            verdict = "종류가 다름"
        rows.append({"파일": name, "실제 변경": real, "보고": said, "판정": verdict})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 그림
def draw_report(record, report, table: pd.DataFrame, ko: bool, out_path: Path) -> None:
    L = _krfont.label
    english = {"생성": "created", "수정": "modified", "삭제": "deleted"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax = axes[0]
    real = [sum(1 for k, _ in record if k == kind) for kind in KINDS]
    said = [sum(1 for k, _ in report if k == kind) for kind in KINDS]
    xs = range(len(KINDS))
    ax.bar([x - 0.2 for x in xs], real, width=0.4, color=BLUE,
           label=L("실제 변경(파일에서 계산)", "actual (from files)", ko))
    ax.bar([x + 0.2 for x in xs], said, width=0.4, color=ORANGE,
           label=L("에이전트의 보고", "agent's report", ko))
    for x, (a, b) in enumerate(zip(real, said)):
        ax.text(x - 0.2, a + 0.08, str(a), ha="center", fontsize=12)
        ax.text(x + 0.2, b + 0.08, str(b), ha="center", fontsize=12)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([L(k, english[k], ko) for k in KINDS])
    ax.set_title(L("종류별 건수: 실제와 보고", "counts by kind: actual vs report", ko),
                 fontsize=13, pad=10)
    ax.set_ylim(0, max(real + said + [1]) * 1.6)
    ax.legend(fontsize=10, frameon=False, loc="upper left")

    ax = axes[1]
    verdicts = [MATCH, MISSING, ONLY_REPORT]
    counts = [int((table["판정"] == v).sum()) for v in verdicts]
    names = [L(v, e, ko) for v, e in zip(verdicts, ["match", "missing in report", "report only"])]
    ax.bar(names, counts, color=[GREEN, RED, GRAY], width=0.55)
    for x, value in enumerate(counts):
        ax.text(x, value + 0.08, str(value), ha="center", fontsize=12)
    ax.set_title(L("파일별 대조 결과", "per-file comparison", ko), fontsize=13, pad=10)
    ax.set_ylim(0, max(counts + [1]) * 1.6)

    for ax in axes:
        ax.set_ylabel(L("파일 수 (개)", "files", ko), fontsize=11)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=11, length=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(L("에이전트의 보고와 실제 변경 대조 (합성 예제)",
                   "agent report vs actual changes (synthetic example)", ko), fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not SOURCE_FOLDER.is_dir():
        raise FileNotFoundError(f"예제 폴더를 찾을 수 없습니다: {SOURCE_FOLDER}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    original_before = snapshot(SOURCE_FOLDER)

    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp) / "work"            # 에이전트가 일하는 복사본
        savepoint = Path(temp) / "savepoint"  # 일을 시키기 전에 떠 둔 저장 지점
        shutil.copytree(SOURCE_FOLDER, work)
        if USE_SAVEPOINT:
            shutil.copytree(work, savepoint)

        before = snapshot(work)
        print("=== 1. 작업 전 ===")
        print(f"복사본 폴더의 파일 수: {len(before)}개")
        print(f"저장 지점: {'만들었다' if USE_SAVEPOINT else '만들지 않았다'}")
        print()

        apply_actions(work)
        after = snapshot(work)
        record = diff(before, after)

        print("=== 2. 실제 변경과 보고 대조 ===")
        table = compare(record, AGENT_REPORT)
        print_table(list(table.columns), [list(row) for row in table.itertuples(index=False)])
        missing = int((table["판정"] == MISSING).sum())
        only_report = int((table["판정"] == ONLY_REPORT).sum())
        matched = int((table["판정"] == MATCH).sum())
        print(f"실제 변경 {len(record)}건, 보고 {len(AGENT_REPORT)}건")
        print(f"일치 {matched}건, 보고에서 빠진 변경 {missing}건, 보고에만 있는 변경 {only_report}건")
        print(f"작업 후 파일 수: {len(after)}개")
        print()

        print("=== 3. 되돌리기 ===")
        if USE_SAVEPOINT:
            shutil.rmtree(work)               # 임시 폴더 안의 복사본만 지운다
            shutil.copytree(savepoint, work)
            restored = snapshot(work)
            different = len(diff(before, restored))
            print(f"저장 지점으로 되돌린 뒤 파일 수: {len(restored)}개")
            print(f"작업 전과 내용이 다른 파일: {different}개")
        else:
            lost = sum(1 for kind, _ in record if kind in ("수정", "삭제"))
            print("저장 지점이 없어 되돌릴 수 없다.")
            print(f"원래 내용을 되찾을 수 없는 파일: {lost}개")
        print()

    original_after = snapshot(SOURCE_FOLDER)
    print("=== 4. 원본 폴더 확인 ===")
    print(f"원본 sample_folder 에서 바뀐 파일: {len(diff(original_before, original_after))}개")
    print()

    csv_path = RESULTS_DIR / "change_table.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    png_path = RESULTS_DIR / "report_vs_record.png"
    draw_report(record, AGENT_REPORT, table, ko, png_path)
    print(f"저장: results/{csv_path.name}")
    print(f"저장: results/{png_path.name}")
    print()
    print("작업 목록과 보고는 사람이 적은 합성 예제다. 실제 에이전트의 기록이 아니다.")


if __name__ == "__main__":
    main()
