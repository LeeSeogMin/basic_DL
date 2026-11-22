"""실습 12.1 - 같은 명령 12개를 권한 설정 셋으로 판정한다.

코딩 에이전트는 코드를 쓰기만 하지 않는다. 파일을 읽고, 쓰고, 지우고,
인터넷에 접속하는 명령을 직접 실행한다. 그래서 에이전트 도구에는
"어떤 명령은 바로 실행하고, 어떤 명령은 사람에게 먼저 묻고,
어떤 명령은 막는다"를 정하는 권한 설정이 들어 있다.

이 코드는 그 판정을 작게 흉내 낸다.

[입력]  에이전트가 실행하려는 명령 12개. 아래 COMMANDS 에 적어 둔 합성 예제다.
        실제 에이전트의 기록이 아니다.
[출력]  1. 권한 설정 셋으로 판정한 표를 화면에 출력한다.
        2. 같은 표를 results/permission_table.csv 로 저장한다.
        3. 설정별 건수를 results/permission_report.png 막대그래프로 그린다.
[하면 안 되는 일]
        - 목록의 명령을 실제로 실행하지 않는다. 글자로 읽고 판정만 한다.
        - 인터넷에 접속하지 않는다.
[실패 시 동작]
        - 모르는 동작 이름이 목록에 있으면 ValueError 를 내고 멈춘다.

판정 규칙은 수업용으로 줄인 것이다. 실제 제품의 규칙은 더 세밀하고 제품마다 다르다.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

import _krfont

# ===================== 학생이 바꿔 볼 값 =====================
WORKSPACE = "my-project"      # 에이전트에게 맡긴 작업 폴더 이름
NETWORK_ALLOWED = False       # True 로 바꾸면 "작업 폴더 쓰기" 설정에서 인터넷 명령을 묻지 않는다

# 에이전트가 실행하려는 명령 목록(합성 예제).
# (동작, 대상, 설명) 순서다. 동작은 읽기·쓰기·삭제·전송·설치 다섯 가지다.
COMMANDS = [
    ("읽기", "my-project/README.md",              "안내 파일을 읽는다"),
    ("읽기", "my-project/data/scores.csv",        "점수 표를 읽는다"),
    ("쓰기", "my-project/report.py",              "새 코드 파일을 만든다"),
    ("쓰기", "my-project/results/chart.png",      "그림을 저장한다"),
    ("쓰기", "my-project/data/scores.csv",        "원본 점수 표를 덮어쓴다"),
    ("삭제", "my-project/results/old_chart.png",  "예전 그림을 지운다"),
    ("삭제", "my-project/data",                   "데이터 폴더를 통째로 지운다"),
    ("읽기", "~/.ssh/id_rsa",                     "작업 폴더 밖의 비밀키 파일을 읽는다"),
    ("쓰기", "~/.zshrc",                          "작업 폴더 밖의 설정 파일을 고친다"),
    ("삭제", "~/Documents",                       "작업 폴더 밖의 문서 폴더를 지운다"),
    ("전송", "https://example.com/upload",        "파일 내용을 인터넷으로 보낸다"),
    ("설치", "pip install some-package",          "인터넷에서 패키지를 받아 설치한다"),
]
# ============================================================

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

SETTINGS = ["읽기 전용", "작업 폴더 쓰기", "전부 허용"]
RUN, ASK, BLOCK = "바로 실행", "사람에게 묻기", "차단"

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


# ---------------------------------------------------------------- 판정 규칙
def is_inside(target: str) -> bool:
    """대상 경로가 작업 폴더 안에 있는지 글자로만 확인한다. 실제 파일은 건드리지 않는다."""
    parts = PurePosixPath(target).parts
    return bool(parts) and parts[0] == WORKSPACE


def is_risky(action: str, target: str, note: str) -> bool:
    """되돌리기 어렵거나 밖으로 새는 명령인지 표시한다.

    삭제·전송·설치, 작업 폴더 밖을 건드리는 명령, 원본을 덮어쓰는 명령을 위험으로 본다.
    """
    if action in ("삭제", "전송", "설치"):
        return True
    if action in ("읽기", "쓰기") and not is_inside(target):
        return True
    return "덮어" in note


def decide(setting: str, action: str, target: str) -> str:
    """권한 설정 하나에서 명령 하나를 어떻게 처리할지 정한다."""
    if action not in ("읽기", "쓰기", "삭제", "전송", "설치"):
        raise ValueError(f"모르는 동작입니다: {action}")

    if setting == "전부 허용":
        return RUN                       # 아무것도 묻지 않는다

    uses_network = action in ("전송", "설치")
    inside = (not uses_network) and is_inside(target)

    if setting == "읽기 전용":
        if action == "읽기":
            return RUN if inside else ASK
        return BLOCK                     # 읽기 말고는 전부 막는다

    # 여기부터 "작업 폴더 쓰기"
    if uses_network:
        return RUN if NETWORK_ALLOWED else ASK
    if inside and action in ("읽기", "쓰기"):
        return RUN                       # 작업 폴더 안에서 읽고 쓰는 일은 묻지 않는다
    return ASK                           # 삭제와 작업 폴더 밖은 사람에게 묻는다


def build_table() -> pd.DataFrame:
    rows = []
    for number, (action, target, note) in enumerate(COMMANDS, start=1):
        row = {"번호": number, "동작": action, "대상": target, "설명": note,
               "위험": "예" if is_risky(action, target, note) else "아니오"}
        for setting in SETTINGS:
            row[setting] = decide(setting, action, target)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """설정마다 판정 건수와, 사람 확인 없이 실행되는 위험 명령 수를 센다."""
    rows = []
    for setting in SETTINGS:
        column = table[setting]
        unchecked = int(((column == RUN) & (table["위험"] == "예")).sum())
        rows.append({
            "설정": setting,
            RUN: int((column == RUN).sum()),
            ASK: int((column == ASK).sum()),
            BLOCK: int((column == BLOCK).sum()),
            "확인 없이 실행되는 위험 명령": unchecked,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 그림
def draw_report(summary: pd.DataFrame, ko: bool, out_path: Path) -> None:
    L = _krfont.label
    names = [L(s, e, ko) for s, e in zip(SETTINGS, ["read-only", "workspace-write", "allow all"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax = axes[0]
    bottom = [0] * len(names)
    parts = [(RUN, GREEN, "run"), (ASK, ORANGE, "ask the human"), (BLOCK, GRAY, "block")]
    for column, color, english in parts:
        values = list(summary[column])
        ax.bar(names, values, bottom=bottom, color=color, label=L(column, english, ko), width=0.6)
        for x, (value, base) in enumerate(zip(values, bottom)):
            if value:
                ax.text(x, base + value / 2, str(value), ha="center", va="center",
                        fontsize=12, color="white")
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_title(L("명령 12개를 어떻게 처리하는가", "how the 12 commands are handled", ko),
                 fontsize=13, pad=10)
    ax.set_ylabel(L("명령 수 (개)", "commands", ko), fontsize=11)
    ax.set_ylim(0, len(COMMANDS) * 1.3)
    ax.legend(fontsize=10, frameon=False, ncol=3, loc="upper center")

    ax = axes[1]
    values = list(summary["확인 없이 실행되는 위험 명령"])
    ax.bar(names, values, color=RED, width=0.6)
    for x, value in enumerate(values):
        ax.text(x, value + 0.15, str(value), ha="center", fontsize=12, color="#333333")
    ax.set_title(L("사람 확인 없이 실행되는 위험 명령", "risky commands run without asking", ko),
                 fontsize=13, pad=10)
    ax.set_ylabel(L("명령 수 (개)", "commands", ko), fontsize=11)
    ax.set_ylim(0, max(max(values), 1) * 1.3)

    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=11, length=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(L("권한 설정별 판정 결과 (합성 명령 목록)",
                   "decisions per permission setting (synthetic command list)", ko), fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 설정 ===")
    print(f"작업 폴더: {WORKSPACE}")
    print(f"인터넷 허용(작업 폴더 쓰기 설정): {NETWORK_ALLOWED}")
    print(f"명령 수: {len(COMMANDS)}개 (합성 예제. 한 줄도 실행하지 않고 판정만 한다)")
    print()

    table = build_table()
    print("=== 명령별 판정 ===")
    print_table(
        ["번호", "동작", "대상", "위험"] + SETTINGS,
        [[str(r["번호"]), r["동작"], r["대상"], r["위험"]] + [r[s] for s in SETTINGS]
         for _, r in table.iterrows()],
        right={0},
    )
    print()

    summary = summarize(table)
    print("=== 설정별 집계 ===")
    print_table(
        list(summary.columns),
        [[str(v) for v in row] for row in summary.itertuples(index=False)],
        right={1, 2, 3, 4},
    )
    risky_total = int((table["위험"] == "예").sum())
    print(f"위험 표시가 붙은 명령: {len(COMMANDS)}개 중 {risky_total}개")
    print()

    csv_path = RESULTS_DIR / "permission_table.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    png_path = RESULTS_DIR / "permission_report.png"
    draw_report(summary, ko, png_path)
    print(f"저장: results/{csv_path.name}")
    print(f"저장: results/{png_path.name}")
    print()
    print("목록의 명령은 실행하지 않았다. 이 표는 수업용으로 줄인 규칙으로 판정한 결과다.")


if __name__ == "__main__":
    main()
