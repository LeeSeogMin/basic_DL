"""실습 14.2 - 제출 전에 프로젝트 폴더를 자동으로 점검한다.

폴더 하나를 훑어 여섯 가지를 확인하고 통과·미통과 표로 낸다.

  1  실행되는 파이썬 파일이 있는가
  2  결과 파일(그림·로그, results 폴더의 파일)이 있는가
  3  데이터 또는 입력 예시가 있는가
  4  코드에 비밀키처럼 보이는 문자열이 없는가
  5  데이터에 실명·연락처·학번처럼 보이는 문자열이 없는가
  6  사용 방법을 적은 파일이 있는가

이 도구는 파일을 읽기만 한다. 고치거나 지우지 않는다.
파이썬 파일도 실행하지 않고 문법만 검사한다.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import _krfont

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.
# 자기 프로젝트를 점검하려면 PROJECT_DIR 에 그 폴더 경로를 적는다.
PROJECT_DIR = CHAPTER_DIR / "example_project"          # 기본값: 저장소 안의 예시 폴더
COMPARE_DIR = CHAPTER_DIR / "example_project_rough"    # 문제를 일부러 넣어 둔 폴더
MAX_READ_KB = 512          # 이보다 큰 파일은 읽지 않는다
# ---------------------------------------------------------------------------

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".ipynb_checkpoints"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".csv", ".json", ".jsonl", ".yml", ".yaml",
    ".cfg", ".ini", ".env", ".html", ".js", ".ipynb", ".log",
}
OUTPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".log", ".pdf"}
OUTPUT_DIRS = {"results", "result", "output", "outputs", "figures", "figs"}
DATA_SUFFIXES = {".csv", ".json", ".jsonl", ".txt", ".tsv", ".xlsx"}
DATA_DIRS = {"data", "dataset", "datasets", "input", "inputs"}
USAGE_STEMS = {"readme", "usage", "howto", "사용법", "실행방법", "사용방법"}

SECRET_PATTERNS = [
    (re.compile(
        r"(?i)[a-z_]*(?:api[_-]?key|secret|token|password|passwd|credential)[a-z_]*"
        r"\s*[:=]\s*[\"'][^\"']{6,}[\"']"),
     "코드에 값이 그대로 적힌 키·비밀번호"),
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "키 형태의 긴 문자열"),
]

PERSONAL_PATTERNS = [
    (re.compile(r"01[016-9][-. ]?\d{3,4}[-. ]?\d{4}"), "휴대폰 번호 형태"),
    (re.compile(r"\b\d{6}-[1-4]\d{6}\b"), "주민등록번호 형태"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "이메일 주소"),
    (re.compile(r"이름|성명|학번|연락처|휴대폰|전화번호|주민등록번호"), "이름·학번을 담는 열 이름"),
    (re.compile(r"(?i)\b(?:student_?id|full_?name|phone_?number)\b"), "이름·연락처 열 이름(영문)"),
]


# ===========================================================================
# 폴더 읽기
# ===========================================================================
def walk_files(root: Path) -> list[Path]:
    """폴더 안의 파일을 모두 모은다. 작업용 폴더는 건너뛴다."""
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        found.append(path)
    return found


def read_text(path: Path) -> list[str]:
    """텍스트 파일을 줄 단위로 읽는다. 못 읽으면 빈 목록을 돌려준다."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    if path.stat().st_size > MAX_READ_KB * 1024:
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return []


def in_dir(path: Path, root: Path, names: set[str]) -> bool:
    """파일이 특정 이름의 하위 폴더 안에 있는지 본다."""
    parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
    return any(part in names for part in parts)


# ===========================================================================
# 여섯 가지 점검
# ===========================================================================
def check_runnable_python(root: Path, files: list[Path]) -> tuple[bool, str]:
    """1. 실행되는 파이썬 파일이 있는가. 실행하지 않고 문법만 본다."""
    py_files = [p for p in files if p.suffix == ".py"]
    if not py_files:
        return False, "폴더 안에 .py 파일이 없다"

    broken = []
    for path in py_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as error:
            broken.append(f"{path.relative_to(root).as_posix()}:{error.lineno}")
        except (UnicodeDecodeError, OSError):
            broken.append(f"{path.relative_to(root).as_posix()}(읽기 실패)")

    if broken:
        return False, f".py {len(py_files)}개 중 문법 오류 {len(broken)}개 - {', '.join(broken)}"
    return True, f".py {len(py_files)}개 모두 문법 오류 없음"


def check_output_files(root: Path, files: list[Path]) -> tuple[bool, str]:
    """2. 결과 파일이 있는가. 그림·로그, 또는 results 폴더 안의 파일을 센다."""
    hits = [
        p for p in files
        if p.suffix.lower() in OUTPUT_SUFFIXES or in_dir(p, root, OUTPUT_DIRS)
    ]
    if not hits:
        return False, "그림·로그도 없고 results 폴더도 없다"
    shown = ", ".join(p.relative_to(root).as_posix() for p in hits[:3])
    return True, f"결과 파일 {len(hits)}개 - {shown}"


def check_input_data(root: Path, files: list[Path]) -> tuple[bool, str]:
    """3. 데이터 또는 입력 예시가 있는가."""
    hits = [
        p for p in files
        if not in_dir(p, root, OUTPUT_DIRS)
        and (p.suffix.lower() in DATA_SUFFIXES or in_dir(p, root, DATA_DIRS))
    ]
    if not hits:
        return False, "csv·json·txt 파일도 없고 data 폴더도 없다"
    shown = ", ".join(p.relative_to(root).as_posix() for p in hits[:3])
    return True, f"입력 파일 {len(hits)}개 - {shown}"


def scan_patterns(root: Path, files: list[Path], patterns) -> list[str]:
    """정해 둔 문자열 모양을 찾아 위치를 돌려준다."""
    found = []
    for path in files:
        for number, line in enumerate(read_text(path), start=1):
            for pattern, reason in patterns:
                if pattern.search(line):
                    found.append(f"{path.relative_to(root).as_posix()}:{number} {reason}")
                    break
    return found


def check_no_secret(root: Path, files: list[Path]) -> tuple[bool, str]:
    """4. 코드에 비밀키처럼 보이는 문자열이 없는가."""
    found = scan_patterns(root, files, SECRET_PATTERNS)
    if found:
        return False, f"{len(found)}곳 - " + " / ".join(found[:2])
    return True, "키처럼 보이는 문자열을 찾지 못했다"


def check_no_personal(root: Path, files: list[Path]) -> tuple[bool, str]:
    """5. 데이터에 실명·연락처·학번처럼 보이는 문자열이 없는가."""
    found = scan_patterns(root, files, PERSONAL_PATTERNS)
    if found:
        return False, f"{len(found)}곳 - " + " / ".join(found[:2])
    return True, "개인정보처럼 보이는 문자열을 찾지 못했다"


def check_usage_doc(root: Path, files: list[Path]) -> tuple[bool, str]:
    """6. 사용 방법을 적은 파일이 있는가."""
    hits = [p for p in files if p.stem.lower() in USAGE_STEMS]
    if not hits:
        return False, "README 또는 사용법 파일이 없다"
    return True, ", ".join(p.relative_to(root).as_posix() for p in hits)


CHECKS = [
    ("실행되는 파이썬 파일이 있는가", check_runnable_python),
    ("결과 파일이 있는가", check_output_files),
    ("데이터 또는 입력 예시가 있는가", check_input_data),
    ("코드에 비밀키가 적혀 있지 않은가", check_no_secret),
    ("데이터에 실명·연락처·학번이 없는가", check_no_personal),
    ("사용 방법을 적은 파일이 있는가", check_usage_doc),
]


def inspect(root: Path) -> list[dict]:
    """폴더 하나를 여섯 가지로 점검한다."""
    if not root.is_dir():
        return [
            {"번호": i + 1, "점검 항목": title, "결과": "미통과",
             "근거": f"폴더를 찾지 못했다: {root}"}
            for i, (title, _fn) in enumerate(CHECKS)
        ]

    files = walk_files(root)
    rows = []
    for i, (title, check_fn) in enumerate(CHECKS, start=1):
        passed, reason = check_fn(root, files)
        rows.append({
            "번호": i,
            "점검 항목": title,
            "결과": "통과" if passed else "미통과",
            "근거": reason,
        })
    return rows


# ===========================================================================
# 그림 그리기
# ===========================================================================
def mark(ax, x, y, passed: bool, ko: bool) -> None:
    """통과는 초록 O, 미통과는 빨강 X 로 표시한다."""
    color = GREEN if passed else RED
    ax.add_patch(
        FancyBboxPatch(
            (x, y), 0.66, 0.6,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor="none",
        )
    )
    ax.text(x + 0.33, y + 0.3, "O" if passed else "X",
            ha="center", va="center", fontsize=15, color="white")


def save_figure(ko: bool, left_rows: list[dict], right_rows: list[dict],
                out_path: Path) -> None:
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.55, 7.4)
    ax.axis("off")

    ax.set_title(
        L("제출 전 점검표 — 두 폴더를 같은 여섯 항목으로 확인했다",
          "Pre-submission checklist - two folders, the same six checks", ko),
        fontsize=13.5, pad=12, loc="left",
    )

    ax.text(6.95, 6.55, L("예시 프로젝트", "example project", ko),
            ha="center", fontsize=11.5, color=BLUE)
    ax.text(8.75, 6.55, L("비교용 폴더", "rough folder", ko),
            ha="center", fontsize=11.5, color=ORANGE)

    labels = [
        L("1  실행되는 파이썬 파일이 있는가", "1  A Python file that runs", ko),
        L("2  결과 파일이 있는가", "2  An output file exists", ko),
        L("3  데이터 또는 입력 예시가 있는가", "3  Data or an input example", ko),
        L("4  코드에 비밀키가 적혀 있지 않은가", "4  No secret key in the code", ko),
        L("5  데이터에 실명·연락처·학번이 없는가", "5  No names, phones, or IDs", ko),
        L("6  사용 방법을 적은 파일이 있는가", "6  A file explaining how to run it", ko),
    ]

    for i, text in enumerate(labels):
        y = 5.65 - i * 0.86
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.25, y - 0.1), 9.5, 0.8,
                                       facecolor="#F2F2F2", edgecolor="none"))
        ax.text(0.5, y + 0.3, text, ha="left", va="center",
                fontsize=11.5, color="#222222")
        mark(ax, 6.62, y, left_rows[i]["결과"] == "통과", ko)
        mark(ax, 8.42, y, right_rows[i]["결과"] == "통과", ko)

    left_pass = sum(row["결과"] == "통과" for row in left_rows)
    right_pass = sum(row["결과"] == "통과" for row in right_rows)
    ax.text(5.0, 0.78,
            L(f"예시 프로젝트 {left_pass}/6 통과 · 비교용 폴더 {right_pass}/6 통과  "
              f"— 미통과 이유는 실행 로그에 파일 이름과 줄 번호까지 나온다",
              f"example {left_pass}/6 · rough {right_pass}/6  "
              f"- the log names the file and line for each failure", ko),
            ha="center", fontsize=10.5, color=GRAY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ===========================================================================
# 본체
# ===========================================================================
def print_table(name: str, root: Path, rows: list[dict]) -> None:
    passed = sum(row["결과"] == "통과" for row in rows)
    print(f"--- {name}: {root.name} ---")
    print(f"{'번호':<4}{'결과':<6}{'점검 항목'}")
    for row in rows:
        print(f"{row['번호']:<5}{row['결과']:<6}{row['점검 항목']}")
        print(f"     {'':<6}근거: {row['근거']}")
    print(f"=> {passed}/6 통과")
    print()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    left_rows = inspect(PROJECT_DIR)
    right_rows = inspect(COMPARE_DIR)

    table = pd.concat([
        pd.DataFrame(left_rows).assign(폴더=PROJECT_DIR.name),
        pd.DataFrame(right_rows).assign(폴더=COMPARE_DIR.name),
    ])[["폴더", "번호", "점검 항목", "결과", "근거"]]
    table_path = RESULTS_DIR / "submission_check_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    plot_path = RESULTS_DIR / "submission_check.png"
    save_figure(ko, left_rows, right_rows, plot_path)

    # ------------------------------------------------------------------
    print("=== 실습 14.2 제출 전 자동 점검 ===")
    _krfont.report(ko)
    print("이 도구는 파일을 읽기만 한다. 고치거나 지우지 않는다.")
    print(f"점검 대상 1: {PROJECT_DIR.relative_to(CHAPTER_DIR.parents[1]).as_posix()}")
    print(f"점검 대상 2: {COMPARE_DIR.relative_to(CHAPTER_DIR.parents[1]).as_posix()}")
    print()

    print_table("예시 프로젝트", PROJECT_DIR, left_rows)
    print_table("비교용 폴더", COMPARE_DIR, right_rows)

    print("=== 저장된 결과 ===")
    print(f"점검표 그림: results/{plot_path.name}")
    print(f"점검 결과 표: results/{table_path.name}")
    print()
    print("자기 프로젝트를 점검하려면 PROJECT_DIR 에 그 폴더 경로를 적는다.")
    print("4번과 5번이 미통과면 파일을 올리기 전에 반드시 고친다.")


if __name__ == "__main__":
    main()
