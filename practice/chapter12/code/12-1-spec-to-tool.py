"""실습 12.1 - 명세를 먼저 쓰고, 그 명세대로 도구를 만든다.

=========================  이 도구의 명세  =========================

[무엇을 하는가]
  폴더 하나를 통째로 읽어, 확장자별로 파일이 몇 개이고 용량이 얼마인지 센다.

[입력]
  practice/chapter12/sample_folder/ 폴더. 하위 폴더까지 모두 본다.
  이 폴더는 실습용으로 만든 합성 예제다. 실제 수업 자료나 개인 자료가 아니다.

[출력]
  1. 확장자별 파일 수·총 용량 표를 화면에 출력한다.
  2. 같은 표를 results/folder_extension_table.csv 로 저장한다.
  3. 확장자별 파일 수와 총 용량을 results/folder_report.png 막대그래프로 그린다.

[하면 안 되는 일]
  - 입력 폴더의 파일을 지우거나 고치거나 새로 만들지 않는다. 읽기만 한다.
  - 인터넷에 접속하지 않는다. 어떤 데이터도 밖으로 보내지 않는다.
  - 파일 안의 내용을 화면에 출력하거나 저장하지 않는다.
    이름·확장자·크기만 결과에 넣는다.

[실패 시 동작]
  - 입력 폴더가 없으면 FileNotFoundError 를 내고 멈춘다.
    빈 표를 만들거나 다른 폴더로 조용히 바꾸지 않는다.
  - 폴더에 파일이 하나도 없으면 "파일 0개"라고 알리고 그림 없이 끝낸다.

===================================================================

명세를 먼저 쓰는 이유는 AI에게 일을 시킬 때와 같다.
무엇을 만들지 문장으로 못 쓰면, 만들어진 것이 맞는지도 확인할 수 없다.
이 코드는 마지막에 명세 항목을 하나씩 실제 동작과 대조해 점검표로 출력한다.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

import _krfont

# ===================== 학생이 바꿔 볼 값 =====================
TARGET_FOLDER_NAME = "sample_folder"   # 점검할 폴더 이름
CHART_TOP_N = 8                        # 그래프에 그릴 확장자 개수(많은 순)
SIZE_UNIT = "B"                        # 용량 단위: "B" 또는 "KB"
# ============================================================

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"
TARGET_FOLDER = CHAPTER_DIR / TARGET_FOLDER_NAME

NO_EXT = "(확장자 없음)"

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


# ---------------------------------------------------------------- 도구 본체
def collect_files(folder: Path) -> list[tuple[str, int]]:
    """폴더 안의 모든 파일에서 (확장자, 크기)만 뽑는다.

    파일을 열지 않는다. 이름과 stat() 으로 얻은 크기만 쓴다.
    폴더가 없으면 명세대로 FileNotFoundError 를 낸다.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"점검할 폴더를 찾을 수 없습니다: {folder}")

    found = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower() if path.suffix else NO_EXT
        found.append((ext, path.stat().st_size))
    return found


def build_table(found: list[tuple[str, int]]) -> pd.DataFrame:
    """확장자별로 파일 수와 총 용량을 센다. 많은 순으로 정렬한다."""
    frame = pd.DataFrame(found, columns=["확장자", "크기"])
    table = (
        frame.groupby("확장자", as_index=False)
        .agg(파일수=("크기", "size"), 총용량=("크기", "sum"))
        .sort_values(["파일수", "총용량"], ascending=False)
        .reset_index(drop=True)
    )
    table["평균용량"] = (table["총용량"] / table["파일수"]).round(1)
    return table


def to_unit(num_bytes: float) -> float:
    """바이트 값을 SIZE_UNIT 에 맞춰 바꾼다."""
    return round(num_bytes / 1024, 2) if SIZE_UNIT == "KB" else float(num_bytes)


def snapshot(folder: Path) -> dict[str, tuple[int, str]]:
    """폴더 안 모든 파일의 크기와 내용 해시를 기록한다.

    이 함수만 파일 내용을 읽는다. 읽은 내용은 해시로만 바꾸고 어디에도 출력하지 않는다.
    도구가 정말로 폴더를 건드리지 않았는지 확인하려는 용도다.
    """
    state = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            state[str(path.relative_to(folder))] = (path.stat().st_size, digest)
    return state


def draw_report(table: pd.DataFrame, ko: bool, out_path: Path) -> None:
    """확장자별 파일 수와 총 용량을 가로 막대 두 장으로 그린다."""
    L = _krfont.label
    top = table.head(CHART_TOP_N).iloc[::-1]      # 가로 막대는 아래부터 쌓인다
    labels = list(top["확장자"])
    counts = list(top["파일수"])
    sizes = [to_unit(v) for v in top["총용량"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 0.62 * len(labels) + 3.0))

    ax = axes[0]
    ax.barh(labels, counts, color=BLUE)
    ax.set_xlabel(L("파일 수 (개)", "number of files", ko), fontsize=11)
    ax.set_title(L("확장자별 파일 수", "files per extension", ko), fontsize=13, pad=10)
    ax.set_xlim(0, max(counts) * 1.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # 파일 수는 정수다
    for y, value in enumerate(counts):
        ax.text(value + max(counts) * 0.03, y, str(value),
                va="center", fontsize=11, color="#333333")

    ax = axes[1]
    ax.barh(labels, sizes, color=ORANGE)
    ax.set_xlabel(L(f"총 용량 ({SIZE_UNIT})", f"total size ({SIZE_UNIT})", ko), fontsize=11)
    ax.set_title(L("확장자별 총 용량", "total size per extension", ko), fontsize=13, pad=10)
    ax.set_xlim(0, max(sizes) * 1.25)
    for y, value in enumerate(sizes):
        ax.text(value + max(sizes) * 0.03, y, f"{value:g}",
                va="center", fontsize=11, color="#333333")

    for ax in axes:
        ax.tick_params(labelsize=11, length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(
        L(f"{TARGET_FOLDER_NAME} 폴더 점검 결과 (합성 예제 폴더)",
          f"{TARGET_FOLDER_NAME} folder report (synthetic sample folder)", ko),
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- 명세 자기 점검
def self_check(
    table: pd.DataFrame,
    found: list[tuple[str, int]],
    before: dict[str, tuple[int, str]],
    after: dict[str, tuple[int, str]],
    made: list[Path],
) -> list[dict]:
    """명세에 적은 항목을 하나씩 실제 동작과 대조한다."""
    checks = []

    def add(item: str, expected: str, actual: str, ok: bool) -> None:
        checks.append({"항목": item, "기대": expected, "실제": actual, "판정": "통과" if ok else "실패"})

    # [입력] 하위 폴더까지 본다
    # 하위 폴더가 있으면 그 안의 파일도 셌는지 본다. 하위 폴더가 없는 폴더면 볼 것이 없다.
    subdirs = sorted(p.name for p in TARGET_FOLDER.iterdir() if p.is_dir())
    nested = sum(1 for p in TARGET_FOLDER.rglob("*")
                 if p.is_file() and p.parent != TARGET_FOLDER)
    if subdirs:
        add("입력: 하위 폴더까지 읽는다",
            f"하위 폴더 {len(subdirs)}개({', '.join(subdirs)})의 파일 포함",
            f"하위 폴더 안 파일 {nested}개를 셈",
            nested > 0)
    else:
        add("입력: 하위 폴더까지 읽는다", "하위 폴더 없음", "볼 하위 폴더가 없음", True)

    # [출력] 개수가 맞는가
    add("출력: 확장자별 파일 수의 합",
        f"전체 {len(found)}개",
        f"{int(table['파일수'].sum())}개",
        int(table["파일수"].sum()) == len(found))

    # [출력] 용량이 맞는가
    total_bytes = sum(size for _ext, size in found)
    add("출력: 확장자별 용량의 합",
        f"전체 {total_bytes} B",
        f"{int(table['총용량'].sum())} B",
        int(table["총용량"].sum()) == total_bytes)

    # [출력] 결과 파일이 실제로 생겼는가
    exists = [p for p in made if p.exists()]
    add("출력: 결과 파일 생성",
        f"{len(made)}개 파일",
        f"{len(exists)}개 생성 ({', '.join(p.name for p in exists)})",
        len(exists) == len(made))

    # [하면 안 되는 일] 입력 폴더를 건드리지 않았는가
    changed = [name for name in set(before) | set(after)
               if before.get(name) != after.get(name)]
    add("금지: 입력 폴더를 고치지 않는다",
        "바뀐 파일 0개",
        f"바뀐 파일 {len(changed)}개",
        not changed)

    # [실패 시 동작] 없는 폴더를 주면 멈추는가
    # 아래 try 는 오류를 감추려는 것이 아니라, 실패 동작이 명세대로인지 확인하려는 시험이다.
    missing = CHAPTER_DIR / "__없는_폴더__"
    try:
        collect_files(missing)
        reason, ok = "오류 없이 넘어감", False
    except FileNotFoundError:
        reason, ok = "FileNotFoundError 발생", True
    add("실패: 없는 폴더를 주면 멈춘다", "FileNotFoundError", reason, ok)

    return checks


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    print("=== 실습 12.1 명세를 먼저 쓰고 그대로 만든 폴더 점검 도구 ===")
    _krfont.report(ko)
    print(f"점검 대상: practice/chapter12/{TARGET_FOLDER_NAME}/ (합성 예제 폴더)")
    print("이 도구는 폴더를 읽기만 한다. 파일을 지우거나 고치지 않는다.")
    print()

    before = snapshot(TARGET_FOLDER)
    found = collect_files(TARGET_FOLDER)

    if not found:
        print("파일 0개. 셀 것이 없어 그림 없이 끝낸다.")
        return

    table = build_table(found)
    total_bytes = sum(size for _ext, size in found)

    print("=== 확장자별 집계 ===")
    print_table(
        ["확장자", "파일 수", f"총 용량({SIZE_UNIT})", f"평균 용량({SIZE_UNIT})"],
        [[row["확장자"], str(int(row["파일수"])),
          f"{to_unit(row['총용량']):g}", f"{to_unit(row['평균용량']):g}"]
         for _i, row in table.iterrows()],
        right={1, 2, 3},
    )
    print(f"합계: 파일 {len(found)}개, 용량 {to_unit(total_bytes):g} {SIZE_UNIT}, "
          f"확장자 {len(table)}종")
    print()

    csv_path = RESULTS_DIR / "folder_extension_table.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    png_path = RESULTS_DIR / "folder_report.png"
    draw_report(table, ko, png_path)

    after = snapshot(TARGET_FOLDER)
    checks = self_check(table, found, before, after, [csv_path, png_path])

    print("=== 명세 자기 점검 ===")
    print_table(
        ["번호", "명세 항목", "기대", "실제", "판정"],
        [[str(i + 1), c["항목"], c["기대"], c["실제"], c["판정"]]
         for i, c in enumerate(checks)],
    )
    print()

    failed = [c for c in checks if c["판정"] == "실패"]
    print("=== 저장된 결과 ===")
    print(f"집계 표: results/{csv_path.name}")
    print(f"막대그래프: results/{png_path.name}")
    print()

    if failed:
        print(f"명세와 어긋난 항목 {len(failed)}건. 코드를 고쳐야 한다.")
        raise SystemExit(1)
    print(f"명세 {len(checks)}개 항목 모두 통과. 만든 것과 적은 것이 같다.")


if __name__ == "__main__":
    main()
