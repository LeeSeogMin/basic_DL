"""실습 2.2 - 오류 메시지를 읽는 연습.

이 코드는 오류를 일부러 네 번 낸다. 초보자가 가장 자주 만나는 네 가지다.
  NameError, TypeError, IndexError, ZeroDivisionError

여기서 try/except를 쓰는 이유를 분명히 해 둔다.

  이 코드의 try/except는 오류를 숨기려고 쓰는 것이 아니다.
  오류를 하나 낼 때마다 프로그램이 멈추면 네 가지를 한 번에 볼 수 없다.
  그래서 오류를 잡아 두고, 파이썬이 준 메시지를 그대로 화면에 꺼내 읽는다.
  잡은 오류를 조용히 버리는 코드(except: pass)는 이 수업에서 쓰지 않는다.
  그런 코드는 프로그램이 왜 틀렸는지 알려주는 유일한 단서를 지운다.

마지막에 오류 종류별 정리 표 그림 results/error_guide_table.png 를 만든다.
"""

from __future__ import annotations

import re
import textwrap
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import _krfont

# ============================================================
# 학생이 바꾸는 값
#   SHOW_FULL_TRACEBACK 을 False 로 바꾸면 원본 오류 화면을 감춘다.
#   한 번은 True 로 실행해 진짜 오류 화면을 눈에 익힌다.
# ============================================================
SHOW_FULL_TRACEBACK = True
# ============================================================

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


# ---------- 오류를 내는 함수 네 개 ----------

def cause_name_error() -> str:
    """변수 이름을 잘못 부른다."""
    mesage = "안녕하세요"          # 만들 때는 mesage
    return message                 # 부를 때는 message -> 그런 이름이 없다


def cause_type_error() -> str:
    """문자열과 숫자를 그대로 더한다."""
    student_count = 3
    return "우리 조 인원은 " + student_count   # 문자열 + 정수


def cause_index_error() -> int:
    """리스트에 없는 자리를 꺼낸다."""
    scores = [90, 85, 77]          # 자리 번호는 0, 1, 2 뿐이다
    return scores[3]               # 네 번째 자리는 없다


def cause_zero_division_error() -> float:
    """0으로 나눈다."""
    total_score = 250
    student_count = 0              # 아직 아무도 등록하지 않았다
    return total_score / student_count


# 오류마다: 함수, 무슨 뜻인가, 어디를 고치는가
ERROR_CASES = [
    {
        "trigger": cause_name_error,
        "meaning_ko": "그런 이름의 변수가 없다",
        "meaning_en": "no variable with that name",
        "fix_ko": "만들 때 쓴 이름과 부를 때 쓴 이름이 같은지 본다",
        "fix_en": "check the name where it was created",
    },
    {
        "trigger": cause_type_error,
        "meaning_ko": "종류가 다른 값을 섞어 썼다",
        "meaning_en": "mixed two different kinds of value",
        "fix_ko": "숫자를 str()로 바꾸거나 f-문자열을 쓴다",
        "fix_en": "wrap the number in str() or use an f-string",
    },
    {
        "trigger": cause_index_error,
        "meaning_ko": "리스트에 없는 자리를 꺼냈다",
        "meaning_en": "asked for a slot the list does not have",
        "fix_ko": "len()으로 길이를 세고 자리 번호가 0부터인 것을 확인한다",
        "fix_en": "check len() and remember slots start at 0",
    },
    {
        "trigger": cause_zero_division_error,
        "meaning_ko": "0으로 나눴다",
        "meaning_en": "divided by zero",
        "fix_ko": "나누기 전에 나누는 값이 0인지 먼저 확인한다",
        "fix_en": "check the divisor before dividing",
    },
]


def shorten_paths(text: str) -> str:
    """오류 화면에 찍히는 긴 폴더 경로를 파일 이름만 남긴다.

    강의자료에 싣기 위해 줄이는 것이고, 줄 번호와 메시지는 그대로 둔다.
    내 화면에서는 전체 경로가 그대로 나온다.
    """
    return re.sub(
        r'File "([^"]+)"',
        lambda m: f'File "{Path(m.group(1)).name}"',
        text,
    )


def where_it_happened(error: BaseException) -> tuple[str, int, str]:
    """오류가 실제로 난 파일, 줄 번호, 함수 이름을 꺼낸다."""
    frames = traceback.extract_tb(error.__traceback__)
    last = frames[-1]
    return Path(last.filename).name, last.lineno, last.name


def run_one_case(index: int, case: dict) -> dict:
    """오류를 한 번 내고, 파이썬이 준 정보를 그대로 읽어 출력한다."""
    print(f"--- {index}. ", end="")
    try:
        case["trigger"]()
    except Exception as error:      # 숨기려는 것이 아니라 읽으려고 잡는다
        name = type(error).__name__
        message = str(error)
        filename, lineno, funcname = where_it_happened(error)

        print(f"{name} ---")
        print(f"파이썬이 준 메시지: {message}")
        print(f"어디서 났는가: {filename} {lineno}번째 줄, {funcname}() 안")
        print(f"무슨 뜻인가: {case['meaning_ko']}")
        print(f"어디를 고치는가: {case['fix_ko']}")

        if SHOW_FULL_TRACEBACK and index == 1:
            print()
            print("[참고] 오류를 잡지 않았다면 화면에 이렇게 나온다.")
            print("      (폴더 경로는 파일 이름만 남기고 줄였다)")
            screen = shorten_paths(traceback.format_exc().rstrip())
            print(textwrap.indent(screen, "    "))

        print()
        return {"name": name, "message": message, "case": case}

    raise RuntimeError(f"{index}번 사례에서 오류가 나지 않았다. 코드를 확인한다.")


def draw_table(ko: bool, results: list[dict], out_path: Path) -> None:
    """오류 종류별 정리 표를 그림 한 장으로 만든다."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    ax.text(0.2, 7.25,
            L("초보자가 자주 만나는 오류 네 가지",
              "Four errors beginners meet most often", ko),
            fontsize=15, color="#1A1A1A", va="center")
    ax.text(0.2, 6.78,
            L("메시지는 이 코드를 실제로 실행해 파이썬에서 받은 것이다.",
              "The messages below came from actually running this file.", ko),
            fontsize=10.5, color=GRAY, va="center")

    columns = [
        (0.2, 2.1, L("오류 이름", "Error name", ko)),
        (2.4, 4.9, L("파이썬이 보여준 메시지", "Message Python printed", ko)),
        (7.4, 3.0, L("무슨 뜻인가", "What it means", ko)),
        (10.5, 3.3, L("어디를 고치는가", "Where to fix it", ko)),
    ]

    header_y = 6.00
    for x, w, title in columns:
        ax.add_patch(
            FancyBboxPatch(
                (x, header_y), w, 0.55,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=BLUE, edgecolor="none",
            )
        )
        ax.text(x + w / 2, header_y + 0.28, title, ha="center", va="center",
                fontsize=11, color="white")

    colors = [ORANGE, GREEN, RED, GRAY]
    msg_width = 46
    text_width = 16 if ko else 26

    for i, item in enumerate(results):
        y = 5.10 - i * 1.22
        for x, w, _title in columns:
            ax.add_patch(
                FancyBboxPatch(
                    (x, y - 0.5), w, 1.05,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor="#F7F9FB", edgecolor="#DCE3E9", linewidth=1.0,
                )
            )
        ax.add_patch(
            FancyBboxPatch(
                (0.2, y - 0.5), 2.1, 1.05,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=colors[i], edgecolor="none",
            )
        )
        ax.text(1.25, y, item["name"], ha="center", va="center",
                fontsize=11, color="white")
        ax.text(2.6, y, textwrap.fill(item["message"], msg_width),
                ha="left", va="center", fontsize=9, family="monospace",
                color="#222222", linespacing=1.5)
        ax.text(7.6, y,
                textwrap.fill(L(item["case"]["meaning_ko"],
                                item["case"]["meaning_en"], ko), text_width),
                ha="left", va="center", fontsize=10, color="#222222",
                linespacing=1.5)
        ax.text(10.7, y,
                textwrap.fill(L(item["case"]["fix_ko"],
                                item["case"]["fix_en"], ko), text_width),
                ha="left", va="center", fontsize=10, color="#222222",
                linespacing=1.5)

    ax.text(7.0, 0.35,
            L("오류 메시지는 프로그램이 멈춘 이유를 알려주는 문장이다. 감추지 말고 읽는다.",
              "An error message states why the program stopped. Read it, do not hide it.", ko),
            ha="center", va="center", fontsize=10.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 오류 메시지 읽기 연습 ===")
    print("오류를 네 번 일부러 낸다. 오류를 감추려고 잡는 것이 아니라,")
    print("파이썬이 준 메시지를 그대로 꺼내 읽으려고 잡는다.")
    print()

    results = [run_one_case(i + 1, case) for i, case in enumerate(ERROR_CASES)]

    print("=== 읽는 순서 ===")
    print("1. 맨 아랫줄을 읽는다. 오류 이름과 무엇이 잘못됐는지가 거기 있다.")
    print("2. 그 위에서 파일 이름과 줄 번호를 찾는다. 그 줄을 연다.")
    print("3. 맨 윗줄(Traceback ...)은 호출 기록이다. 마지막에 본다.")
    print()

    ko = _krfont.setup()
    _krfont.report(ko)
    out_path = RESULTS_DIR / "error_guide_table.png"
    draw_table(ko, results, out_path)
    print(f"저장: results/{out_path.name}  (LMS 제출물 2)")
    print()
    print(f"오류 {len(results)}건을 모두 잡아 읽었다. 프로그램은 끝까지 실행됐다.")


if __name__ == "__main__":
    main()
