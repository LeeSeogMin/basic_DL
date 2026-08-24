"""실습 2.1 - 파이썬이 내 컴퓨터에서 실제로 도는지 확인한다.

하는 일은 두 가지다.
  1. 아래 상수에 적은 내 정보로 자기소개 문장을 만들어 화면에 출력한다.
  2. 이 코드를 실행한 환경(파이썬 버전, 운영체제)을 함께 출력한다.

마지막에 실행 요약 카드 그림 results/hello_ai_card.png 를 만든다.
이 그림 한 장이 "내 컴퓨터에서 파이썬이 돌았다"는 증거다.

input()을 쓰지 않는다. input()을 쓰면 자동 실행 도구가 입력을 기다리다 멈춘다.
대신 파일 맨 위 상수를 학생이 직접 고친다.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import _krfont

# ============================================================
# 학생이 바꾸는 값 - 여기 세 줄만 자기 것으로 고친다
# ============================================================
MY_NAME = "민지"
MY_INTEREST = "이미지 분류"
MY_GOAL = "파이썬 파일 하나를 끝까지 실행하기"

# 한글 폰트가 없는 컴퓨터에서 그림에 대신 쓸 영어 표기.
# 한글이 잘 보이면 고치지 않아도 된다.
MY_NAME_EN = "Minji"
MY_INTEREST_EN = "image classification"
MY_GOAL_EN = "run one Python file end to end"
# ============================================================

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
GRAY = "#79706E"


def make_intro(name: str, interest: str, goal: str) -> str:
    """자기소개 문장 한 줄을 만든다."""
    clean_name = name.strip() or "이름 없는 학생"
    clean_interest = interest.strip() or "인공지능"
    clean_goal = goal.strip() or "오늘 만든 코드를 직접 실행하기"
    return (
        f"{clean_name} 학생은 {clean_interest}에 관심이 있고, "
        f"오늘의 목표는 '{clean_goal}'이다."
    )


def collect_environment() -> list[tuple[str, str, str]]:
    """실행 환경을 (한글 항목, 영어 항목, 값) 목록으로 모은다."""
    return [
        ("파이썬 버전", "Python version", sys.version.split()[0]),
        ("운영체제", "Operating system",
         f"{platform.system()} {platform.release()}"),
        ("처리장치", "Machine", platform.machine()),
        ("실행한 파일", "Script", Path(__file__).name),
    ]


def draw_card(ko: bool, intro: str, environment, out_path: Path) -> None:
    """실행 요약 카드 한 장을 그린다."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (0.2, 0.3), 9.6, 6.4,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#F7F9FB", edgecolor="#D3DCE4", linewidth=1.4,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.2, 5.7), 9.6, 1.0,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=BLUE, edgecolor="none",
        )
    )
    ax.text(5.0, 6.2,
            L("실습 2.1 실행 요약 카드", "Practice 2.1 - run summary card", ko),
            ha="center", va="center", fontsize=15, color="white")

    ax.text(5.0, 5.05,
            L(intro, "This card was produced by running the file below.", ko),
            ha="center", va="center", fontsize=12, color="#1A1A1A")

    rows = [
        (L("이름", "Name", ko), L(MY_NAME, MY_NAME_EN, ko)),
        (L("관심 분야", "Interest", ko), L(MY_INTEREST, MY_INTEREST_EN, ko)),
        (L("오늘의 목표", "Goal today", ko), L(MY_GOAL, MY_GOAL_EN, ko)),
    ]
    rows += [(L(k, e, ko), v) for k, e, v in environment]

    top = 4.40
    step = 0.55
    for i, (key, value) in enumerate(rows):
        y = top - i * step
        color = ORANGE if i < 3 else GRAY
        ax.add_patch(
            FancyBboxPatch(
                (0.7, y - 0.2), 2.6, 0.4,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=color, edgecolor="none",
            )
        )
        ax.text(2.0, y, key, ha="center", va="center",
                fontsize=10.5, color="white")
        ax.text(3.6, y, value, ha="left", va="center",
                fontsize=11, color="#1A1A1A")

    ax.text(5.0, 0.62,
            L("이 카드가 만들어졌다면 내 컴퓨터에서 파이썬이 끝까지 돌았다는 뜻이다.",
              "If this card exists, Python ran to the end on this machine.", ko),
            ha="center", va="center", fontsize=10.5, color=GREEN)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    intro = make_intro(MY_NAME, MY_INTEREST, MY_GOAL)
    environment = collect_environment()

    print("=== 나의 첫 파이썬 실행 ===")
    print(intro)
    print()

    print("=== 실행 환경 ===")
    for korean, _english, value in environment:
        print(f"{korean}: {value}")
    print()

    ko = _krfont.setup()
    _krfont.report(ko)
    out_path = RESULTS_DIR / "hello_ai_card.png"
    draw_card(ko, intro, environment, out_path)
    print(f"저장: results/{out_path.name}  (LMS 제출물 1)")
    print()

    print("=== 확인할 것 ===")
    print("- 위 자기소개 문장에 내 이름이 나왔는가")
    print("- 파이썬 버전과 운영체제가 내 컴퓨터와 맞는가")
    print("- 코드에 비밀번호, API 키, 학번을 넣지 않았는가")


if __name__ == "__main__":
    main()
