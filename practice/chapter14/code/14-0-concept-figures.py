"""14장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig14-1  프로젝트의 뼈대 (문제 -> 데이터 -> 모델 -> 평가 -> 한계)
  fig14-2  좋은 프로젝트와 피해야 할 결과물의 갈림길
  fig14-3  제출 전 점검 항목 여섯 가지

이 그림들은 데이터 분석 결과가 아니라 개념 설명용 도식이다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _krfont

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def box(ax, x, y, w, h, text, facecolor, fontsize=11, textcolor="white", ha="center"):
    """모서리가 둥근 상자 하나를 그리고 글자를 넣는다."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=facecolor,
            edgecolor="none",
        )
    )
    text_x = x + w / 2 if ha == "center" else x + 0.18
    ax.text(
        text_x,
        y + h / 2,
        text,
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=textcolor,
        linespacing=1.55,
    )


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.6,
            color=color,
        )
    )


def blank_axes(ax, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def figure_skeleton(ko: bool, out_path: Path) -> None:
    """그림 14-1: 프로젝트를 다섯 칸으로 나눈 뼈대."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(11.0, 4.7))
    blank_axes(ax, xlim=(0, 10), ylim=(1.0, 5.3))

    ax.set_title(
        L("기말 프로젝트의 뼈대 — 다섯 칸을 채우면 프로젝트가 된다",
          "The skeleton of a final project - fill in five boxes", ko),
        fontsize=13.5, pad=12, loc="left",
    )

    stages = [
        (L("문제", "Problem", ko), BLUE,
         L("무엇을 맞히는가\n한 문장으로 쓴다", "What do you predict?\nWrite one sentence", ko)),
        (L("데이터", "Data", ko), ORANGE,
         L("어디서 구하고\n몇 개인가", "Where from,\nhow many rows?", ko)),
        (L("모델", "Model", ko), GREEN,
         L("기준선보다\n나은가", "Better than\nthe baseline?", ko)),
        (L("평가", "Evaluation", ko), BLUE,
         L("무엇으로 잴지\n미리 정했는가", "Did you fix the\nmetric in advance?", ko)),
        (L("한계", "Limits", ko), GRAY,
         L("언제 틀리는지\n말할 수 있는가", "Can you say when\nit fails?", ko)),
    ]

    width, gap = 1.65, 0.30
    start = (10.0 - (len(stages) * width + (len(stages) - 1) * gap)) / 2
    for i, (name, color, question) in enumerate(stages):
        x = start + i * (width + gap)
        box(ax, x, 3.45, width, 1.15, name, color, 13)
        ax.text(x + width / 2, 3.15, question, ha="center", va="top",
                fontsize=10, color="#333333", linespacing=1.5)
        if i < len(stages) - 1:
            arrow(ax, x + width + 0.03, 4.02, x + width + gap - 0.03, 4.02)

    ax.text(5.0, 2.15,
            L("발표도 이 순서로 한다. 다섯 칸 중 비어 있는 칸이 곧 남은 할 일이다.",
              "Present in this same order. An empty box is the work that remains.", ko),
            ha="center", fontsize=11, color="#333333")
    ax.text(5.0, 1.50,
            L("크기를 키우는 방향이 아니라, 다섯 칸을 모두 채우는 방향으로 만든다.",
              "Aim to fill all five boxes, not to build a bigger model.", ko),
            ha="center", fontsize=11, color=RED)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_good_vs_avoid(ko: bool, out_path: Path) -> None:
    """그림 14-2: 같은 시간을 써도 갈라지는 두 방향."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 7.6))

    ax.set_title(
        L("같은 시간을 써도 결과물은 두 방향으로 갈린다",
          "The same hours can end in two very different deliverables", ko),
        fontsize=13.5, pad=12, loc="left",
    )

    box(ax, 0.25, 6.45, 4.6, 0.75,
        L("인정되는 프로젝트", "Accepted", ko), GREEN, 12.5)
    box(ax, 5.15, 6.45, 4.6, 0.75,
        L("피해야 할 결과물", "To avoid", ko), RED, 12.5)

    good = [
        L("실행되는 코드가 있다", "Code that actually runs", ko),
        L("데이터 또는 입력 예시가 있다", "Data or an input example", ko),
        L("AI에게 무엇을 시켰는지 적혀 있다", "A record of what AI was asked", ko),
        L("틀린 사례를 직접 보여준다", "Shows its own failure cases", ko),
        L("배운 개념 두 개를 연결해 설명한다", "Connects two course concepts", ko),
        L("위험 두 가지를 점검했다", "Checks two risks", ko),
    ]
    avoid = [
        L("프롬프트 결과만 붙여넣은 보고서", "A report of pasted prompt output", ko),
        L("실행 코드 없는 아이디어 발표", "An idea talk with no code", ko),
        L("출처 없는 AI 설명 모음", "AI explanations with no source", ko),
        L("해석 없는 성능 수치", "Numbers with no interpretation", ko),
        L("본인이 이해하지 못한 대형 코드", "Large code the author cannot read", ko),
        L("개인정보·API 키를 넣은 결과물", "Work built on personal data or keys", ko),
    ]

    for i, (left, right) in enumerate(zip(good, avoid)):
        y = 5.35 - i * 0.85
        box(ax, 0.25, y, 4.6, 0.62, left, "#E8F1E4", 10.5, "#2C4A22", ha="left")
        box(ax, 5.15, y, 4.6, 0.62, right, "#FBE6E5", 10.5, "#7A2321", ha="left")

    ax.text(5.0, 0.05,
            L("왼쪽은 강의계획서의 기말 프로젝트 필수 요건, 오른쪽은 같은 문서의 피해야 할 결과물 목록이다.",
              "Left: the syllabus requirements. Right: the syllabus list of deliverables to avoid.", ko),
            ha="center", fontsize=10.5, color=GRAY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_submission_check(ko: bool, out_path: Path) -> None:
    """그림 14-3: 제출 전에 스스로 확인하는 여섯 가지."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    blank_axes(ax, xlim=(0, 10), ylim=(0.35, 7.15))

    ax.set_title(
        L("제출 전 점검 여섯 가지 — 실습 14.2가 자동으로 확인한다",
          "Six checks before you submit - practice 14.2 runs them for you", ko),
        fontsize=13.5, pad=12, loc="left",
    )

    items = [
        (L("실행되는 파이썬 파일이 있는가", "Is there a Python file that runs?", ko),
         L("폴더에 .py 가 있고 문법 오류가 없다", "A .py exists and parses without error", ko), BLUE),
        (L("결과 파일이 있는가", "Is there an output file?", ko),
         L("그림·로그, 또는 results 폴더의 파일", "A figure or log, or a file under results/", ko), BLUE),
        (L("데이터 또는 입력 예시가 있는가", "Is there data or an input example?", ko),
         L("csv·json·txt, 또는 data 폴더의 파일", "csv/json/txt, or a file under data/", ko), BLUE),
        (L("코드에 비밀키가 적혀 있지 않은가", "No secret key written in the code?", ko),
         L("api_key = \"...\" 같은 줄을 찾는다", "Looks for lines like api_key = \"...\"", ko), RED),
        (L("데이터에 실명·연락처·학번이 없는가", "No real names, phones, IDs in the data?", ko),
         L("이름·학번 열 이름과 전화번호 형태를 찾는다", "Looks for such column names and phone patterns", ko), RED),
        (L("사용 방법을 적은 파일이 있는가", "Is there a file explaining how to run it?", ko),
         L("README 또는 사용법 파일", "A README or usage file", ko), ORANGE),
    ]

    for i, (title, detail, color) in enumerate(items):
        y = 5.85 - i * 0.95
        box(ax, 0.3, y, 0.72, 0.72, str(i + 1), color, 13)
        ax.text(1.25, y + 0.46, title, ha="left", va="center",
                fontsize=11.5, color="#222222")
        ax.text(1.25, y + 0.15, detail, ha="left", va="center",
                fontsize=9.8, color=GRAY)

    ax.text(5.0, 0.62,
            L("네 번째와 다섯 번째는 되돌릴 수 없는 항목이다. 한 번 올라가면 지워도 남는다.",
              "Items 4 and 5 cannot be undone: once uploaded, deleting does not erase it.", ko),
            ha="center", fontsize=10.5, color=RED)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 14장 개념 그림 생성 ===")

    targets = [
        ("fig14-1-project-skeleton.png", figure_skeleton,
         "프로젝트의 뼈대 다섯 칸"),
        ("fig14-2-good-vs-avoid.png", figure_good_vs_avoid,
         "인정되는 프로젝트와 피해야 할 결과물"),
        ("fig14-3-submission-check.png", figure_submission_check,
         "제출 전 점검 여섯 가지"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")


if __name__ == "__main__":
    main()
