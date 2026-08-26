"""2장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig2-1  코드 한 줄이 실행되기까지의 흐름(편집기 -> 파일 -> 파이썬 -> 결과)
  fig2-2  오류 메시지를 읽는 순서(맨 아랫줄부터 읽는다)
  fig2-3  AI 코딩 도우미가 대신하는 일과 내가 책임지는 일의 경계

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


def box(ax, x, y, w, h, text, facecolor, fontsize=11, textcolor="white"):
    """모서리가 둥근 상자 하나를 그리고 가운데에 글자를 넣는다."""
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
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        linespacing=1.5,
    )


def plain_box(ax, x, y, w, h, facecolor, edgecolor="none"):
    """글자 없는 배경 상자. 글자는 호출한 쪽에서 직접 얹는다."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.2,
        )
    )


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color=color,
        )
    )


def blank_axes(ax, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def figure_code_to_result(ko: bool, out_path: Path) -> None:
    """그림 2-1: 내가 친 글자가 화면의 결과로 바뀌기까지."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 5.2))

    ax.set_title(
        L("코드 한 줄이 결과로 바뀌기까지",
          "From one line of code to a result on screen", ko),
        fontsize=13.5, pad=12, loc="left",
    )

    steps = [
        (0.10, BLUE,
         L("1. 편집기\nVS Code에서\n글자를 친다",
           "1. Editor\ntype the code\nin VS Code", ko)),
        (2.60, ORANGE,
         L("2. 파일\n저장하면\nhello.py가 된다",
           "2. File\nsaving makes\nhello.py", ko)),
        (5.10, GREEN,
         L("3. 파이썬\npython hello.py\n한 줄씩 실행",
           "3. Python\npython hello.py\nruns line by line", ko)),
        (7.60, GRAY,
         L("4. 터미널\nprint가 남긴\n글자가 보인다",
           "4. Terminal\nprint output\nappears", ko)),
    ]
    for x, color, text in steps:
        box(ax, x, 1.9, 2.2, 2.2, text, color, 11)

    for x in (2.35, 4.85, 7.35):
        arrow(ax, x, 3.0, x + 0.2, 3.0)

    ax.text(5.0, 1.05,
            L("저장하지 않으면 파이썬은 이전에 저장된 내용을 실행한다.\n"
              "화면의 코드와 결과가 어긋나면 저장부터 확인한다.",
              "If you do not save, Python runs the previously saved version.\n"
              "When the code on screen and the result disagree, check saving first.", ko),
            ha="center", fontsize=10.5, color=RED, linespacing=1.6)

    ax.text(5.0, 0.25,
            L("컴퓨터는 편집기 화면을 보지 않는다. 저장된 파일만 읽는다.",
              "The computer never looks at the editor; it reads the saved file.", ko),
            ha="center", fontsize=10.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_error_reading(ko: bool, out_path: Path) -> None:
    """그림 2-2: 오류 메시지를 어느 줄부터 읽는가."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 6.2))

    ax.set_title(
        L("오류 메시지는 맨 아랫줄부터 읽는다",
          "Read an error message from the bottom line up", ko),
        fontsize=13.5, pad=12, loc="left",
    )

    # 오류 메시지 본문(실제 파이썬 출력 형태 그대로)
    plain_box(ax, 0.2, 1.5, 6.0, 3.6, "#F4F4F2", "#CCCCCC")

    traceback_lines = [
        (4.60, "Traceback (most recent call last):"),
        (3.95, '  File "hello_ai.py", line 12, in <module>'),
        (3.30, "    print(mesage)"),
        (2.65, "NameError: name 'mesage' is not defined"),
    ]
    for y, text in traceback_lines:
        ax.text(0.45, y, text, fontsize=10.5, family="monospace",
                color="#222222", va="center")

    # 읽는 순서 표시
    ax.text(0.45, 1.95,
            L("이 네 줄이 파이썬이 알려주는 전부다.",
              "These four lines are everything Python tells you.", ko),
            fontsize=10, color=GRAY, va="center")

    order = [
        (2.65, RED, "1",
         L("먼저 여기를 읽는다\n오류 이름 + 무엇이 잘못됐는지",
           "read this first\nerror name + what went wrong", ko)),
        (3.95, ORANGE, "2",
         L("그다음 여기\n어느 파일 몇 번째 줄인지",
           "then here\nwhich file, which line", ko)),
        (4.60, GRAY, "3",
         L("맨 윗줄은 마지막에\n어디서부터 호출됐는지 기록",
           "top line last\nthe call trail", ko)),
    ]
    for y, color, num, text in order:
        ax.add_patch(plt.Circle((6.6, y), 0.24, color=color))
        ax.text(6.6, y, num, ha="center", va="center",
                fontsize=11, color="white")
        ax.text(7.0, y, text, fontsize=10.5, color="#222222",
                va="center", linespacing=1.5)
        arrow(ax, 6.35, y, 6.25, y, color=color)

    ax.text(5.0, 0.65,
            L("오류가 났다는 사실보다, 오류가 알려주는 파일 이름과 줄 번호가 중요하다.\n"
              "그 줄을 열어 보는 것이 첫 번째 행동이다.",
              "The file name and line number matter more than the fact that it failed.\n"
              "Opening that line is the first action.", ko),
            ha="center", fontsize=10.5, color="#333333", linespacing=1.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_ai_boundary(ko: bool, out_path: Path) -> None:
    """그림 2-3: AI 코딩 도우미가 대신하는 일과 내가 책임지는 일."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 6.6))

    ax.set_title(
        L("AI가 대신하는 일과 내가 책임지는 일",
          "What the AI does for you and what stays yours", ko),
        fontsize=13.5, pad=12, loc="left",
    )

    plain_box(ax, 0.2, 1.2, 4.5, 4.4, "#EAF0F6")
    plain_box(ax, 5.3, 1.2, 4.5, 4.4, "#FDF0E3")

    box(ax, 0.2, 4.9, 4.5, 0.7,
        L("AI 코딩 도우미가 대신할 수 있는 일",
          "The AI assistant can do this", ko), BLUE, 11.5)
    box(ax, 5.3, 4.9, 4.5, 0.7,
        L("내가 책임지는 일",
          "This stays on me", ko), ORANGE, 11.5)

    left_items = [
        L("코드 초안을 빠르게 쓴다", "drafts code quickly", ko),
        L("오류 메시지의 뜻을 풀어 준다", "explains what an error means", ko),
        L("반복되는 코드를 채운다", "fills in repetitive code", ko),
        L("문법 실수를 먼저 찾아 준다", "spots syntax slips first", ko),
    ]
    right_items = [
        L("코드를 실제로 실행한다", "actually runs the code", ko),
        L("결과가 맞는지 판단한다", "judges whether the result is right", ko),
        L("개인정보·비밀번호·API 키를 넣지 않는다",
          "keeps secrets out of the prompt", ko),
        L("이해 못한 코드는 못했다고 적는다",
          "marks code I do not understand", ko),
    ]

    for i, text in enumerate(left_items):
        y = 4.35 - i * 0.85
        ax.text(0.55, y, "•", fontsize=13, color=BLUE, va="center")
        ax.text(0.95, y, text, fontsize=10.5, color="#222222", va="center")

    for i, text in enumerate(right_items):
        y = 4.35 - i * 0.85
        ax.text(5.65, y, "•", fontsize=13, color=ORANGE, va="center")
        ax.text(6.05, y, text, fontsize=10.5, color="#222222", va="center")

    ax.plot([5.0, 5.0], [1.3, 5.5], color=GRAY, linewidth=1.2, linestyle="--")

    ax.text(5.0, 0.6,
            L("AI가 쓴 코드라도 내가 제출하면 내 코드가 된다. 실행 결과의 책임은 오른쪽에 남는다.",
              "Code written by the AI becomes mine the moment I submit it.", ko),
            ha="center", fontsize=10.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 2장 개념 그림 생성 ===")

    targets = [
        ("fig2-1-code-to-result.png", figure_code_to_result,
         "코드가 결과로 바뀌기까지의 흐름"),
        ("fig2-2-error-reading.png", figure_error_reading,
         "오류 메시지 읽는 순서"),
        ("fig2-3-ai-boundary.png", figure_ai_boundary,
         "AI가 대신하는 일과 내가 책임지는 일"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")


if __name__ == "__main__":
    main()
