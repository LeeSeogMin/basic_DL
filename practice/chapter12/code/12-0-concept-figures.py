"""12장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig12-1  자동완성과 코딩 에이전트가 맡는 범위 비교
  fig12-2  큰 요구 하나를 작업 단위로 쪼개는 방법
  fig12-3  AI에게 맡긴 일과 사람이 책임지는 일의 경계

이 그림들은 실험 결과가 아니라 개념 설명용 도식이다.
외부 서비스에 접속하지 않는다. 그림 파일만 만든다.
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


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            color=color,
        )
    )


def blank_axes(ax, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


# 다섯 단계를 두 그림에서 똑같이 쓴다.
STEP_X = [0.0, 2.05, 4.1, 6.15, 8.2]
STEP_W = 1.8


def draw_pipeline(ax, ko, steps, owners, colors):
    """다섯 단계 상자를 한 줄로 그리고, 아래에 담당을 적는다."""
    for x, text, owner, color in zip(STEP_X, steps, owners, colors):
        box(ax, x, 1.7, STEP_W, 1.9, text, color, 10.5)
        ax.text(
            x + STEP_W / 2, 1.25, owner,
            ha="center", va="center", fontsize=10.5, color="#333333",
        )
    for x in STEP_X[:-1]:
        arrow(ax, x + STEP_W + 0.03, 2.65, x + STEP_W + 0.22, 2.65)


def figure_autocomplete_vs_agent(ko: bool, out_path: Path) -> None:
    """그림 12-1: 무엇을 얼마나 맡기는가."""
    L = _krfont.label
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.6))

    steps = [
        L("무엇을 만들지\n정한다", "decide what\nto build", ko),
        L("작업으로\n쪼갠다", "split into\ntasks", ko),
        L("코드를\n쓴다", "write the\ncode", ko),
        L("실행하고\n확인한다", "run and\ncheck", ko),
        L("틀린 곳을\n고친다", "fix what\nis wrong", ko),
    ]
    human = L("사람", "human", ko)
    both = L("사람 + AI", "human + AI", ko)
    ai = L("AI", "AI", ko)

    # 위쪽: 자동완성
    ax = axes[0]
    blank_axes(ax, ylim=(0, 4.4))
    ax.set_title(
        L("자동완성 — 내가 쓰는 줄의 다음 줄을 제안한다",
          "Autocomplete: suggests the next line I am typing", ko),
        fontsize=13, pad=8, loc="left",
    )
    draw_pipeline(
        ax, ko, steps,
        [human, human, both, human, human],
        [BLUE, BLUE, ORANGE, BLUE, BLUE],
    )
    ax.text(
        5.0, 0.5,
        L("사람이 계속 운전한다. AI는 한 번에 몇 줄만 제안하고, 받아들일지는 그 자리에서 정한다.",
          "The human keeps driving; the AI only offers a few lines at a time.", ko),
        ha="center", fontsize=10.5, color=GRAY,
    )

    # 아래쪽: 코딩 에이전트
    ax = axes[1]
    blank_axes(ax, ylim=(0, 4.4))
    ax.set_title(
        L("코딩 에이전트 — 작업 하나를 통째로 맡아 코드를 쓰고 실행하고 고친다",
          "Coding agent: takes a whole task, writes, runs, and fixes code", ko),
        fontsize=13, pad=8, loc="left",
    )
    draw_pipeline(
        ax, ko, steps,
        [human, both, ai, ai, ai],
        [BLUE, ORANGE, GREEN, GREEN, GREEN],
    )
    ax.text(
        5.0, 0.5,
        L("맡기는 범위가 넓어질수록, 사람이 결과를 확인하지 않으면 무엇이 바뀌었는지 모른 채 지나간다.",
          "The wider the delegation, the easier it is to miss what actually changed.", ko),
        ha="center", fontsize=10.5, color=RED,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_task_breakdown(ko: bool, out_path: Path) -> None:
    """그림 12-2: 큰 요구를 작업 단위로 쪼갠다."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 6.6))

    ax.set_title(
        L("큰 요구 한 줄을 확인 가능한 작업 네 개로 쪼갠다",
          "Split one big request into four checkable tasks", ko),
        fontsize=13, pad=10, loc="left",
    )

    box(ax, 0.6, 5.0, 8.8, 0.95,
        L("\"폴더를 정리해 주는 도구를 만들어 줘\"  ← 확인할 방법이 없는 요구",
          "\"Build me a folder tidy-up tool\"  <- no way to check it", ko),
        RED, 12)

    tasks = [
        (L("① 폴더 안 파일\n목록을 읽는다",
           "1. list files\nin the folder", ko),
         L("확인: 파일 수가\n실제와 같은가", "check: file count\nmatches", ko)),
        (L("② 확장자별로 개수와\n용량을 센다",
           "2. count and size\nby extension", ko),
         L("확인: 용량 합이\n전체와 같은가", "check: sizes add up\nto the total", ko)),
        (L("③ 표로 출력한다",
           "3. print a table", ko),
         L("확인: 표에 빠진\n확장자가 없는가", "check: no extension\nis missing", ko)),
        (L("④ 막대그래프를\n그린다",
           "4. draw a bar chart", ko),
         L("확인: 그림 파일이\n실제로 생겼는가", "check: the image file\nexists", ko)),
    ]

    xs = [0.1, 2.6, 5.1, 7.6]
    w = 2.3
    for x, (title, check) in zip(xs, tasks):
        arrow(ax, 5.0, 4.95, x + w / 2, 3.95)
        box(ax, x, 2.5, w, 1.4, title, BLUE, 10.5)
        ax.text(x + w / 2, 1.75, check,
                ha="center", va="center", fontsize=10, color="#333333",
                linespacing=1.5)

    ax.text(5.0, 0.75,
            L("작업 하나가 끝날 때마다 맞는지 확인할 수 있으면, 어디서 틀렸는지 바로 찾는다.\n"
              "네 작업을 한꺼번에 맡기면, 틀렸을 때 어느 단계가 문제인지 알 수 없다.",
              "If each task can be checked, you find the broken step immediately.\n"
              "If all four go at once, you cannot tell which step failed.", ko),
            ha="center", fontsize=10.5, color=GRAY, linespacing=1.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_responsibility(ko: bool, out_path: Path) -> None:
    """그림 12-3: 맡긴 일과 책임지는 일의 경계."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 7.2))

    ax.set_title(
        L("맡길 수 있는 일과 넘길 수 없는 일",
          "What can be delegated and what cannot", ko),
        fontsize=13, pad=10, loc="left",
    )

    ax.plot([5.0, 5.0], [0.9, 6.5], color="#333333", linewidth=2.2)

    box(ax, 0.2, 5.7, 4.4, 0.8,
        L("AI에게 맡길 수 있는 일", "Can be delegated to AI", ko), GREEN, 12)
    box(ax, 5.4, 5.7, 4.4, 0.8,
        L("사람이 책임지는 일", "Stays with the human", ko), BLUE, 12)

    left_items = [
        L("비슷한 코드를 여러 벌 만들기", "writing repetitive code", ko),
        L("문법 오류 찾아 고치기", "fixing syntax errors", ko),
        L("함수 이름과 주석 다듬기", "naming and comments", ko),
        L("형식 변환 같은 반복 작업", "repetitive conversions", ko),
    ]
    right_items = [
        L("무엇을 만들지 정하기", "deciding what to build", ko),
        L("어떤 데이터를 쓸지 정하기", "choosing the data", ko),
        L("실행해도 되는지 판단하기", "deciding whether to run it", ko),
        L("결과가 맞는지 확인하기", "checking the result", ko),
        L("틀렸을 때 책임지기", "owning the mistake", ko),
    ]

    for i, text in enumerate(left_items):
        y = 5.05 - 0.62 * i
        ax.text(0.35, y, "•  " + text, ha="left", va="center",
                fontsize=11, color="#333333")
    for i, text in enumerate(right_items):
        y = 5.05 - 0.62 * i
        ax.text(5.55, y, "•  " + text, ha="left", va="center",
                fontsize=11, color="#333333")

    ax.text(5.0, 0.35,
            L("AI가 코드를 썼어도, 그 코드를 실행해서 생긴 결과는 실행 버튼을 누른 사람의 결과다.",
              "Even if the AI wrote the code, the result belongs to whoever ran it.", ko),
            ha="center", fontsize=11.5, color=RED)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 12장 개념 그림 생성 ===")

    targets = [
        ("fig12-1-autocomplete-vs-agent.png", figure_autocomplete_vs_agent,
         "자동완성과 코딩 에이전트가 맡는 범위"),
        ("fig12-2-task-breakdown.png", figure_task_breakdown,
         "큰 요구를 작업 단위로 쪼개기"),
        ("fig12-3-responsibility-line.png", figure_responsibility,
         "맡긴 일과 책임지는 일의 경계"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")


if __name__ == "__main__":
    main()
