"""12장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig12-1  자동완성과 코딩 에이전트가 맡는 범위 비교
  fig12-2  권한 설정 세 단계
  fig12-3  저장 지점을 만들고, 기록을 보고, 되돌리는 순서

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


def figure_permission_levels(ko: bool, out_path: Path) -> None:
    """그림 12-2: 권한 설정 세 단계."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 6.2))

    ax.set_title(
        L("권한 설정 세 단계 — 넓힐수록 묻는 횟수가 줄고, 확인 없이 실행되는 일이 늘어난다",
          "Three permission levels: wider means fewer questions and more unchecked actions", ko),
        fontsize=13, pad=10, loc="left",
    )

    levels = [
        (L("읽기 전용", "read-only", ko), GREEN,
         L("파일을 읽기만 한다\n고치거나 지우지 못한다",
           "can only read files\ncannot edit or delete", ko),
         L("쓰는 때: 처음 보는 폴더를\n살펴보게 할 때",
           "use when: exploring\nan unfamiliar folder", ko)),
        (L("작업 폴더 쓰기", "workspace-write", ko), ORANGE,
         L("맡긴 폴더 안에서만 고친다\n폴더 밖·삭제·인터넷은 묻는다",
           "edits only inside the folder\nasks for outside, delete, network", ko),
         L("쓰는 때: 평소 작업",
           "use when: everyday work", ko)),
        (L("전부 허용", "allow all", ko), RED,
         L("아무것도 묻지 않는다\n삭제와 전송도 바로 실행한다",
           "asks nothing\ndeletes and uploads at once", ko),
         L("쓰는 때: 지워져도 되는\n격리된 환경에서만",
           "use when: only in an isolated\nthrowaway environment", ko)),
    ]

    xs = [0.2, 3.5, 6.8]
    w = 3.0
    for x, (name, color, can, when) in zip(xs, levels):
        box(ax, x, 4.3, w, 0.95, name, color, 13)
        ax.text(x + w / 2, 3.35, can, ha="center", va="center",
                fontsize=10.5, color="#333333", linespacing=1.6)
        ax.text(x + w / 2, 2.05, when, ha="center", va="center",
                fontsize=10.5, color=GRAY, linespacing=1.6)
    for x in xs[:-1]:
        arrow(ax, x + w + 0.05, 4.78, x + w + 0.27, 4.78)

    ax.text(5.0, 0.6,
            L("규칙 파일에 적은 문장은 부탁이다. 권한 설정은 실제로 막는 장치다.",
              "A sentence in a rules file is a request. A permission setting actually blocks.", ko),
            ha="center", fontsize=11.5, color=BLUE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_savepoint(ko: bool, out_path: Path) -> None:
    """그림 12-3: 저장 지점을 만들고, 기록을 보고, 되돌린다."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 5.8))

    ax.set_title(
        L("맡기기 전에 저장 지점을 만들고, 맡긴 뒤에 기록을 본다",
          "Save a point before delegating; read the record afterwards", ko),
        fontsize=13, pad=10, loc="left",
    )

    steps = [
        (L("① 저장 지점\n만들기", "1. make a\nsave point", ko), BLUE,
         L("지금 상태를\n따로 남긴다", "keep the\ncurrent state", ko)),
        (L("② 에이전트에게\n맡기기", "2. hand over\nto the agent", ko), GREEN,
         L("파일이 만들어지고\n고쳐지고 지워진다", "files are created,\nedited, deleted", ko)),
        (L("③ 기록 보기", "3. read the\nrecord", ko), BLUE,
         L("바뀐 파일 목록을\n보고와 대조한다", "compare changed files\nwith the report", ko)),
        (L("④ 받아들이거나\n되돌리기", "4. accept or\nroll back", ko), BLUE,
         L("틀렸으면 ①로\n돌아간다", "if wrong, go\nback to 1", ko)),
    ]

    xs = [0.1, 2.6, 5.1, 7.6]
    w = 2.3
    for x, (title, color, note) in zip(xs, steps):
        box(ax, x, 3.2, w, 1.4, title, color, 11)
        ax.text(x + w / 2, 2.45, note, ha="center", va="center",
                fontsize=10.5, color="#333333", linespacing=1.6)
    for x in xs[:-1]:
        arrow(ax, x + w + 0.03, 3.9, x + w + 0.19, 3.9)

    ax.text(5.0, 0.95,
            L("되돌릴 수 없는 것: 인터넷으로 나간 데이터, 저장 지점 밖에 있던 파일.\n"
              "파란 칸은 사람이 하고, 초록 칸만 에이전트가 한다.",
              "Cannot be undone: data already sent out, files outside the save point.\n"
              "Blue steps are the human's; only the green step is the agent's.", ko),
            ha="center", fontsize=10.5, color=RED, linespacing=1.7)

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
        ("fig12-2-permission-levels.png", figure_permission_levels,
         "권한 설정 세 단계"),
        ("fig12-3-savepoint.png", figure_savepoint,
         "저장 지점과 되돌리기"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")


if __name__ == "__main__":
    main()
