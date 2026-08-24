"""13장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig13-1  모델을 돌리는 것과 남이 쓸 수 있게 만드는 것의 차이
  fig13-2  나쁜 출력과 좋은 출력
  fig13-3  남이 실행하지 못하는 코드와 남이 실행할 수 있는 코드

이 그림들은 실험 결과가 아니라 개념 설명용 도식이다.
실제 수치는 실습 13.1과 13.2에서 잰다.
확신 기준선의 맞바꿈은 지어낸 곡선으로 그리지 않고, 실습 13.2에서 실제로 세어 그린다.
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


def card(ax, x, y, w, h, edgecolor, facecolor):
    """옅은 색으로 채운 카드 상자를 그린다. 글자는 호출한 쪽에서 따로 넣는다."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.8,
        )
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


def figure_model_vs_app(ko: bool, out_path: Path) -> None:
    """그림 13-1: 모델을 돌리는 것과 남이 쓸 수 있게 만드는 것."""
    L = _krfont.label
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.6))

    # 위쪽: 모델을 돌리기만 하는 상태
    ax = axes[0]
    blank_axes(ax, xlim=(0, 10), ylim=(0, 4))
    ax.set_title(
        L("모델을 돌린다 — 만든 사람만 쓸 수 있다",
          "Running a model - only its author can use it", ko),
        fontsize=13, pad=8, loc="left",
    )
    box(ax, 0.4, 1.5, 2.8, 1.6,
        L("내 컴퓨터에서\n코드를 실행한다",
          "Run the script\non my own machine", ko), BLUE, 11)
    box(ax, 4.0, 1.5, 2.8, 1.6,
        L("화면에 숫자가\n찍힌다  0.8213",
          "A number is printed\n0.8213", ko), GRAY, 11)
    arrow(ax, 3.3, 2.3, 3.9, 2.3)
    ax.text(7.1, 2.3,
            L("남은 이 숫자가\n무슨 뜻인지 모른다",
              "Others cannot tell\nwhat this number means", ko),
            ha="left", va="center", fontsize=11, color=RED, linespacing=1.5)

    # 아래쪽: 남이 쓸 수 있는 앱
    ax = axes[1]
    blank_axes(ax, xlim=(0, 10), ylim=(0, 4))
    ax.set_title(
        L("남이 쓸 수 있게 만든다 — 다섯 단계가 더 붙는다",
          "Building an app - five more steps are added", ko),
        fontsize=13, pad=8, loc="left",
    )

    steps = [
        (L("① 입력 받기\n무엇을 넣는지\n알려준다",
           "1 Take input\nsay what to enter", ko), BLUE),
        (L("② 입력 검사\n이상한 값을\n걸러낸다",
           "2 Check input\nreject bad values", ko), BLUE),
        (L("③ 모델 처리\n예측과 확률을\n계산한다",
           "3 Run the model\npredict + probability", ko), GREEN),
        (L("④ 결과 보이기\n확률과 근거를\n함께 쓴다",
           "4 Show result\nprobability + reason", ko), ORANGE),
        (L("⑤ 한계 알리기\n틀릴 수 있는\n경우를 적는다",
           "5 State limits\nwhen it can be wrong", ko), ORANGE),
    ]
    width, gap, left = 1.62, 0.32, 0.35
    for i, (text, color) in enumerate(steps):
        x = left + i * (width + gap)
        box(ax, x, 1.35, width, 1.85, text, color, 9.0)
        if i < len(steps) - 1:
            arrow(ax, x + width + 0.02, 2.28, x + width + gap - 0.02, 2.28)

    ax.text(5.0, 0.55,
            L("③ 하나만 있으면 모델이고, 다섯 단계가 다 있어야 남이 쓸 수 있는 앱이다.",
              "Step 3 alone is a model; all five steps make an app someone else can use.", ko),
            ha="center", fontsize=10.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_output_honesty(ko: bool, out_path: Path) -> None:
    """그림 13-2: 단정하는 출력과 확률·근거·한계를 함께 쓴 출력."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 7.2))

    ax.set_title(
        L("같은 예측, 다른 화면 문구",
          "Same prediction, different wording on screen", ko),
        fontsize=13.5, pad=10, loc="left",
    )

    # 왼쪽 카드 - 나쁜 출력
    card(ax, 0.25, 0.85, 4.4, 5.4, RED, "#FBEAEA")
    ax.text(2.45, 5.75, L("이렇게 쓰지 않는다", "Do not write it this way", ko),
            ha="center", fontsize=12, color=RED)
    bad_lines = [
        L("결과: 스팸입니다", "Result: this is spam", ko),
        L("정확도 96%의 AI가 판정했습니다", "Judged by a 96%-accurate AI", ko),
        L("믿고 삭제하세요", "You can safely delete it", ko),
    ]
    for i, line in enumerate(bad_lines):
        ax.text(0.6, 4.85 - i * 0.75, line, ha="left", fontsize=11, color="#333333")

    bad_notes = [
        L("· 확률을 숨기고 단정한다", "- hides the probability", ko),
        L("· 무엇을 보고 판단했는지 없다", "- no reason is shown", ko),
        L("· 틀릴 수 있다는 말이 없다", "- never says it can be wrong", ko),
    ]
    for i, line in enumerate(bad_notes):
        ax.text(0.6, 2.05 - i * 0.5, line, ha="left", fontsize=10, color=RED)

    # 오른쪽 카드 - 좋은 출력
    card(ax, 5.35, 0.85, 4.4, 5.4, GREEN, "#EAF3E7")
    ax.text(7.55, 5.75, L("이렇게 쓴다", "Write it this way", ko),
            ha="center", fontsize=12, color=GREEN)
    good_lines = [
        L("스팸일 확률 0.82", "Probability of spam: 0.82", ko),
        L("판단에 쓴 값: 링크 7개, 대문자 32%",
          "Values used: 7 links, 32% caps", ko),
        L("이 모델이 틀릴 수 있는 경우:",
          "This model can be wrong when:", ko),
        L("  합성 메일로만 배웠고, 한국어", "  trained on synthetic mail only,", ko),
        L("  광고 메일은 본 적이 없다", "  never saw Korean ad mail", ko),
    ]
    for i, line in enumerate(good_lines):
        ax.text(5.7, 4.85 - i * 0.55, line, ha="left", fontsize=10.5, color="#333333")

    good_notes = [
        L("· 확률을 그대로 보여준다", "- shows the probability", ko),
        L("· 판단에 쓴 값을 밝힌다", "- shows the values used", ko),
        L("· 틀리는 조건을 먼저 적는다", "- states failure cases up front", ko),
    ]
    for i, line in enumerate(good_notes):
        ax.text(5.7, 2.05 - i * 0.5, line, ha="left", fontsize=10, color=GREEN)

    ax.text(5.0, 0.25,
            L("예측값은 같다. 사용자가 이 결과를 얼마나 믿을지가 달라진다.",
              "The prediction is identical; how far the user trusts it is not.", ko),
            ha="center", fontsize=10.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_reproducible_run(ko: bool, out_path: Path) -> None:
    """그림 13-3: 남이 실행하지 못하는 코드와 남이 실행할 수 있는 코드."""
    L = _krfont.label
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 7.0))

    # 위쪽: 남이 실행하지 못하는 상태
    ax = axes[0]
    blank_axes(ax, xlim=(0, 10), ylim=(0, 4))
    ax.set_title(
        L("남이 실행하지 못한다 — 세 가지가 빠졌다",
          "Nobody else can run it - three things are missing", ko),
        fontsize=13, pad=8, loc="left",
    )
    bad_steps = [
        L("파일 경로가\n내 컴퓨터에만 있다",
          "file paths exist only\non my own machine", ko),
        L("어떤 패키지가\n필요한지 안 적었다",
          "required packages\nare not written down", ko),
        L("실행할 때마다\n결과가 달라진다",
          "results change on\nevery run", ko),
    ]
    width, gap, left = 2.30, 0.32, 0.35
    for i, text in enumerate(bad_steps):
        x = left + i * (width + gap)
        box(ax, x, 1.30, width, 1.75, text, RED, 10.0)
        if i < len(bad_steps) - 1:
            arrow(ax, x + width + 0.02, 2.18, x + width + gap - 0.02, 2.18)

    ax.text(8.05, 2.18,
            L("남이 받아 실행하면\n오류가 난다",
              "another person\nhits an error", ko),
            ha="left", va="center", fontsize=10.5, color=RED, linespacing=1.5)

    # 아래쪽: 남이 실행할 수 있는 상태
    ax = axes[1]
    blank_axes(ax, xlim=(0, 10), ylim=(0, 4))
    ax.set_title(
        L("남이 실행할 수 있다 — 네 가지를 갖춘다",
          "Anyone can run it - four things are in place", ko),
        fontsize=13, pad=8, loc="left",
    )
    good_steps = [
        L("① 코드와 데이터가\n한 폴더에 있다",
          "1 code and data\nin one folder", ko),
        L("② 필요한 패키지를\n목록으로 적었다",
          "2 packages listed\nin one file", ko),
        L("③ 난수 seed를\n고정했다",
          "3 random seed\nis fixed", ko),
        L("④ 실행 명령\n한 줄을 적었다",
          "4 one command line\nis written down", ko),
    ]
    width, gap, left = 2.05, 0.30, 0.35
    for i, text in enumerate(good_steps):
        x = left + i * (width + gap)
        box(ax, x, 1.30, width, 1.75, text, GREEN, 10.0)
        if i < len(good_steps) - 1:
            arrow(ax, x + width + 0.02, 2.18, x + width + gap - 0.02, 2.18)

    ax.text(5.0, 0.50,
            L("같은 명령을 넣으면 남의 컴퓨터에서도 같은 숫자가 나온다. 이것이 재현 가능한 실행이다.",
              "The same command on another machine gives the same numbers "
              "- that is a reproducible run.", ko),
            ha="center", fontsize=10.5, color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 13장 개념 그림 생성 ===")

    targets = [
        ("fig13-1-model-vs-app.png", figure_model_vs_app,
         "모델을 돌리는 것과 남이 쓸 수 있게 만드는 것"),
        ("fig13-2-output-honesty.png", figure_output_honesty,
         "나쁜 출력과 좋은 출력"),
        ("fig13-3-reproducible-run.png", figure_reproducible_run,
         "남이 실행하지 못하는 코드와 실행할 수 있는 코드"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")
    print("실제 수치는 실습 13.1과 13.2의 로그에서 인용한다.")


if __name__ == "__main__":
    main()
