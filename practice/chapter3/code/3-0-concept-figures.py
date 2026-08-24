"""3장 이론 강의용 개념 그림을 만든다.

만드는 그림은 세 장이다.
  fig3-1  리스트, 딕셔너리, 표가 서로 어떻게 대응하는가
  fig3-2  표 한 장에서 무엇이 특성이고 무엇이 라벨인가
  fig3-3  나쁜 데이터의 네 가지 모습

이 그림들은 데이터 분석 결과가 아니라 개념 설명용 도식이다.
그림 안의 표는 설명을 위해 지어낸 값이며, 실제 학생 자료가 아니다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import _krfont

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def blank_axes(ax, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


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
            mutation_scale=16,
            linewidth=1.6,
            color=color,
        )
    )


def cell(ax, x, y, w, h, text, facecolor, textcolor="#222222",
         fontsize=10.5, edgecolor="white", linewidth=1.4, style="normal"):
    """표의 칸 하나를 그린다."""
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=facecolor,
                  edgecolor=edgecolor, linewidth=linewidth)
    )
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, style=style)


def mini_table(ax, x0, y0, col_w, row_h, headers, rows,
               header_color=GRAY, body_color="#EFEFEF",
               cell_colors=None, fontsize=9.5):
    """작은 표 하나를 그린다.

    cell_colors 는 {(행번호, 열번호): 색} 형태이며, 지정한 칸만 색을 바꾼다.
    행번호 0이 첫 데이터 행이다(머리글 행은 세지 않는다).
    """
    cell_colors = cell_colors or {}
    n_col = len(headers)
    top_y = y0 + row_h * len(rows)

    for c, name in enumerate(headers):
        cell(ax, x0 + c * col_w, top_y, col_w, row_h, name,
             header_color, textcolor="white", fontsize=fontsize)

    for r, row in enumerate(rows):
        y = top_y - row_h * (r + 1)
        for c in range(n_col):
            color = cell_colors.get((r, c), body_color)
            text = row[c]
            style = "italic" if isinstance(text, str) and text.startswith('"') else "normal"
            cell(ax, x0 + c * col_w, y, col_w, row_h, text, color,
                 fontsize=fontsize, style=style)

    return x0 + n_col * col_w, top_y + row_h


def figure_feature_label(ko: bool, out_path: Path) -> None:
    """그림 3-2: 표의 어느 열이 특성이고 어느 열이 라벨인가."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 6))

    ax.set_title(
        L("표 한 장에 이름표를 붙인다 — 특성과 라벨",
          "Labeling one table: features and label", ko),
        fontsize=14, pad=14, loc="left",
    )

    headers = [
        L("공부시간", "study hrs", ko),
        L("수면시간", "sleep hrs", ko),
        L("과제제출", "homework", ko),
        L("동아리", "club", ko),
        L("시험통과", "passed", ko),
    ]
    rows = [
        ["4.0", "7.0", "5", L("있음", "yes", ko), L("통과", "pass", ko)],
        ["1.5", "5.0", "2", L("없음", "no", ko), L("미통과", "fail", ko)],
        ["6.5", "8.0", "6", L("있음", "yes", ko), L("통과", "pass", ko)],
        ["2.0", "8.5", "3", L("없음", "no", ko), L("미통과", "fail", ko)],
    ]

    col_w = 1.42
    row_h = 0.66
    x0 = 2.55
    y0 = 1.55

    feature_fill = "#DCE6F1"
    label_fill = "#FBE3CC"
    cell_colors = {(r, 4): label_fill for r in range(len(rows))}
    for r in range(len(rows)):
        for c in range(4):
            cell_colors[(r, c)] = feature_fill

    for c, name in enumerate(headers):
        head_color = ORANGE if c == 4 else BLUE
        cell(ax, x0 + c * col_w, y0 + row_h * len(rows), col_w, row_h,
             name, head_color, textcolor="white", fontsize=10)

    for r, row in enumerate(rows):
        y = y0 + row_h * (len(rows) - 1 - r)
        for c in range(len(headers)):
            cell(ax, x0 + c * col_w, y, col_w, row_h, row[c],
                 cell_colors[(r, c)], fontsize=10)

    table_top = y0 + row_h * (len(rows) + 1)

    # 특성 구간과 라벨 구간에 이름표를 단다
    ax.plot([x0, x0 + col_w * 4], [table_top + 0.16] * 2, color=BLUE, linewidth=2.2)
    ax.text(x0 + col_w * 2, table_top + 0.34,
            L("특성(feature) — 모델에게 주는 입력", "features - the input", ko),
            ha="center", fontsize=11.5, color=BLUE)

    ax.plot([x0 + col_w * 4, x0 + col_w * 5], [table_top + 0.16] * 2,
            color=ORANGE, linewidth=2.2)
    ax.text(x0 + col_w * 4.5, table_top + 0.34,
            L("라벨(label) — 맞혀야 하는 정답", "label - the answer", ko),
            ha="center", fontsize=11.5, color=ORANGE)

    # 행 하나의 뜻
    ax.annotate(
        L("행 하나\n= 학생 한 명\n= 데이터 한 건",
          "one row\n= one student\n= one sample", ko),
        xy=(x0 - 0.08, y0 + row_h * 3.5),
        xytext=(0.15, y0 + row_h * 3.5),
        fontsize=10.5, color=GRAY, va="center", ha="left", linespacing=1.6,
        arrowprops=dict(arrowstyle="-|>", color=GRAY, linewidth=1.4,
                        shrinkA=6, shrinkB=2),
    )

    ax.text(5.0, 0.62,
            L("무엇이 라벨인지는 표에 적혀 있지 않다. 내가 무엇을 묻는지가 정한다.\n"
              "\"시험을 통과할까\"라고 물으면 시험통과가 라벨이고, "
              "\"몇 시간 잤을까\"라고 물으면 수면시간이 라벨이다.",
              "The table does not say which column is the label.\n"
              "The question you ask decides it.", ko),
            ha="center", fontsize=10.5, color="#333333", linespacing=1.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_list_dict_table(ko: bool, out_path: Path) -> None:
    """그림 3-1: 리스트, 딕셔너리, 표가 같은 내용을 담는 세 가지 모습."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    blank_axes(ax, xlim=(0, 12), ylim=(0, 6))

    ax.set_title(
        L("같은 내용을 담는 세 가지 그릇 — 리스트, 딕셔너리, 표",
          "Three containers for the same content", ko),
        fontsize=14, pad=14, loc="left",
    )

    items = [L("떡볶이", "tteokbokki", ko), L("김밥", "gimbap", ko),
             L("샐러드", "salad", ko)]
    scores = ["9", "6", "4"]

    # --- 1) 리스트 ---
    ax.text(1.75, 4.7, L("리스트(list)", "list", ko),
            ha="center", fontsize=12.5, color=BLUE)
    ax.text(1.75, 4.25,
            L("순서(0, 1, 2)로 찾는다", "found by position", ko),
            ha="center", fontsize=10, color=GRAY)
    ax.text(1.38, 3.92, L("음식 목록", "foods", ko), ha="center",
            fontsize=9.5, color=BLUE)
    ax.text(2.36, 3.92, L("점수 목록", "scores", ko), ha="center",
            fontsize=9.5, color=BLUE)
    for i, (item, score) in enumerate(zip(items, scores)):
        y = 3.3 - i * 0.72
        ax.text(0.28, y + 0.25, f"[{i}]", ha="left", va="center",
                fontsize=9.5, color=GRAY)
        cell(ax, 0.75, y, 1.25, 0.5, item, "#DCE6F1", fontsize=10)
        cell(ax, 2.05, y, 0.62, 0.5, score, "#DCE6F1", fontsize=10)
    ax.text(1.75, 0.95, L("이름 목록과 점수 목록이 따로 있다.\n둘의 순서가 어긋나면 뜻이 무너진다.",
                          "Two separate lists.\nOrder must match.", ko),
            ha="center", fontsize=9.5, color="#333333", linespacing=1.6)

    arrow(ax, 2.95, 2.6, 3.85, 2.6)

    # --- 2) 딕셔너리 ---
    ax.text(5.55, 4.7, L("딕셔너리(dict)", "dict", ko),
            ha="center", fontsize=12.5, color=ORANGE)
    ax.text(5.55, 4.25, L("이름(키)으로 찾는다", "found by key", ko),
            ha="center", fontsize=10, color=GRAY)
    for i, (item, score) in enumerate(zip(items, scores)):
        y = 3.3 - i * 0.72
        cell(ax, 4.15, y, 1.45, 0.5, item, "#FBE3CC", fontsize=10)
        ax.text(5.72, y + 0.25, ":", ha="center", va="center",
                fontsize=12, color=GRAY)
        cell(ax, 5.85, y, 0.9, 0.5, score, "#FBE3CC", fontsize=10)
    ax.text(5.45, 0.95, L('점수를 꺼낼 때 순서를 몰라도 된다.\nscore["떡볶이"] 처럼 이름으로 부른다.',
                          'No position needed.\nCall it by name.', ko),
            ha="center", fontsize=9.5, color="#333333", linespacing=1.6)

    arrow(ax, 7.05, 2.6, 7.95, 2.6)

    # --- 3) 표 ---
    ax.text(9.65, 4.7, L("표(DataFrame)", "table (DataFrame)", ko),
            ha="center", fontsize=12.5, color=GREEN)
    ax.text(9.65, 4.25, L("열 이름 + 행 번호로 찾는다", "column name + row index", ko),
            ha="center", fontsize=10, color=GRAY)
    headers = [L("음식", "food", ko), L("점수", "score", ko)]
    rows = [[items[i], scores[i]] for i in range(3)]
    mini_table(ax, 8.25, 1.86, 1.4, 0.5, headers, rows,
               header_color=GREEN, body_color="#DDEEDA", fontsize=10)
    ax.text(9.65, 0.95, L("행과 열이 함께 붙어 다닌다.\n한 행을 지우면 이름과 점수가 같이 빠진다.",
                          "Rows and columns stay together.\nDrop a row, both go.", ko),
            ha="center", fontsize=9.5, color="#333333", linespacing=1.6)

    ax.text(6.0, 0.18,
            L("이 그림의 값은 설명용으로 지어낸 값이다.",
              "Values in this figure are illustrative only.", ko),
            ha="center", fontsize=9, color=GRAY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_bad_data(ko: bool, out_path: Path) -> None:
    """그림 3-3: 나쁜 데이터의 네 가지 모습."""
    L = _krfont.label
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))

    headers = [L("번호", "id", ko), L("공부", "study", ko),
               L("수면", "sleep", ko), L("점수", "score", ko)]

    panels = [
        (
            L("① 결측값 — 칸이 비어 있다", "1. Missing value", ko),
            [["1", "4.0", "7.0", "82"],
             ["2", "1.5", "5.0", "?"],
             ["3", "6.5", "8.0", "91"]],
            {(1, 3): "#F6C7C6"},
            L("설문에 답하지 않았거나 기록이 빠졌다.\n비어 있다는 사실 자체가 정보일 때도 있다.",
              "The value was never recorded.\nThe blank itself can be information.", ko),
        ),
        (
            L("② 중복 — 같은 행이 두 번 들어왔다", "2. Duplicate rows", ko),
            [["1", "4.0", "7.0", "82"],
             ["2", "1.5", "5.0", "60"],
             ["2", "1.5", "5.0", "60"]],
            {(1, 0): "#F6C7C6", (1, 1): "#F6C7C6", (1, 2): "#F6C7C6", (1, 3): "#F6C7C6",
             (2, 0): "#F6C7C6", (2, 1): "#F6C7C6", (2, 2): "#F6C7C6", (2, 3): "#F6C7C6"},
            L("한 사람을 두 번 세면 그 사람의 목소리가 두 배가 된다.\n파일을 두 번 붙였을 때 흔히 생긴다.",
              "Counting one person twice doubles their weight.\nUsually caused by pasting a file twice.", ko),
        ),
        (
            L("③ 이상치 — 있을 수 없는 값이다", "3. Outlier", ko),
            [["1", "4.0", "7.0", "82"],
             ["2", "1.5", "99.0", "60"],
             ["3", "-3.0", "8.0", "91"]],
            {(1, 2): "#F6C7C6", (2, 1): "#F6C7C6"},
            L("하루에 99시간을 자거나 -3시간을 공부할 수는 없다.\n입력 실수인지 진짜 특이한 값인지 나눠 봐야 한다.",
              "Nobody sleeps 99 hours a day.\nSeparate typos from genuinely rare values.", ko),
        ),
        (
            L("④ 잘못된 자료형 — 숫자가 글자로 들어왔다", "4. Wrong data type", ko),
            [["1", "4.0", "7.0", "82"],
             ["2", '"3.5"', "5.0", "60"],
             ["3", "6.5", "8.0", '"91"']],
            {(1, 1): "#F6C7C6", (2, 3): "#F6C7C6"},
            L('따옴표가 붙은 "3.5"는 숫자가 아니라 글자다.\n평균을 구하면 계산이 아예 되지 않는다.',
              'Quoted "3.5" is text, not a number.\nAveraging it fails outright.', ko),
        ),
    ]

    for ax, (title, rows, marks, note) in zip(axes.ravel(), panels):
        blank_axes(ax, xlim=(0, 10), ylim=(0, 6))
        ax.set_title(title, fontsize=12.5, pad=8, loc="left", color=RED)
        mini_table(ax, 1.35, 2.35, 1.8, 0.62, headers, rows,
                   header_color=GRAY, body_color="#EFEFEF",
                   cell_colors=marks, fontsize=10)
        ax.text(5.0, 1.15, note, ha="center", va="center",
                fontsize=10, color="#333333", linespacing=1.7)

    fig.suptitle(
        L("나쁜 데이터의 네 가지 모습 (설명용으로 지어낸 표)",
          "Four shapes of bad data (illustrative table)", ko),
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    print("=== 3장 개념 그림 생성 ===")

    targets = [
        ("fig3-1-list-dict-table.png", figure_list_dict_table,
         "리스트·딕셔너리·표의 대응"),
        ("fig3-2-feature-label.png", figure_feature_label,
         "표에서 특성과 라벨 구분하기"),
        ("fig3-3-bad-data-four.png", figure_bad_data,
         "나쁜 데이터의 네 가지 모습"),
    ]

    for filename, builder, description in targets:
        out_path = RESULTS_DIR / filename
        builder(ko, out_path)
        print(f"저장: results/{filename}  ({description})")

    print()
    print("이 그림은 개념 설명용 도식이며, 실험 결과가 아니다.")


if __name__ == "__main__":
    main()
