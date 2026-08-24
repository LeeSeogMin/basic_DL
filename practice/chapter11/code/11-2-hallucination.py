"""실습 11.2 - 환각이 기계적으로 왜 생기는지 같은 모델로 확인한다.

실습 11.1에서 만든 것과 똑같은 모델을 쓴다. 달라지는 것은 온도 하나다.
온도를 낮추면 확률 분포가 뾰족해지고, 높이면 평평해진다.

이 작은 세계에서 "사실"은 `_tinylm.py`의 학습 문장 12종뿐이다. 목록에 없는
문장은 어법이 맞아도 이 세계에서는 틀린 문장이다. 그 틀린 문장이 어떻게
만들어지는지, 어느 어절에서 갈라져 나오는지 본다.

학습 문장은 모두 합성 한국어 문장이다. 외부 API를 쓰지 않는다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _krfont
import _tinylm

# ---------------------------------------------------------------------------
# 학생이 바꿔 볼 값 (여기만 고치면 된다)
# ---------------------------------------------------------------------------
TEMPERATURES = [0.3, 1.0, 2.0]   # 온도 세 개. 낮으면 뾰족, 높으면 평평
START_WORD = "민준이는"           # 문장을 시작할 단어 = 사람이 주는 프롬프트
# ---------------------------------------------------------------------------

SEED = 2026        # 난수 고정
N_WORDS = 5        # 만들 어절 수
N_SAMPLES = 200    # 온도마다 만들어 볼 문장 개수
N_SHOW = 3         # 화면에 보여줄 문장 개수

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


def mark_divergence(words: list[str], truths: set[str]) -> tuple[str, int]:
    """학습 문장에서 갈라져 나온 어절을 대괄호로 표시한다.

    Returns:
        (표시한 문장, 갈라진 어절 번호). 학습 문장 그대로면 번호가 0이다.
    """
    k = _tinylm.match_length(words, truths)
    if k >= len(words):
        return " ".join(words), 0

    marked = list(words)
    marked[k] = f"[{marked[k]}]"
    return " ".join(marked), k + 1


def sample_sentences(vocab, probs, temperature, rng, truths):
    """한 온도로 문장을 N_SAMPLES개 만들고 결과를 모은다."""
    made = []
    for _ in range(N_SAMPLES):
        words, _ps = _tinylm.generate(
            vocab, probs, START_WORD, N_WORDS, rng, temperature=temperature
        )
        sentence = " ".join(words)
        marked, cut = mark_divergence(words, truths)
        made.append((sentence, sentence in truths, marked, cut))
    return made


def draw_figure(vocab, probs, summary, ko: bool, out_path: Path) -> None:
    """왼쪽에 온도별 확률 분포, 오른쪽에 온도별 새 조합 비율을 그린다."""
    L = _krfont.label
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    # 왼쪽: START_WORD 다음 단어 확률을 온도별로 나란히
    base = _tinylm.next_word_table(vocab, probs, START_WORD, 1.0)
    labels = [word for word, _ in base]
    if not ko:
        labels = [f"w{i + 1}" for i in range(len(labels))]

    x = np.arange(len(labels), dtype=float)
    width = 0.26
    colors = [GREEN, BLUE, RED]
    for i, (temperature, color) in enumerate(zip(TEMPERATURES, colors)):
        row = dict(_tinylm.next_word_table(vocab, probs, START_WORD, temperature))
        values = [row.get(word, 0.0) for word, _ in base]
        offset = (i - (len(TEMPERATURES) - 1) / 2) * width
        ax_left.bar(x + offset, values, width=width, color=color,
                    label=f"T = {temperature}")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontsize=10.5)
    ax_left.set_ylim(0, 1.12)
    ax_left.set_yticks([0, 0.5, 1.0])
    ax_left.set_yticklabels(["0%", "50%", "100%"], fontsize=10)
    ax_left.set_ylabel(L("뽑힐 확률", "probability", ko), fontsize=10.5)
    ax_left.set_title(
        L(f"'{START_WORD}' 다음 단어의 확률", "next-word probabilities", ko),
        fontsize=12.5, pad=10,
    )
    ax_left.spines[["top", "right"]].set_visible(False)
    ax_left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                   ncol=3, frameon=False, fontsize=10)

    # 오른쪽: 온도별로 학습 문장에 없던 조합이 나온 비율
    ratios = [row["new_ratio"] for row in summary]
    kinds = [row["n_kinds"] for row in summary]
    bars = ax_right.bar([f"T = {t}" for t in TEMPERATURES], ratios,
                        color=colors, width=0.5)
    ax_right.set_ylim(0, 1.12)
    ax_right.set_yticks([0, 0.5, 1.0])
    ax_right.set_yticklabels(["0%", "50%", "100%"], fontsize=10)
    ax_right.set_ylabel(
        L("학습 문장에 없던 조합의 비율", "share of unseen combinations", ko),
        fontsize=10.5,
    )
    ax_right.set_title(
        L(f"{N_SAMPLES}문장 중 학습 문장에 없던 조합",
          f"unseen combinations out of {N_SAMPLES}", ko),
        fontsize=12.5, pad=10,
    )
    ax_right.spines[["top", "right"]].set_visible(False)
    for rect, ratio, kind in zip(bars, ratios, kinds):
        ax_right.text(rect.get_x() + rect.get_width() / 2, ratio + 0.03,
                      L(f"{ratio * 100:.0f}%\n문장 {kind}종",
                        f"{ratio * 100:.0f}%\n{kind} kinds", ko),
                      ha="center", fontsize=10, color="#333333")

    fig.suptitle(
        L("온도를 올리면 분포가 평평해지고, 학습 문장에 없던 조합이 늘어난다",
          "Higher temperature flattens the distribution and produces more unseen combinations", ko),
        fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    rng = np.random.default_rng(SEED)
    vocab, _counts, probs, corpus = _tinylm.build_model()
    truths = _tinylm.unique_sentences()

    print("=== 실습 11.2  환각은 어디서 생기는가 ===")
    print("데이터: 코드 안에 적어 둔 합성 한국어 문장(실제 기록 아님). 외부 API를 쓰지 않는다.")
    print(f"이 세계에서 참인 문장: {len(truths)}종  (이 목록에 없으면 틀린 문장이다)")
    print(f"횟수만큼 펼친 학습 문장: {len(corpus)}개")
    print(f"설정: START_WORD={START_WORD}, TEMPERATURES={TEMPERATURES}, "
          f"N_SAMPLES={N_SAMPLES}, SEED={SEED}")
    print()

    print("[1] 온도가 확률 분포를 어떻게 바꾸는가")
    print(f"  '{START_WORD}' 다음에 올 단어의 확률")
    base = _tinylm.next_word_table(vocab, probs, START_WORD, 1.0)
    header = "    " + f"{'단어':<10}" + "".join(f"T={t:<7}" for t in TEMPERATURES)
    print(header)
    for word, _p in base:
        line = f"    {word:<10}"
        for temperature in TEMPERATURES:
            row = dict(_tinylm.next_word_table(vocab, probs, START_WORD, temperature))
            line += f"{row.get(word, 0.0) * 100:5.1f}%  "
        print(line)
    print("  T가 작으면 1등 단어가 거의 다 가져간다. T가 크면 격차가 줄어든다.")
    print("  단어의 순위는 바뀌지 않는다. 격차만 바뀐다.")
    print()

    print(f"[2] 온도마다 문장을 {N_SAMPLES}개씩 만든다")
    summary = []
    example_rows = []
    for temperature in TEMPERATURES:
        made = sample_sentences(vocab, probs, temperature, rng, truths)
        n_new = sum(1 for _s, in_train, _m, _c in made if not in_train)
        kinds = len({s for s, _t, _m, _c in made})
        summary.append({
            "temperature": temperature,
            "n_new": n_new,
            "new_ratio": n_new / len(made),
            "n_kinds": kinds,
        })

        print(f"  --- T = {temperature} ---")
        for i, (sentence, in_train, marked, cut) in enumerate(made[:N_SHOW], start=1):
            if in_train:
                note = "학습 문장 그대로"
            else:
                note = f"{cut}번째 어절에서 갈라짐 -> 학습 문장에 없는 조합"
            print(f"   {i}. {marked}")
            print(f"      {note}")
            example_rows.append([
                temperature, i, sentence,
                "있다" if in_train else "없다",
                cut if cut else "",
            ])
        first_new = next((item for item in made if not item[1]), None)
        if first_new is not None:
            _s, _t, marked, cut = first_new
            print(f"   이 온도에서 처음 나온 새 조합: {marked}")
            print(f"      {cut}번째 어절에서 갈라짐")
        print(f"   {N_SAMPLES}문장 중 학습 문장에 없던 조합: {n_new}개 "
              f"({n_new / len(made) * 100:.1f}%), 서로 다른 문장 {kinds}종")
        print()

    print("[3] 온도별 정리")
    print(f"  {'온도':<8}{'새 조합 수':<12}{'새 조합 비율':<14}{'문장 종류':<10}")
    for row in summary:
        print(f"  {row['temperature']:<8}{row['n_new']:<12}"
              f"{row['new_ratio'] * 100:5.1f}%        {row['n_kinds']:<10}")
    print()

    print("[4] 환각이 만들어지는 자리")
    print("  갈라지는 자리는 확률이 반반에 가까운 지점이다.")
    p_row = dict(_tinylm.next_word_table(vocab, probs, "도서관에서", 1.0))
    for word, p in sorted(p_row.items(), key=lambda item: -item[1]):
        print(f"    P({word} | 도서관에서) = {p * 100:.1f}%")
    print("  두 값이 비슷하면 어느 쪽이 뽑혀도 어법은 맞는다.")
    print("  그런데 이 세계에서 참인 조합은 한쪽뿐이다. 모델은 그 차이를 모른다.")
    print("  온도를 0에 가깝게 낮춰도 이 갈림은 사라지지 않는다. 1등 쪽으로 굳을 뿐이다.")
    print("  1등이 틀린 답이면, 온도를 낮출수록 늘 틀린 답을 낸다.")
    print()

    fig_path = RESULTS_DIR / "hallucination_by_temperature.png"
    draw_figure(vocab, probs, summary, ko, fig_path)
    print(f"저장: results/{fig_path.name}")

    ex_path = RESULTS_DIR / "hallucination_examples.csv"
    with ex_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["온도", "번호", "만들어진 문장", "학습 문장에 있는가",
                         "갈라진 어절 번호"])
        writer.writerows(example_rows)
    print(f"저장: results/{ex_path.name}")

    sum_path = RESULTS_DIR / "hallucination_summary.csv"
    with sum_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["온도", "새 조합 수", "새 조합 비율", "문장 종류 수"])
        for row in summary:
            writer.writerow([row["temperature"], row["n_new"],
                             f"{row['new_ratio']:.3f}", row["n_kinds"]])
    print(f"저장: results/{sum_path.name}")


if __name__ == "__main__":
    main()
