"""실습 11.1 - 다음 단어를 예측하는 아주 작은 언어모델을 직접 만든다.

학습 문장은 `_tinylm.py`에 들어 있는 **합성 한국어 문장**이다. 실제 사람도,
실제 기록도 아니고, 인터넷이나 외부 API를 전혀 쓰지 않는다.

하는 일은 세 가지다.
  1. "앞 단어 -> 다음 단어"가 몇 번 나왔는지 센다.
  2. 그 횟수를 확률로 바꾼다.
  3. 확률대로 단어를 하나씩 뽑아 이어 붙여 문장을 만든다.

이 모델은 단어의 뜻을 하나도 모른다. 아는 것은 빈도뿐이다.
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
START_WORD = "민준이는"   # 문장을 시작할 단어. 학습 문장에 있는 단어여야 한다
N_WORDS = 5               # 만들 어절 수. 학습 문장이 다섯 어절이라 5가 기본값이다
# ---------------------------------------------------------------------------

SEED = 11                 # 난수 고정. 같은 값이면 늘 같은 결과가 나온다
N_SENTENCES = 5           # 만들어 볼 문장 개수

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GRAY = "#79706E"


def generate_with_steps(vocab, probs, start_word, n_words, rng):
    """문장을 만들면서 각 단계의 확률표를 함께 기록한다.

    Returns:
        (만든 단어 목록, 단계 기록). 단계 기록의 항목 하나는
        (앞 단어, [(후보 단어, 확률), ...], 뽑힌 단어, 뽑힌 확률)이다.
    """
    words = [start_word]
    steps = []

    for _ in range(n_words - 1):
        row = probs[vocab.index(words[-1])]
        if row.sum() == 0:
            break  # 이 단어 뒤에 온 단어가 학습 문장에 없다. 문장을 끝낸다
        candidates = [(vocab[j], float(row[j])) for j in np.nonzero(row)[0]]
        candidates.sort(key=lambda item: item[1], reverse=True)
        j = int(rng.choice(len(vocab), p=row))
        steps.append((words[-1], candidates, vocab[j], float(row[j])))
        words.append(vocab[j])

    return words, steps


def print_count_table(vocab, counts, word) -> None:
    """한 단어 뒤에 어떤 단어가 몇 번 나왔는지 표로 찍는다."""
    row = counts[vocab.index(word)]
    total = row.sum()
    print(f"  '{word}' 다음에 나온 단어 (모두 {int(total)}번)")
    order = np.argsort(-row)
    for j in order:
        if row[j] == 0:
            continue
        print(f"    {vocab[j]:<8} {int(row[j]):>3}번   {row[j] / total * 100:5.1f}%")


def draw_steps(steps, words, ko: bool, out_path: Path) -> None:
    """각 단계의 후보 확률을 막대그래프로 나란히 그린다."""
    L = _krfont.label

    if not steps:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.axis("off")
        ax.text(0.5, 0.5,
                L(f"'{words[0]}' 뒤에 온 단어가 학습 문장에 없다",
                  "no word ever followed the start word", ko),
                ha="center", va="center", fontsize=13, color=GRAY)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.6))
    if n == 1:
        axes = [axes]

    # 후보 개수가 달라도 막대 굵기가 같아 보이도록 세로 범위를 하나로 맞춘다
    max_candidates = max(len(cands) for _p, cands, _w, _q in steps)

    for i, (ax, (prev, candidates, picked, _picked_p)) in enumerate(zip(axes, steps)):
        labels = [word for word, _ in candidates]
        values = [p for _, p in candidates]
        if not ko:
            labels = [f"w{k + 1}" for k in range(len(labels))]

        y = (max_candidates - 1) - np.arange(len(labels))
        colors = [ORANGE if word == picked else BLUE for word, _ in candidates]
        ax.barh(y, values, color=colors, height=0.55)
        ax.set_ylim(-0.7, max_candidates - 0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10.5)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0%", "50%", "100%"], fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            L(f"{i + 1}단계   '{prev}' 다음",
              f"step {i + 1}: after word {i + 1}", ko),
            fontsize=11.5, pad=8,
        )
        for yy, value in zip(y, values):
            ax.text(value + 0.03, yy, f"{value * 100:.0f}%",
                    va="center", fontsize=9.5, color="#333333")

    made = " ".join(words)
    fig.suptitle(
        L(f"다음 단어 확률과 실제로 뽑힌 단어(주황) — 만들어진 문장: {made}",
          "Next-word probabilities; the sampled word is orange", ko),
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
    vocab, counts, probs, corpus = _tinylm.build_model()
    truths = _tinylm.unique_sentences()

    print("=== 실습 11.1  다음 단어를 예측하는 아주 작은 언어모델 ===")
    print("데이터: 코드 안에 적어 둔 합성 한국어 문장(실제 기록 아님). 외부 API를 쓰지 않는다.")
    print(f"서로 다른 학습 문장: {len(truths)}종")
    print(f"횟수만큼 펼친 학습 문장: {len(corpus)}개")
    print(f"단어 수(어휘): {len(vocab)}개")
    print(f"설정: START_WORD={START_WORD}, N_WORDS={N_WORDS}, SEED={SEED}")
    print()

    if START_WORD not in vocab:
        print(f"'{START_WORD}'는 학습 문장에 없는 단어다. 아래 단어 중에서 고른다.")
        print("  " + ", ".join(vocab))
        raise SystemExit(1)

    print("[1] 빈도표 — 앞 단어 뒤에 어떤 단어가 몇 번 나왔는가")
    print_count_table(vocab, counts, START_WORD)
    print()
    print_count_table(vocab, counts, "월요일에")
    print()

    print("[2] 빈도를 확률로 바꾼다 (각 줄의 합이 1이 되도록 나눈다)")
    print(f"  '{START_WORD}' 다음에 올 단어의 확률")
    for word, p in _tinylm.next_word_table(vocab, probs, START_WORD):
        print(f"    {word:<8} {p * 100:5.1f}%")
    print()

    print("[3] 확률대로 단어를 이어 붙여 문장을 만든다")
    words, steps = generate_with_steps(vocab, probs, START_WORD, N_WORDS, rng)
    for i, (prev, _cands, picked, picked_p) in enumerate(steps, start=1):
        print(f"  {i}단계  '{prev}' 다음 -> '{picked}' 선택 (확률 {picked_p * 100:.1f}%)")
    made = " ".join(words)
    print(f"  만들어진 문장: {made}")
    print(f"  학습 문장에 그대로 있는 문장인가: {'있다' if made in truths else '없다'}")
    print()

    print(f"[4] 같은 방법으로 문장을 {N_SENTENCES}개 더 만든다")
    rows = []
    same_count = 0
    for k in range(1, N_SENTENCES + 1):
        w, _s = generate_with_steps(vocab, probs, START_WORD, N_WORDS, rng)
        sentence = " ".join(w)
        in_train = sentence in truths
        same_count += int(in_train)
        rows.append((k, sentence, "있다" if in_train else "없다"))
        mark = "학습 문장 그대로" if in_train else "학습 문장에 없던 새 조합"
        print(f"  {k}. {sentence}   [{mark}]")
    print(f"  {N_SENTENCES}개 중 학습 문장 그대로인 것: {same_count}개")
    print()

    print("[5] 이 모델은 뜻을 모르고 빈도만 안다")
    p_lab = probs[vocab.index("실습실에서"), vocab.index("파이썬을")]
    p_lib = probs[vocab.index("도서관에서"), vocab.index("파이썬을")]
    print(f"  P(파이썬을 | 실습실에서) = {p_lab * 100:.1f}%")
    print(f"  P(파이썬을 | 도서관에서) = {p_lib * 100:.1f}%")
    print("  두 값이 다른 이유는 '실습실에 컴퓨터가 있어서'가 아니다.")
    print("  학습 문장에서 그 두 단어가 붙어 나온 횟수가 다르기 때문이다.")
    print("  단어를 아무 뜻 없는 기호로 바꿔도 이 모델의 계산은 한 글자도 달라지지 않는다.")
    print()

    fig_path = RESULTS_DIR / "next_word_probs.png"
    draw_steps(steps, words, ko, fig_path)
    print(f"저장: results/{fig_path.name}")

    csv_path = RESULTS_DIR / "next_word_table.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["앞단어", "다음단어", "횟수", "확률"])
        for i, prev in enumerate(vocab):
            total = counts[i].sum()
            if total == 0:
                continue
            for j in np.argsort(-counts[i]):
                if counts[i, j] == 0:
                    continue
                writer.writerow([prev, vocab[j], int(counts[i, j]),
                                 f"{probs[i, j]:.3f}"])
    print(f"저장: results/{csv_path.name}")

    made_csv = RESULTS_DIR / "next_word_sentences.csv"
    with made_csv.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["번호", "만들어진 문장", "학습 문장에 있는가"])
        writer.writerows(rows)
    print(f"저장: results/{made_csv.name}")


if __name__ == "__main__":
    main()
