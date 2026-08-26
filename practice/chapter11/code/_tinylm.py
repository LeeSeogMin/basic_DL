"""11장 실습 두 개가 함께 쓰는 아주 작은 언어모델과 학습 문장.

이 파일은 이름이 밑줄로 시작하므로 실행 증거 게이트(run_and_capture.py)가
직접 실행하지 않는다. 11-1, 11-2 실습 코드에서 import 해서 쓴다.

여기 들어 있는 문장은 전부 **합성 한국어 문장**이다. 실제 사람도, 실제 기록도
아니다. 인터넷이나 외부 API를 전혀 쓰지 않는다.

이 작은 세계에서 "사실"은 아래 SENTENCES_WITH_COUNT 목록에 적힌 문장뿐이다.
목록에 없는 문장은 어법이 맞아도 이 세계에서는 틀린 문장이다.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 학습 문장 (합성 데이터)
#
# (문장, 자료에 나온 횟수) 형태로 적는다. 실제 글뭉치에서는 어떤 표현이 다른
# 표현보다 훨씬 자주 나온다. 그 치우침을 흉내 내려고 횟수를 다르게 두었다.
# 문장은 모두 "누가 / 언제 / 어디서 / 무엇을 / 어떻게 한다" 다섯 어절이다.
# ---------------------------------------------------------------------------
SENTENCES_WITH_COUNT: list[tuple[str, int]] = [
    ("민준이는 월요일에 도서관에서 통계학을 공부한다", 5),
    ("민준이는 화요일에 실습실에서 파이썬을 연습한다", 2),
    ("민준이는 수요일에 강의실에서 영어를 복습한다", 1),
    ("서연이는 월요일에 도서관에서 미적분을 공부한다", 3),
    ("서연이는 화요일에 실습실에서 파이썬을 공부한다", 2),
    ("서연이는 목요일에 동아리방에서 발표자료를 정리한다", 2),
    ("지호는 수요일에 강의실에서 영어를 복습한다", 3),
    ("지호는 월요일에 실습실에서 파이썬을 연습한다", 2),
    ("지호는 목요일에 도서관에서 통계학을 정리한다", 1),
    ("하윤이는 목요일에 동아리방에서 발표자료를 정리한다", 3),
    ("하윤이는 화요일에 도서관에서 미적분을 복습한다", 2),
    ("하윤이는 수요일에 실습실에서 영어를 연습한다", 1),
]


def load_sentences() -> list[list[str]]:
    """학습 문장을 어절 목록으로 펼친다. 횟수만큼 되풀이해 넣는다."""
    corpus: list[list[str]] = []
    for sentence, count in SENTENCES_WITH_COUNT:
        for _ in range(count):
            corpus.append(sentence.split())
    return corpus


def unique_sentences() -> set[str]:
    """서로 다른 학습 문장의 집합. 이 세계에서 참인 문장 목록이다."""
    return {sentence for sentence, _ in SENTENCES_WITH_COUNT}


# ---------------------------------------------------------------------------
# 아주 작은 언어모델: 앞 단어 하나만 보고 다음 단어를 고른다
# ---------------------------------------------------------------------------
def count_pairs(corpus: list[list[str]]) -> tuple[list[str], np.ndarray]:
    """"앞 단어 -> 다음 단어"가 몇 번 나왔는지 세어 빈도표를 만든다.

    Returns:
        (단어 목록, 빈도표). 빈도표[i, j]는 i번 단어 바로 뒤에 j번 단어가
        나온 횟수다.
    """
    vocab = sorted({word for sentence in corpus for word in sentence})
    index = {word: i for i, word in enumerate(vocab)}

    counts = np.zeros((len(vocab), len(vocab)), dtype=float)
    for sentence in corpus:
        for prev, nxt in zip(sentence[:-1], sentence[1:]):
            counts[index[prev], index[nxt]] += 1.0
    return vocab, counts


def to_probability(counts: np.ndarray) -> np.ndarray:
    """빈도표를 확률표로 바꾼다. 각 줄의 합을 1로 맞춘다.

    뒤에 아무 단어도 온 적이 없는 줄(문장 끝 단어)은 0으로 남겨 둔다.
    """
    row_sum = counts.sum(axis=1, keepdims=True)
    probs = np.zeros_like(counts)
    nonzero = row_sum[:, 0] > 0
    probs[nonzero] = counts[nonzero] / row_sum[nonzero]
    return probs


def apply_temperature(prob_row: np.ndarray, temperature: float) -> np.ndarray:
    """확률 한 줄에 온도를 적용한다.

    온도가 낮으면(<1) 큰 값이 더 커져 분포가 뾰족해지고,
    온도가 높으면(>1) 값들이 서로 가까워져 분포가 평평해진다.
    계산은 p^(1/T) 를 다시 합이 1이 되도록 나누는 것이다.
    """
    if temperature <= 0:
        raise ValueError("온도는 0보다 커야 합니다.")

    mask = prob_row > 0
    if not mask.any():
        return np.zeros_like(prob_row)

    scaled = np.zeros_like(prob_row)
    scaled[mask] = prob_row[mask] ** (1.0 / temperature)
    return scaled / scaled.sum()


def next_word_table(
    vocab: list[str],
    probs: np.ndarray,
    word: str,
    temperature: float = 1.0,
) -> list[tuple[str, float]]:
    """어떤 단어 뒤에 올 수 있는 단어와 그 확률을 확률 높은 순으로 돌려준다."""
    if word not in vocab:
        raise KeyError(f"학습 문장에 없는 단어입니다: {word}")

    row = apply_temperature(probs[vocab.index(word)], temperature)
    pairs = [(vocab[j], float(row[j])) for j in np.nonzero(row)[0]]
    return sorted(pairs, key=lambda item: item[1], reverse=True)


def generate(
    vocab: list[str],
    probs: np.ndarray,
    start_word: str,
    n_words: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> tuple[list[str], list[float]]:
    """시작 단어에서 출발해 확률대로 단어를 이어 붙인다.

    Returns:
        (만들어진 단어 목록, 각 단계에서 뽑힌 단어의 확률).
        첫 단어는 사람이 정했으므로 확률을 1.0으로 적는다.
    """
    if start_word not in vocab:
        raise KeyError(f"학습 문장에 없는 단어입니다: {start_word}")

    words = [start_word]
    picked_probs = [1.0]

    for _ in range(n_words - 1):
        row = apply_temperature(probs[vocab.index(words[-1])], temperature)
        if row.sum() == 0:
            break  # 이 단어 뒤에 온 단어가 학습 문장에 없다. 문장을 끝낸다.
        j = int(rng.choice(len(vocab), p=row))
        words.append(vocab[j])
        picked_probs.append(float(row[j]))

    return words, picked_probs


def match_length(words: list[str], truths: set[str]) -> int:
    """만든 문장이 학습 문장과 앞에서부터 몇 어절까지 같은지 센다.

    다섯 어절을 다 만들고 5가 나오면 학습 문장을 그대로 되풀이한 것이다.
    3이 나오면 네 번째 어절에서 학습 문장에서 갈라져 나온 것이다.
    """
    truth_words = [sentence.split() for sentence in truths]
    best = 0
    for k in range(1, len(words) + 1):
        prefix = words[:k]
        if any(truth[:k] == prefix for truth in truth_words):
            best = k
        else:
            break
    return best


def build_model() -> tuple[list[str], np.ndarray, np.ndarray, list[list[str]]]:
    """학습 문장을 읽어 빈도표와 확률표를 한 번에 만든다."""
    corpus = load_sentences()
    vocab, counts = count_pairs(corpus)
    probs = to_probability(counts)
    return vocab, counts, probs, corpus
