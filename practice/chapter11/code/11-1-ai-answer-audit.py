"""11주차 실습: n-gram 언어모델과 AI 답변 검증표

이 코드는 두 가지를 한다.

Part 1 — 간단한 n-gram 언어모델
  작은 한국어 말뭉치에서 bigram 빈도를 세고,
  시작 단어를 주면 다음 단어를 확률 기반으로 예측하여 문장을 생성한다.
  "다음 토큰 예측"이 언어모델의 핵심 동작임을 보여준다.

Part 2 — AI 답변 검증표
  미리 준비한 AI 답변 예시(환각 포함)를 읽고
  검증 항목별로 점검표를 출력하고 CSV로 저장한다.

CPU 즉시 실행. 외부 의존성 없이 numpy만 사용한다.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

# ──────────────────────────────────────────────
# Part 1: 간단한 bigram 언어모델
# ──────────────────────────────────────────────

# 수업용 작은 한국어 말뭉치
CORPUS = [
    "딥러닝은 데이터에서 패턴을 찾는 방법이다",
    "딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습한다",
    "딥러닝은 이미지 인식에 많이 사용된다",
    "모델은 데이터에서 규칙을 배운다",
    "모델은 훈련 데이터로 학습하고 테스트 데이터로 평가한다",
    "모델은 항상 정확하지 않다",
    "인공지능은 사람의 판단을 완전히 대신하지 못한다",
    "인공지능은 데이터에서 패턴을 찾아 예측한다",
    "데이터가 편향되면 모델도 편향된 결과를 낸다",
    "데이터가 부족하면 모델은 새로운 상황에 약하다",
    "학습 데이터에 없는 내용을 모델이 지어내는 것을 환각이라 한다",
    "환각은 모델이 그럴듯하지만 사실이 아닌 답변을 만드는 현상이다",
    "프롬프트를 바꾸면 모델의 답변이 달라진다",
    "프롬프트는 모델에게 주는 입력 조건이다",
    "출처를 확인하지 않은 정보는 위험하다",
    "출처를 확인하는 습관이 중요하다",
]


def build_bigram_model(
    corpus: list[str],
) -> dict[str, Counter]:
    """말뭉치에서 bigram 빈도를 세어 언어모델을 만든다.

    반환값은 {앞_단어: Counter({뒤_단어: 빈도, ...})} 형태이다.
    """
    model: dict[str, Counter] = {}
    for sentence in corpus:
        words = sentence.split()
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            if current_word not in model:
                model[current_word] = Counter()
            model[current_word][next_word] += 1
    return model


def predict_next_words(
    bigram_model: dict[str, Counter],
    word: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """주어진 단어 뒤에 올 수 있는 다음 단어와 확률을 반환한다."""
    if word not in bigram_model:
        return []
    counter = bigram_model[word]
    total = sum(counter.values())
    predictions = [
        (next_word, round(count / total, 3))
        for next_word, count in counter.most_common(top_k)
    ]
    return predictions


def generate_sentence(
    bigram_model: dict[str, Counter],
    start_word: str,
    max_length: int = 12,
    rng: np.random.Generator | None = None,
) -> str:
    """시작 단어에서 bigram 확률에 따라 문장을 생성한다."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    words = [start_word]
    current = start_word
    for _ in range(max_length - 1):
        if current not in bigram_model:
            break
        counter = bigram_model[current]
        candidates = list(counter.keys())
        counts = np.array(list(counter.values()), dtype=float)
        probs = counts / counts.sum()
        next_word = rng.choice(candidates, p=probs)
        words.append(next_word)
        current = next_word
    return " ".join(words)


# ──────────────────────────────────────────────
# Part 2: AI 답변 검증표
# ──────────────────────────────────────────────

# 미리 준비한 AI 답변 예시 (일부는 환각, 일부는 편향, 일부는 정확)
AI_ANSWERS = [
    {
        "질문": "딥러닝의 창시자는 누구인가?",
        "AI_답변": "딥러닝은 1986년 제프리 힌턴이 역전파 알고리즘을 발명하면서 시작되었다.",
        "검증_메모": "역전파는 힌턴 혼자 발명한 것이 아니다. "
        "1986년 논문은 Rumelhart, Hinton, Williams 공저이며, "
        "역전파 아이디어는 그 이전에도 여러 연구자가 제안했다. "
        "힌턴을 유일한 창시자로 단정하는 것은 부정확하다.",
        "검증_등급": "불확실",
        "문제_유형": "과장/단순화",
    },
    {
        "질문": "CNN은 어떤 문제에 가장 효과적인가?",
        "AI_답변": "CNN은 이미지 분류에서 항상 99% 이상의 정확도를 달성하며, "
        "모든 이미지 관련 문제에서 최고의 성능을 보인다.",
        "검증_메모": "'항상 99%'는 과장이다. 데이터셋, 클래스 수, "
        "이미지 품질에 따라 정확도는 크게 달라진다. "
        "또한 최근에는 Vision Transformer 등 CNN이 아닌 "
        "구조가 일부 벤치마크에서 더 높은 성능을 보인다.",
        "검증_등급": "틀림",
        "문제_유형": "환각/과장",
    },
    {
        "질문": "학습률이 너무 크면 어떻게 되는가?",
        "AI_답변": "학습률이 너무 크면 손실 함수가 수렴하지 않고 발산할 수 있다. "
        "가중치가 너무 크게 변하면서 최적점을 지나칠 수 있기 때문이다.",
        "검증_메모": "이 설명은 기본 원리에 부합한다. "
        "6주차 실습에서 학습률을 크게 설정하면 "
        "손실이 발산하는 것을 직접 확인했다.",
        "검증_등급": "확인됨",
        "문제_유형": "없음",
    },
    {
        "질문": "GPT-4의 파라미터 수는 몇 개인가?",
        "AI_답변": "GPT-4는 약 1.8조 개의 파라미터를 가지고 있으며, "
        "8개의 전문가 모델을 혼합한 MoE 구조를 사용한다.",
        "검증_메모": "OpenAI는 GPT-4의 정확한 파라미터 수와 "
        "내부 구조를 공식 발표하지 않았다. "
        "1.8조, MoE 구조 등은 비공식 유출 정보에 기반한 추측이다. "
        "공식 출처로 확인할 수 없으므로 사실로 단정할 수 없다.",
        "검증_등급": "판단 보류",
        "문제_유형": "출처 미확인",
    },
    {
        "질문": "AI는 편향 없이 공정한 판단을 하는가?",
        "AI_답변": "AI는 데이터에서 학습하므로, 학습 데이터에 편향이 있으면 "
        "AI의 판단도 편향된다. 예를 들어 채용 심사에 사용된 AI가 "
        "과거 채용 데이터에서 특정 성별을 선호하는 패턴을 학습한 사례가 있다.",
        "검증_메모": "이 설명은 편향의 원리를 올바르게 설명한다. "
        "아마존의 채용 AI 사례는 2018년에 실제로 보도되었다. "
        "다만 '아마존'이라는 구체적 이름은 이 답변에 나오지 않았고, "
        "출처를 더 명확히 할 수 있다.",
        "검증_등급": "확인됨",
        "문제_유형": "없음",
    },
]


def build_audit_table(
    answers: list[dict],
) -> list[dict]:
    """AI 답변 목록을 받아 검증표를 만든다."""
    audit_rows = []
    for idx, item in enumerate(answers, start=1):
        audit_rows.append(
            {
                "번호": idx,
                "질문": item["질문"],
                "AI_답변_요약": item["AI_답변"][:50] + "..."
                if len(item["AI_답변"]) > 50
                else item["AI_답변"],
                "검증_등급": item["검증_등급"],
                "문제_유형": item["문제_유형"],
                "검증_메모": item["검증_메모"],
            }
        )
    return audit_rows


def print_audit_table(audit_rows: list[dict]) -> None:
    """검증표를 보기 좋게 출력한다."""
    print(f"{'번호':>4}  {'검증_등급':<8}  {'문제_유형':<12}  {'질문'}")
    print("-" * 72)
    for row in audit_rows:
        print(
            f"{row['번호']:>4}  {row['검증_등급']:<8}  "
            f"{row['문제_유형']:<12}  {row['질문']}"
        )


def save_audit_csv(audit_rows: list[dict], output_path: Path) -> None:
    """검증표를 CSV로 저장한다. pandas 없이 직접 작성한다."""
    headers = ["번호", "질문", "AI_답변_요약", "검증_등급", "문제_유형", "검증_메모"]
    lines = [",".join(headers)]
    for row in audit_rows:
        values = []
        for h in headers:
            cell = str(row[h]).replace('"', '""')
            values.append(f'"{cell}"')
        lines.append(",".join(values))
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Part 1: n-gram 언어모델 ──
    print("=" * 60)
    print("Part 1: 간단한 bigram 언어모델")
    print("=" * 60)
    print()

    bigram_model = build_bigram_model(CORPUS)
    vocab_size = len(bigram_model)
    total_bigrams = sum(sum(c.values()) for c in bigram_model.values())

    print(f"말뭉치 문장 수: {len(CORPUS)}")
    print(f"bigram 모델의 고유 앞단어 수: {vocab_size}")
    print(f"bigram 총 빈도: {total_bigrams}")
    print()

    # 다음 단어 예측 시연
    test_words = ["딥러닝은", "모델은", "데이터가", "프롬프트를"]
    print("--- 다음 단어 예측 ---")
    for word in test_words:
        predictions = predict_next_words(bigram_model, word, top_k=5)
        if predictions:
            pred_str = ", ".join(
                f"{w}({p:.1%})" for w, p in predictions
            )
            print(f"  '{word}' 다음 → {pred_str}")
        else:
            print(f"  '{word}' 다음 → (말뭉치에 없는 단어)")
    print()

    # 문장 생성 시연: 같은 시작 단어로 시드를 바꿔 여러 문장 생성
    print("--- 문장 생성 (시작 단어: '딥러닝은') ---")
    seeds_for_generation = [42, 123, 7, 2024, 99]
    for seed_val in seeds_for_generation:
        rng = np.random.default_rng(seed_val)
        sentence = generate_sentence(bigram_model, "딥러닝은", max_length=10, rng=rng)
        print(f"  [seed={seed_val:>4}] {sentence}")
    print()

    print("--- 문장 생성 (시작 단어: '모델은') ---")
    for seed_val in seeds_for_generation:
        rng = np.random.default_rng(seed_val)
        sentence = generate_sentence(bigram_model, "모델은", max_length=10, rng=rng)
        print(f"  [seed={seed_val:>4}] {sentence}")
    print()

    print("--- 핵심 관찰 ---")
    print("1. 같은 시작 단어라도 무작위 선택(시드)에 따라 다른 문장이 나온다.")
    print("2. bigram 모델은 바로 앞 한 단어만 보므로 문맥을 길게 기억하지 못한다.")
    print("3. 말뭉치가 작으면 선택지가 적어 비슷한 문장이 반복된다.")
    print("4. 실제 대형 언어모델도 원리는 같다: 앞 토큰들을 보고 다음 토큰을 예측한다.")
    print("   다만 볼 수 있는 토큰 수(컨텍스트)가 훨씬 많고 파라미터가 크다.")
    print()

    # ── Part 2: AI 답변 검증표 ──
    print("=" * 60)
    print("Part 2: AI 답변 검증표")
    print("=" * 60)
    print()

    audit_rows = build_audit_table(AI_ANSWERS)

    print("--- 검증표 요약 ---")
    print_audit_table(audit_rows)
    print()

    # 등급별 통계
    grade_counts: dict[str, int] = {}
    for row in audit_rows:
        grade = row["검증_등급"]
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    print("--- 검증 등급별 빈도 ---")
    for grade, count in sorted(grade_counts.items()):
        print(f"  {grade}: {count}건")
    print()

    # 상세 출력
    print("--- 검증 상세 ---")
    for item in AI_ANSWERS:
        print(f"\n질문: {item['질문']}")
        print(f"AI 답변: {item['AI_답변']}")
        print(f"검증 등급: {item['검증_등급']}")
        print(f"문제 유형: {item['문제_유형']}")
        print(f"검증 메모: {item['검증_메모']}")
        print("-" * 40)

    # CSV 저장
    csv_path = RESULTS_DIR / "ai_answer_audit.csv"
    save_audit_csv(audit_rows, csv_path)
    print()

    print("=" * 60)
    print("저장된 결과")
    print("=" * 60)
    print(f"검증표 CSV: {csv_path.relative_to(CHAPTER_DIR)}")
    print()
    print("=" * 60)
    print("해석 주의")
    print("=" * 60)
    print("1. 언어모델은 '다음에 올 가능성이 높은 단어'를 고르는 것이지,")
    print("   '사실인 단어'를 고르는 것이 아니다.")
    print("2. 따라서 그럴듯하지만 사실이 아닌 답변(환각)이 발생할 수 있다.")
    print("3. AI 답변을 받으면 반드시 검증 등급을 매기고 출처를 확인해야 한다.")


if __name__ == "__main__":
    main()
