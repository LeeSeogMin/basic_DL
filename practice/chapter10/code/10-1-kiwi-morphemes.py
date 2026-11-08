"""실습 10.1 - 형태소 분석기 Kiwi를 설치하고 한국어 문장을 쪼갠다.

세 단계로 진행한다.
  1단계  Kiwi가 설치되어 있는지 확인한다
  2단계  같은 문장을 공백으로 자른 결과와 Kiwi로 쪼갠 결과를 나란히 본다
  3단계  조사만 다른 문장들을 Kiwi로 쪼개, 사전에 들어가는 단어 수가 어떻게 줄어드는지 센다

Kiwi(kiwipiepy)는 한국어 형태소 분석기다. 설치 명령은 다음 한 줄이다.
    python -m pip install kiwipiepy
처음 실행할 때 모델 파일을 불러오느라 몇 초 걸린다. 인터넷 연결은 설치할 때만 필요하다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 학생이 바꾸는 곳 1: 쪼개 볼 문장. 자기 문장을 넣고 다시 실행한다.
# ---------------------------------------------------------------------------
MY_SENTENCES = [
    "배송이 정말 빨라요",
    "삼성전자가 새 스마트폰을 공개했다",
    "손흥민, 시즌 10호 골로 팀 승리 이끌어",
]

# ---------------------------------------------------------------------------
# 학생이 바꾸는 곳 2: 조사·어미만 다른 문장 묶음.
# 공백으로 자르면 서로 다른 단어가 되고, Kiwi로 쪼개면 같은 단어로 모이는지 본다.
# ---------------------------------------------------------------------------
SAME_WORD_SENTENCES = [
    "배송이 빨라요",
    "배송은 느렸어요",
    "배송도 하루 만에 왔어요",
    "화면이 선명해요",
    "화면은 어두워요",
    "화면도 커요",
]

# 뜻을 담은 형태소로 남길 품사: 일반명사, 고유명사, 동사, 형용사, 어근, 외국어, 한자
# 불규칙 활용 용언은 "VA-I"처럼 뒤에 표지가 붙으므로 앞부분만 비교한다.
KEEP_TAGS = ("NNG", "NNP", "VV", "VA", "XR", "SL", "SH")


def is_content(tag: str) -> bool:
    """뜻을 담은 형태소의 품사인지 본다."""
    return tag.split("-")[0] in KEEP_TAGS

# 자주 나오는 품사 표지의 뜻
TAG_NAMES = {
    "NNG": "일반명사", "NNP": "고유명사", "NNB": "의존명사", "NR": "수사", "NP": "대명사",
    "VV": "동사", "VA": "형용사", "VX": "보조용언", "VCP": "긍정지정사(이다)",
    "MAG": "부사", "MM": "관형사", "XR": "어근", "XSA": "형용사파생접미사",
    "JKS": "주격조사", "JKO": "목적격조사", "JKB": "부사격조사", "JX": "보조사",
    "EP": "선어말어미", "EF": "종결어미", "EC": "연결어미", "ETM": "관형형어미",
    "XSN": "명사파생접미사", "XSV": "동사파생접미사",
    "SN": "숫자", "SL": "외국어", "SH": "한자", "SP": "쉼표 등", "SF": "마침표 등",
}

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


def split_by_space(text: str) -> list[str]:
    """문장부호를 공백으로 바꾸고 공백으로 자른다."""
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return [token for token in cleaned.split() if token]


def step1_check_install():
    print("=== 1단계. Kiwi 설치 확인 ===")
    try:
        import kiwipiepy
        from kiwipiepy import Kiwi
    except ImportError:
        print("kiwipiepy가 설치되어 있지 않다. 아래 명령으로 설치한 뒤 다시 실행한다.")
        print("    python -m pip install kiwipiepy")
        sys.exit(1)
    print(f"kiwipiepy 버전: {kiwipiepy.__version__}")
    kiwi = Kiwi()
    print("Kiwi 모델을 불러왔다.")
    print()
    return kiwi


def step2_compare(kiwi) -> None:
    print("=== 2단계. 공백으로 자르기 vs Kiwi로 쪼개기 ===")
    for i, sentence in enumerate(MY_SENTENCES, start=1):
        tokens = kiwi.tokenize(sentence)
        print(f"문장{i}  {sentence}")
        print(f"  공백으로 자르기 {len(split_by_space(sentence))}개: "
              f"{' | '.join(split_by_space(sentence))}")
        print(f"  Kiwi로 쪼개기   {len(tokens)}개: "
              f"{' | '.join(f'{t.form}/{t.tag}' for t in tokens)}")
        kept = [t.form for t in tokens if is_content(t.tag)]
        print(f"  뜻을 담은 형태소만 남기기: {' | '.join(kept)}")
        print()

    used = sorted({t.tag.split("-")[0] for s in MY_SENTENCES for t in kiwi.tokenize(s)})
    print("위 결과에 나온 품사 표지의 뜻 (* 표시는 뜻을 담은 형태소로 남기는 품사)")
    for tag in used:
        mark = "*" if tag in KEEP_TAGS else " "
        print(f"  {mark} {tag:<4} {TAG_NAMES.get(tag, '(표에 없음: Kiwi 문서 참고)')}")
    print()


def step3_vocabulary(kiwi) -> None:
    print("=== 3단계. 조사만 다른 문장에서 사전 크기 비교 ===")
    rows = []
    space_vocab: set[str] = set()
    kiwi_vocab: set[str] = set()
    for sentence in SAME_WORD_SENTENCES:
        by_space = split_by_space(sentence)
        by_kiwi = [t.form for t in kiwi.tokenize(sentence) if is_content(t.tag)]
        space_vocab.update(by_space)
        kiwi_vocab.update(by_kiwi)
        rows.append({"문장": sentence,
                     "공백으로 자르기": " | ".join(by_space),
                     "Kiwi(뜻을 담은 형태소)": " | ".join(by_kiwi)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print()
    print(f"공백으로 자른 사전: {len(space_vocab)}개 -> {', '.join(sorted(space_vocab))}")
    print(f"Kiwi로 만든 사전:   {len(kiwi_vocab)}개 -> {', '.join(sorted(kiwi_vocab))}")
    print()
    for word in ("배송", "화면"):
        space_forms = sorted(w for w in space_vocab if w.startswith(word))
        print(f"'{word}'이 공백 사전에서는 {len(space_forms)}개 단어로 갈렸다: "
              f"{', '.join(space_forms)}")
        print(f"  Kiwi 사전에서는 '{word}' 하나다: {word in kiwi_vocab}")
    print()
    table.to_csv(RESULTS_DIR / "kiwi_compare.csv", index=False, encoding="utf-8-sig")
    print("비교 표: results/kiwi_compare.csv")
    print()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== 실습 10.1 형태소 분석기 Kiwi ===")
    print()
    kiwi = step1_check_install()
    step2_compare(kiwi)
    step3_vocabulary(kiwi)
    print("=== 정리 ===")
    print("공백으로 자르면 조사와 어미가 단어에 붙어 남는다.")
    print("Kiwi는 '배송이'를 '배송 + 이'로 나누므로, 조사만 다른 단어가 한 단어로 모인다.")


if __name__ == "__main__":
    main()
