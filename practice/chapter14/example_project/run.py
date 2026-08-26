"""예시 프로젝트 - 단어 목록으로 리뷰를 긍정/부정으로 나눈다.

실습 14.2의 점검 도구가 읽는 예시 파일이다. 표준 라이브러리만 쓴다.
데이터는 직접 지어낸 문장 12개이며 실제 리뷰가 아니다.
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
POSITIVE_WORDS = ["좋다", "만족", "빠르다", "친절", "추천"]
NEGATIVE_WORDS = ["별로", "느리다", "불친절", "환불", "실망"]


def predict(text: str) -> str:
    """긍정 단어와 부정 단어를 세어 더 많은 쪽으로 판정한다."""
    plus = sum(word in text for word in POSITIVE_WORDS)
    minus = sum(word in text for word in NEGATIVE_WORDS)
    if plus > minus:
        return "긍정"
    if minus > plus:
        return "부정"
    return "중립"


def main() -> None:
    rows = list(csv.DictReader((HERE / "data" / "input_example.csv").open(encoding="utf-8")))
    hit = sum(predict(row["문장"]) == row["정답"] for row in rows)
    print(f"문장 {len(rows)}개 중 {hit}개를 맞혔다. 정확도 {hit / len(rows):.3f}")


if __name__ == "__main__":
    main()
