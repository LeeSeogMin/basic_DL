"""실습 12.2의 검사 대상 ① - 위험 신호가 없는 예제 코드.

이 파일은 "AI에게 시켜서 받았다고 가정한" 합성 예제다. 실제 프로젝트 코드가 아니다.
실습 12.2는 이 파일을 **텍스트로 읽기만 한다.** 실행하지 않는다.

파일 이름이 밑줄로 시작하므로 실행 증거 게이트(run_and_capture.py)도 이 파일을
직접 실행하지 않는다.

하는 일: 점수가 든 CSV를 읽어 평균과 최고점을 구하고, 요약을 CSV 한 장으로 저장한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_CSV = Path("sample_folder") / "data" / "scores.csv"
OUTPUT_CSV = Path("results") / "score_summary.csv"


def load_scores(csv_path: Path) -> pd.DataFrame:
    """점수 표를 읽는다. 파일이 없으면 이유를 알리고 멈춘다."""
    if not csv_path.exists():
        raise FileNotFoundError(f"점수 파일을 찾을 수 없습니다: {csv_path}")
    return pd.read_csv(csv_path)


def summarize(table: pd.DataFrame, column: str = "score") -> pd.DataFrame:
    """평균·최고점·인원을 한 줄짜리 표로 만든다."""
    return pd.DataFrame([{
        "인원": len(table),
        "평균": round(float(table[column].mean()), 2),
        "최고점": float(table[column].max()),
    }])


def main() -> None:
    table = load_scores(INPUT_CSV)
    summary = summarize(table)
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
