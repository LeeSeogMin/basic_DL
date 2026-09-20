"""실습 3.2 준비 - 흠집 난 표를 만들어 저장한다.

이 스크립트는 흠집 난 표를 만들기만 한다. 찾고 고치는 일은 학생이
AI에게 요구사항을 전달해 받은 코드로 한다(교재 실습 3.2).

흠집의 종류·건수·위치는 참고 구현(3-2-data-quality.py)의 상수와
생성 함수를 그대로 가져다 쓴다. 그래서 두 파일이 만드는 표는 같고,
참고 구현의 로그를 정답지로 쓸 수 있다.

데이터: 컴퓨터로 만든 합성 데이터다. 실제 설문 응답이 아니고,
        실명·학번·연락처가 들어갈 자리가 없다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_reference():
    """참고 구현(3-2-data-quality.py)을 모듈로 읽어 온다.

    파일 이름이 숫자로 시작해 일반 import 문으로는 불러올 수 없어서
    importlib를 쓴다. 생성 함수와 상수를 한 곳에만 두기 위해서다.
    """
    path = Path(__file__).with_name("3-2-data-quality.py")
    spec = importlib.util.spec_from_file_location("data_quality_ref", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ref = load_reference()
    ref.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 참고 구현과 같은 seed, 같은 순서로 난수를 쓰므로 같은 표가 나온다.
    rng = np.random.default_rng(ref.RANDOM_SEED)
    truth = ref.make_clean_table(rng)
    dirty = ref.make_dirty_table(truth, rng)

    out_path = ref.RESULTS_DIR / "week03-dirty.csv"
    dirty.to_csv(out_path, index=False, encoding="utf-8-sig")
    loaded = pd.read_csv(out_path)

    print("=== 실습 3.2 준비. 흠집 난 표 만들기 ===")
    print("데이터: 컴퓨터로 만든 합성 데이터(실제 설문 응답 아님)")
    print(f"원본 {ref.N_STUDENTS}행에 낸 흠집:")
    print(f"  중복 행 {ref.N_DUPLICATE_ROWS}건 / 문자열 점수 {ref.N_TEXT_SCORE}건 / "
          f"수면시간 이상치 {ref.N_OUTLIER_SLEEP}건 / "
          f"공부시간 결측 {ref.N_MISSING_STUDY}건 / 시험점수 결측 {ref.N_MISSING_SCORE}건")
    print(f"저장: results/{out_path.name}")
    print(f"저장한 표: {loaded.shape[0]}행 x {loaded.shape[1]}열")
    print()
    print("표 앞부분 8행")
    print(loaded.head(8).to_string(index=False))
    print()
    print("어느 행, 어느 칸에 흠집이 있는지는 여기에 적지 않는다.")
    print("찾는 일부터가 실습이다. 교재의 검증 체크리스트와 대조하며 진행한다.")


if __name__ == "__main__":
    main()
