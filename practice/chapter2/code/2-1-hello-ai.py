from __future__ import annotations

import platform
import sys


def make_intro(name: str, interest: str, goal: str) -> str:
    """학생의 첫 AI 코딩 실습 문장을 만든다."""
    clean_name = name.strip() or "이름 없는 학생"
    clean_interest = interest.strip() or "AI"
    clean_goal = goal.strip() or "오늘 만든 코드를 직접 실행해보기"
    return (
        f"{clean_name} 학생은 {clean_interest}에 관심이 있으며, "
        f"오늘의 목표는 '{clean_goal}'이다."
    )


def main() -> None:
    # 수업 중에는 input()으로 바꾸어 직접 입력해도 된다.
    name = "민지"
    interest = "이미지 분류"
    goal = "AI가 만든 코드를 실행하고 검증하기"

    print("=== 2주차 첫 Python 실행 ===")
    print(make_intro(name, interest, goal))
    print()
    print("=== 실행 환경 ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print()
    print("=== 안전 점검 ===")
    print("- 비밀번호, API 키, 개인정보를 코드에 넣지 않았다.")
    print("- AI가 만든 코드라도 직접 실행해서 결과를 확인해야 한다.")


if __name__ == "__main__":
    main()

