"""그림에 한글 라벨을 쓰기 위한 폰트 설정 도우미.

이 파일은 이름이 밑줄로 시작하므로 실행 증거 게이트(run_and_capture.py)가
직접 실행하지 않는다. 다른 실습 코드에서 import 해서 쓴다.

환경에 한글 폰트가 없으면 영어 라벨로 자동 전환한다. 그래서 어떤 운영체제에서
실행해도 그림의 글자가 네모(□□□)로 깨지지 않는다.
"""

from __future__ import annotations

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


# 운영체제별로 흔히 설치된 한글 폰트 후보
KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",      # Windows
    "AppleGothic",        # macOS
    "NanumGothic",        # Linux(나눔글꼴)
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
]


def setup() -> bool:
    """한글 폰트를 찾아 matplotlib에 설정한다.

    Returns:
        찾았으면 True, 못 찾았으면 False. False면 라벨을 영어로 써야 한다.
    """
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in KOREAN_FONT_CANDIDATES:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True

    matplotlib.rcParams["axes.unicode_minus"] = False
    return False


def label(korean: str, english: str, korean_ok: bool) -> str:
    """한글 폰트가 있으면 한글 라벨을, 없으면 영어 라벨을 돌려준다."""
    return korean if korean_ok else english


def report(korean_ok: bool) -> None:
    """어떤 라벨로 그렸는지 로그에 남긴다."""
    if korean_ok:
        print(f"[그림 폰트] 한글 폰트 사용: {plt.rcParams['font.family']}")
    else:
        print("[그림 폰트] 한글 폰트를 찾지 못해 영어 라벨로 그립니다.")
