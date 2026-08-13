"""12주차 실습: AI와 함께 만드는 작은 Python 도구

이미지 크기 변환 도구를 만들고, 코드에 보안 위험이 없는지 점검한다.
보안 점검은 파일 삭제 명령, 외부 전송 코드, 비밀키 하드코딩 여부를 검사한다.

이 코드는 AI 코딩 에이전트와 협업해 만든 도구의 예시다.
학생은 AI에게 요청한 부분과 직접 수정한 부분을 구분해 기록해야 한다.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


# ---------------------------------------------------------------------------
# 1부: 이미지 크기 변환 도구
# ---------------------------------------------------------------------------

def create_sample_image(path: Path, width: int = 200, height: int = 150) -> None:
    """실습용 샘플 이미지를 만든다. 합성 데이터이다."""
    rng = np.random.default_rng(RANDOM_SEED)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    img.save(path)


def resize_image(input_path: Path, output_path: Path, new_width: int, new_height: int) -> dict:
    """이미지 크기를 변환하고 결과 정보를 반환한다."""
    img = Image.open(input_path)
    original_size = img.size  # (width, height)
    resized = img.resize((new_width, new_height), Image.LANCZOS)
    resized.save(output_path)
    return {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "original_width": original_size[0],
        "original_height": original_size[1],
        "new_width": new_width,
        "new_height": new_height,
    }


# ---------------------------------------------------------------------------
# 2부: 보안 점검 도구
# ---------------------------------------------------------------------------

# 점검 대상 패턴: 파일 삭제, 외부 전송, 비밀키 하드코딩
SECURITY_PATTERNS = [
    {
        "category": "파일 삭제",
        "description": "파일이나 디렉토리를 삭제하는 코드",
        "patterns": [
            r"\bos\.remove\b",
            r"\bos\.unlink\b",
            r"\bshutil\.rmtree\b",
            r"\bPath\(.*\)\.unlink\b",
            r"\bos\.rmdir\b",
        ],
    },
    {
        "category": "외부 전송",
        "description": "외부 서버로 데이터를 보내는 코드",
        "patterns": [
            r"\brequests\.post\b",
            r"\brequests\.put\b",
            r"\burllib\.request\.urlopen\b",
            r"\bhttp\.client\b",
            r"\bsmtplib\b",
            r"\bsocket\.connect\b",
        ],
    },
    {
        "category": "비밀키 하드코딩",
        "description": "비밀번호나 API 키가 코드에 직접 쓰여 있는 경우",
        "patterns": [
            r"""(?i)(api[_-]?key|secret|password|token|credential)\s*=\s*["'][^"']{8,}["']""",
            r"""(?i)(aws_access_key|aws_secret)\s*=\s*["']""",
        ],
    },
]


def check_security(source_code: str, filename: str) -> list[dict]:
    """소스 코드에서 보안 위험 패턴을 검사한다."""
    findings: list[dict] = []
    lines = source_code.splitlines()

    for rule in SECURITY_PATTERNS:
        for pattern in rule["patterns"]:
            for line_num, line in enumerate(lines, start=1):
                if re.search(pattern, line):
                    findings.append({
                        "file": filename,
                        "line": line_num,
                        "category": rule["category"],
                        "description": rule["description"],
                        "matched_code": line.strip(),
                    })

    return findings


# ---------------------------------------------------------------------------
# 3부: 점검 대상 예제 코드 (일부러 위험 패턴을 포함한 가상 코드)
# ---------------------------------------------------------------------------

EXAMPLE_RISKY_CODE = textwrap.dedent("""\
    # 이 코드는 보안 점검 실습을 위해 일부러 위험 패턴을 넣은 가상 코드다.
    # 실제로 실행하면 안 되는 코드이다.

    import os
    import requests
    import shutil

    API_KEY = "sk-fake-1234567890abcdef"
    PASSWORD = "my_super_secret_password_123"

    def delete_old_files(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

    def send_data_to_server(data):
        requests.post("https://example.com/collect", json=data)

    def cleanup_temp():
        shutil.rmtree("/tmp/my_temp_folder")

    # 이 코드를 실행하지 않는다. 보안 점검 대상으로만 사용한다.
""")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 12주차 AI 코딩 에이전트와 책임 있는 자동화 ===")
    print()

    # --- 1부: 이미지 크기 변환 ---
    print("--- 1부: 이미지 크기 변환 도구 ---")
    sample_path = RESULTS_DIR / "sample_image.png"
    create_sample_image(sample_path, width=200, height=150)
    print(f"샘플 이미지 생성: {sample_path.name} (200x150, 합성 데이터)")

    resize_configs = [
        (100, 75),
        (50, 50),
        (300, 200),
    ]

    resize_results = []
    for new_w, new_h in resize_configs:
        output_name = f"resized_{new_w}x{new_h}.png"
        output_path = RESULTS_DIR / output_name
        info = resize_image(sample_path, output_path, new_w, new_h)
        resize_results.append(info)
        print(f"  크기 변환: {info['original_width']}x{info['original_height']} -> {new_w}x{new_h} -> {output_name}")

    print()

    # --- 2부: 보안 점검 ---
    print("--- 2부: 보안 점검 ---")
    print("점검 대상: 위험 패턴을 일부러 넣은 가상 코드 (실제 실행하지 않는 코드)")
    print()

    findings = check_security(EXAMPLE_RISKY_CODE, "example_risky_code.py")

    if not findings:
        print("보안 위험이 발견되지 않았습니다.")
    else:
        print(f"발견된 보안 위험: {len(findings)}건")
        print()
        for i, f in enumerate(findings, start=1):
            print(f"  [{i}] {f['category']}")
            print(f"      파일: {f['file']}, 줄: {f['line']}")
            print(f"      설명: {f['description']}")
            print(f"      코드: {f['matched_code']}")
            print()

    # 보안 점검 결과를 CSV로 저장
    report_path = RESULTS_DIR / "security_check_report.csv"
    if findings:
        report_df = pd.DataFrame(findings)
        report_df.to_csv(report_path, index=False)
        print(f"보안 점검 결과 저장: {report_path.name}")
    else:
        print("보안 위험이 없어 CSV를 생성하지 않았습니다.")

    # --- 3부: 이 실습 코드 자체를 점검 ---
    print()
    print("--- 3부: 이 실습 코드 자체의 보안 점검 ---")
    this_file = Path(__file__).resolve()
    this_source = this_file.read_text(encoding="utf-8")
    self_findings = check_security(this_source, this_file.name)

    if not self_findings:
        print("이 실습 코드에서는 보안 위험이 발견되지 않았습니다.")
    else:
        print(f"이 실습 코드에서 보안 위험 {len(self_findings)}건이 발견되었습니다.")
        for f in self_findings:
            print(f"  - {f['category']}: 줄 {f['line']}, {f['matched_code']}")

    print()
    print("=== 저장된 결과 ===")
    print(f"샘플 이미지: {sample_path.relative_to(CHAPTER_DIR)}")
    for info in resize_results:
        print(f"변환 이미지: results/{info['output_file']}")
    print(f"보안 점검 CSV: {report_path.relative_to(CHAPTER_DIR)}")
    print()
    print("=== 보안 점검 요약 ===")
    print("이 실습에서는 세 가지 보안 위험을 점검했다.")
    print("1. 파일 삭제: os.remove, shutil.rmtree 같은 코드가 있는지 확인한다.")
    print("2. 외부 전송: requests.post 같은 코드로 데이터를 외부로 보내는지 확인한다.")
    print("3. 비밀키 하드코딩: API 키나 비밀번호가 코드에 직접 쓰여 있는지 확인한다.")
    print("AI가 생성한 코드를 실행하기 전에 이 세 가지를 반드시 확인해야 한다.")


if __name__ == "__main__":
    main()
