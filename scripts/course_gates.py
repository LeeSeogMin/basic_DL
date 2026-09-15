#!/usr/bin/env python3
"""학기 운영 게이트.

교재 15장 작성이 끝난 뒤의 위험은 "장을 안 썼다"가 아니라
"고친 뒤 뒤따라야 할 것을 안 고쳤다"이다. 세 가지를 검사한다.

  G1 실행 증거 무결성 — 코드를 고치고 결과를 다시 만들지 않은 장이 있는가
  G2 본문 인용 로그 실존 — docs 본문이 가리키는 로그 파일이 실제로 있는가
  G3 md - PDF 동기     — 본문을 고치고 PDF를 다시 만들지 않은 장이 있는가

검사 로직은 새로 쓰지 않고 run_and_capture.py의 것을 그대로 쓴다.

사용법:
  python3 scripts/course_gates.py
  python3 scripts/course_gates.py --gates g1,g2
  python3 scripts/course_gates.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_and_capture import (  # noqa: E402
    PRACTICE_DIR,
    discover_code_files,
    sha256_of_text,
    verify as verify_chapter,
)

DOCS_DIR = PROJECT_ROOT / "docs"

# docs 본문의 로그 인용 표기. 예: _출처: `practice/chapter5/results/5-1-one-neuron.log`_
CITATION_RE = re.compile(r"_출처: `([^`]+)`_")


def chapter_number(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


def gate_evidence(verbose: bool) -> tuple[int, int]:
    """G1. 모든 장의 실습 코드와 증거 JSON의 소스 해시가 맞는지 본다."""
    chapter_dirs = sorted(
        (d for d in PRACTICE_DIR.glob("chapter*") if (d / "code").is_dir()),
        key=lambda d: chapter_number(d.name),
    )

    checked = 0
    problems = 0
    for chapter_dir in chapter_dirs:
        files = discover_code_files(chapter_dir, None)
        if not files:
            continue
        checked += len(files)
        problems += verify_chapter(chapter_dir, files, quiet=not verbose)

    print(f"[G1] 실행 증거 무결성: {checked}개 파일 검사, 문제 {problems}건")
    return checked, problems


def gate_citations(verbose: bool) -> tuple[int, int]:
    """G2. 본문이 인용한 로그 파일이 실제로 있는지 본다."""
    checked = 0
    problems = 0

    md_files = sorted(DOCS_DIR.glob("ch*.md"), key=lambda p: chapter_number(p.stem))
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        for match in CITATION_RE.finditer(text):
            rel = match.group(1)
            checked += 1
            target = PROJECT_ROOT / rel
            if not target.exists():
                print(f"[missing] 인용한 로그가 없음: {md_path.name} -> {rel}")
                problems += 1
            elif verbose:
                print(f"[ok] 인용 확인: {md_path.name} -> {rel}")

    print(f"[G2] 본문 인용 로그 실존: {checked}건 검사, 문제 {problems}건")
    return checked, problems


def gate_pdf_sync(verbose: bool) -> tuple[int, int]:
    """G3. PDF를 만든 시점의 본문 해시와 지금 본문 해시를 맞춰 본다.

    git은 파일 수정 시각을 보존하지 않으므로 시각 비교로는 판정할 수 없다.
    md2pdf.py가 변환할 때 남긴 docs/ch{N}.pdf.evidence.json의 해시를 쓴다.
    """
    checked = 0
    problems = 0
    unrecorded: list[str] = []

    pdf_files = sorted(DOCS_DIR.glob("ch*.pdf"), key=lambda p: chapter_number(p.stem))
    for pdf_path in pdf_files:
        md_path = pdf_path.with_suffix(".md")
        if not md_path.exists():
            print(f"[orphan] 원본 마크다운이 없는 PDF: {pdf_path.name}")
            problems += 1
            continue

        evidence_path = pdf_path.with_name(pdf_path.name + ".evidence.json")
        if not evidence_path.exists():
            unrecorded.append(pdf_path.name)
            continue

        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        checked += 1
        current = sha256_of_text(md_path.read_text(encoding="utf-8"))
        if current != evidence.get("source_sha256"):
            print(
                f"[stale] 본문을 고친 뒤 PDF를 다시 만들지 않음: {pdf_path.name} "
                f"- python3 scripts/md2pdf.py {md_path.relative_to(PROJECT_ROOT)}"
            )
            problems += 1
        elif verbose:
            print(f"[ok] PDF 동기: {pdf_path.name}")

    note = ""
    if unrecorded:
        note = f", 생성 기록 없어 건너뜀 {len(unrecorded)}개({', '.join(unrecorded)})"
    print(f"[G3] md-PDF 동기: PDF {len(pdf_files)}개 중 {checked}개 검사, 문제 {problems}건{note}")
    return checked, problems


GATES = {
    "g1": gate_evidence,
    "g2": gate_citations,
    "g3": gate_pdf_sync,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="딥러닝 기초 학기 운영 게이트")
    parser.add_argument(
        "--gates",
        default="g1,g2,g3",
        help="실행할 게이트. 기본값 g1,g2,g3",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="통과한 항목도 한 줄씩 출력한다",
    )
    args = parser.parse_args()

    selected = [name.strip().lower() for name in args.gates.split(",") if name.strip()]
    unknown = [name for name in selected if name not in GATES]
    if unknown:
        print(f"모르는 게이트: {', '.join(unknown)} (가능: g1, g2, g3)")
        return 1

    total_checked = 0
    total_problems = 0
    for name in selected:
        checked, problems = GATES[name](args.verbose)
        total_checked += checked
        total_problems += problems

    print(f"게이트 합계: {total_checked}건 검사, 문제 {total_problems}건")

    if total_checked == 0:
        print("COURSE_GATES_FAIL: 검사 대상을 하나도 찾지 못했다. 경로 설정을 확인한다.")
        return 1

    if total_problems:
        print(f"COURSE_GATES_FAIL: 문제 {total_problems}건")
        return 1

    print("COURSE_GATES_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
