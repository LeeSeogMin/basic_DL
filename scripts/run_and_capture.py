#!/usr/bin/env python3
"""
Execution evidence gate for course practice code.

Runs practice/chapter{N}/code/*.py files and writes:
  - practice/chapter{N}/results/{stem}.log
  - practice/chapter{N}/results/{stem}.evidence.json

Use `--verify` to check whether captured results are stale after source edits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRACTICE_DIR = PROJECT_ROOT / "practice"


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_chapter_dir(chapter: str) -> Path:
    raw = str(chapter)
    candidates = [f"chapter{raw}"]
    try:
        candidates.append(f"chapter{int(raw):02d}")
    except ValueError:
        pass

    for name in dict.fromkeys(candidates):
        chapter_dir = PRACTICE_DIR / name
        if (chapter_dir / "code").is_dir():
            return chapter_dir

    raise FileNotFoundError(
        "code/ 폴더를 가진 장 폴더를 찾지 못했습니다. "
        f"시도: {', '.join(candidates)}"
    )


def discover_code_files(chapter_dir: Path, only: str | None) -> list[Path]:
    code_dir = chapter_dir / "code"
    if only:
        target = code_dir / only
        if not target.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {target}")
        return [target]

    return sorted(
        path
        for path in code_dir.glob("*.py")
        if not path.name.startswith("_") and "requirements" not in path.name
    )


def run_one(py_file: Path, timeout: int) -> tuple[dict, str]:
    source = py_file.read_text(encoding="utf-8")
    started = time.perf_counter()
    started_iso = datetime.now(timezone.utc).astimezone().isoformat()

    try:
        proc = subprocess.run(
            [sys.executable, py_file.name],
            cwd=py_file.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\n[TIMEOUT] {timeout}s 초과로 중단됨"
        timed_out = True

    duration = round(time.perf_counter() - started, 2)
    combined = (
        f"$ {sys.executable} {py_file.name}\n\n"
        f"=== STDOUT ===\n{stdout}\n"
        f"=== STDERR ===\n{stderr}"
    )

    evidence = {
        "file": py_file.name,
        "chapter_relpath": str(py_file.relative_to(PROJECT_ROOT)),
        "started_at": started_iso,
        "duration_sec": duration,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "success": exit_code == 0,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "source_sha256": sha256_of_text(source),
        "output_sha256": sha256_of_text(combined),
        "output_log": None,
    }
    return evidence, combined


def write_results(chapter_dir: Path, py_file: Path, evidence: dict, combined: str) -> None:
    results_dir = chapter_dir / "results"
    results_dir.mkdir(exist_ok=True)

    log_path = results_dir / f"{py_file.stem}.log"
    log_path.write_text(combined, encoding="utf-8")

    evidence["output_log"] = str(log_path.relative_to(PROJECT_ROOT))
    evidence_path = results_dir / f"{py_file.stem}.evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def verify(chapter_dir: Path, files: list[Path]) -> int:
    results_dir = chapter_dir / "results"
    problems = 0

    for py_file in files:
        evidence_path = results_dir / f"{py_file.stem}.evidence.json"
        if not evidence_path.exists():
            print(f"[missing] 증거 없음: {py_file.name}")
            problems += 1
            continue

        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        current_hash = sha256_of_text(py_file.read_text(encoding="utf-8"))

        if current_hash != evidence.get("source_sha256"):
            print(f"[stale] 낡은 결과: {py_file.name} - 소스 변경 후 재실행 필요")
            problems += 1
        elif not evidence.get("success"):
            print(f"[failed] 실패한 실행: {py_file.name} - exit_code={evidence.get('exit_code')}")
            problems += 1
        else:
            print(f"[ok] 유효: {py_file.name}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="딥러닝 기초 실행 증거 게이트")
    parser.add_argument("chapter", help="장 번호. 예: 5")
    parser.add_argument("--file", default=None, help="특정 코드 파일만 실행")
    parser.add_argument("--timeout", type=int, default=1800, help="파일당 제한시간(초)")
    parser.add_argument("--verify", action="store_true", help="실행 없이 무결성만 검증")
    args = parser.parse_args()

    chapter_dir = resolve_chapter_dir(args.chapter)
    files = discover_code_files(chapter_dir, args.file)

    if not files:
        print(f"실행할 .py 파일이 없습니다: {chapter_dir / 'code'}")
        return 1

    if args.verify:
        print(f"[verify] {chapter_dir.name} - {len(files)}개 파일")
        problems = verify(chapter_dir, files)
        print(f"검증 완료: 문제 {problems}건")
        return 1 if problems else 0

    print(f"[run] {chapter_dir.name} - {len(files)}개 파일")
    all_ok = True

    for py_file in files:
        print(f"> {py_file.name} 실행 중", flush=True)
        evidence, combined = run_one(py_file, args.timeout)
        write_results(chapter_dir, py_file, evidence, combined)

        status = "success" if evidence["success"] else f"failed(exit={evidence['exit_code']})"
        print(
            f"  {status} {evidence['duration_sec']}s -> "
            f"results/{py_file.stem}.log "
            f"(output_sha256={evidence['output_sha256'][:12]})"
        )
        all_ok = all_ok and evidence["success"]

    print(f"증거 생성 완료: {(chapter_dir / 'results').relative_to(PROJECT_ROOT)}/")
    print("본문의 실행 결과는 위 로그 파일에서만 인용할 것.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

