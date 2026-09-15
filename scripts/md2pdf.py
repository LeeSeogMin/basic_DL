#!/usr/bin/env python3
"""강의자료 마크다운을 PDF로 바꾼다.

변환 경로:
  마크다운 --(pandoc)--> HTML(이미지 내장) --(Chrome headless)--> PDF

LaTeX를 설치하지 않아도 되고, 그림과 표가 화면에서 보이던 모양 그대로 나온다.

사용법:
  python scripts/md2pdf.py docs/ch4.md
  python scripts/md2pdf.py docs/ch4.md --out docs/ch4.pdf
  python scripts/md2pdf.py docs/ch4.md --keep-html
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# 해시 함수를 실행 증거 게이트와 똑같은 것으로 쓴다.
# course_gates.py의 G3가 여기서 남긴 해시를 그대로 비교한다.
from run_and_capture import sha256_of_text  # noqa: E402

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]

CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
}

body {
  font-family: "Malgun Gothic", "맑은 고딕", "AppleSDGothicNeo-Regular",
               "Apple SD Gothic Neo", "NanumGothic", "Noto Sans KR", sans-serif;
  font-size: 10.5pt;
  line-height: 1.75;
  color: #1a1a1a;
  max-width: 100%;
  margin: 0;
  word-break: keep-all;
}

/* pandoc이 파일명으로 만들어 넣는 제목 줄은 쓰지 않는다.
   문서의 진짜 제목은 마크다운 첫 줄의 # 제목이다. */
h1.title, header#title-block-header { display: none; }

/* 그림 위에 alt 텍스트가 캡션처럼 한 번 더 찍히는 것을 막는다.
   그림 설명은 본문에서 "그림 4.1 ..." 문단으로 따로 단다. */
figcaption { display: none; }

figure { margin: 12px 0; page-break-inside: avoid; }

h1 {
  font-size: 20pt;
  border-bottom: 3px solid #4C78A8;
  padding-bottom: 8px;
  margin-bottom: 24px;
}

h2 {
  font-size: 15pt;
  color: #4C78A8;
  margin-top: 28px;
  padding: 6px 0 6px 10px;
  border-left: 6px solid #4C78A8;
  background: #f3f6fa;
  page-break-after: avoid;
}

h3 {
  font-size: 13pt;
  margin-top: 24px;
  padding-bottom: 4px;
  border-bottom: 1px solid #d5dbe3;
  page-break-after: avoid;
}

h4 {
  font-size: 11.5pt;
  margin-top: 18px;
  color: #333;
  page-break-after: avoid;
}

p { margin: 8px 0; }

ul, ol { margin: 8px 0; padding-left: 22px; }
li { margin: 4px 0; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 14px 0;
  font-size: 9.5pt;
  page-break-inside: avoid;
}

th, td {
  border: 1px solid #c9d1da;
  padding: 6px 9px;
  vertical-align: top;
}

th {
  background: #eef2f7;
  font-weight: 600;
  text-align: left;
}

tr:nth-child(even) td { background: #fafbfc; }

code {
  font-family: "D2Coding", "Consolas", "Courier New", monospace;
  font-size: 9.5pt;
  background: #f2f4f6;
  padding: 1px 4px;
  border-radius: 3px;
}

pre {
  background: #f7f8fa;
  border: 1px solid #dfe3e8;
  border-left: 4px solid #79706E;
  border-radius: 4px;
  padding: 10px 12px;
  overflow-x: auto;
  page-break-inside: avoid;
}

pre code {
  background: none;
  padding: 0;
  font-size: 9pt;
  line-height: 1.5;
}

img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 12px auto;
  page-break-inside: avoid;
}

blockquote {
  border-left: 4px solid #F58518;
  background: #fff8f0;
  margin: 12px 0;
  padding: 8px 14px;
  color: #444;
}

hr {
  border: none;
  border-top: 1px solid #dde2e8;
  margin: 20px 0;
}

a { color: #2f6fb0; text-decoration: none; word-break: break-all; }

/* 표 바로 위의 그림 번호 문단(**그림 4.1** ...)이 그림과 떨어지지 않게 한다 */
p > strong:first-child { color: #2b3a4a; }

.footnotes {
  font-size: 9pt;
  color: #555;
  border-top: 1px solid #dde2e8;
  margin-top: 26px;
  padding-top: 10px;
}
"""


def find_chrome() -> str:
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Chrome 또는 Edge를 찾지 못했습니다. PDF 변환에는 둘 중 하나가 필요합니다."
    )


def require_pandoc() -> str:
    found = shutil.which("pandoc")
    if not found:
        raise FileNotFoundError(
            "pandoc을 찾지 못했습니다. https://pandoc.org 에서 설치하세요."
        )
    return found


def md_to_html(md_path: Path, html_path: Path, css_path: Path) -> None:
    """pandoc으로 마크다운을 이미지가 내장된 단일 HTML로 바꾼다."""
    cmd = [
        require_pandoc(),
        str(md_path),
        "--from", "gfm+footnotes",
        "--to", "html5",
        "--standalone",
        "--embed-resources",
        "--css", str(css_path),
        # 이미지의 상대 경로는 마크다운 파일이 있는 폴더를 기준으로 찾는다
        "--resource-path", f"{md_path.parent}{os.pathsep}{PROJECT_ROOT}",
        "--metadata", f"title={md_path.stem}",
        "--output", str(html_path),
    ]
    subprocess.run(cmd, check=True)


def html_to_pdf(html_path: Path, pdf_path: Path, timeout: int = 300) -> str:
    """Chrome headless로 HTML을 PDF로 인쇄한다.

    Chrome 153의 `--headless=new`는 인쇄를 끝낸 뒤에도 프로세스가 남는 경우가 있다.
    그래서 프로세스가 끝나기를 기다리지 않는다. PDF 파일이 생기고 크기가 더 늘지
    않으면 인쇄가 끝난 것으로 보고, 남아 있는 Chrome을 직접 내린다.
    """
    chrome = find_chrome()
    pdf_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory() as profile_dir:
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=10000",
            f"--user-data-dir={profile_dir}",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        deadline = time.monotonic() + timeout
        last_size = -1
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            if pdf_path.exists():
                size = pdf_path.stat().st_size
                if size > 0 and size == last_size:
                    break
                last_size = size
            time.sleep(1.0)

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

        stdout, stderr = "", ""
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        sys.stderr.write((stdout or "") + "\n" + (stderr or "") + "\n")
        raise RuntimeError(f"PDF가 만들어지지 않았습니다: {pdf_path}")

    return chrome


def write_pdf_evidence(md_path: Path, pdf_path: Path, chrome: str) -> Path:
    """PDF를 만든 시점의 본문 해시를 남긴다.

    git은 파일 수정 시각을 보존하지 않아서, 클론한 저장소에서는 시각 비교로
    "본문을 고친 뒤 PDF를 안 만들었다"를 판정할 수 없다. 해시를 남겨야 판정된다.
    실행 증거(run_and_capture.py)와 같은 형식을 쓴다.
    """
    pandoc_version = ""
    pandoc_path = shutil.which("pandoc")
    if pandoc_path:
        result = subprocess.run(
            [pandoc_path, "--version"], capture_output=True, text=True
        )
        pandoc_version = result.stdout.splitlines()[0].strip() if result.stdout else ""

    def relpath(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    evidence = {
        "source_markdown": relpath(md_path),
        "source_sha256": sha256_of_text(md_path.read_text(encoding="utf-8")),
        "output_pdf": relpath(pdf_path),
        "output_bytes": pdf_path.stat().st_size,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "pandoc": pandoc_version,
        "browser": chrome,
    }

    evidence_path = pdf_path.with_name(pdf_path.name + ".evidence.json")
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return evidence_path


def main() -> int:
    parser = argparse.ArgumentParser(description="강의자료 마크다운 -> PDF 변환")
    parser.add_argument("markdown", help="변환할 마크다운 파일. 예: docs/ch4.md")
    parser.add_argument("--out", default=None, help="저장할 PDF 경로")
    parser.add_argument("--keep-html", action="store_true",
                        help="중간 HTML 파일을 지우지 않는다")
    args = parser.parse_args()

    md_path = Path(args.markdown).resolve()
    if not md_path.exists():
        print(f"파일을 찾을 수 없습니다: {md_path}")
        return 1

    pdf_path = Path(args.out).resolve() if args.out else md_path.with_suffix(".pdf")
    html_path = md_path.with_suffix(".html")
    css_path = md_path.parent / "_pdf-style.css"

    print(f"[1/3] 스타일시트 작성: {css_path.name}")
    css_path.write_text(CSS, encoding="utf-8")

    print(f"[2/3] 마크다운 -> HTML: {html_path.name}")
    md_to_html(md_path, html_path, css_path)

    print(f"[3/3] HTML -> PDF: {pdf_path.name}")
    chrome = html_to_pdf(html_path, pdf_path)
    evidence_path = write_pdf_evidence(md_path, pdf_path, chrome)

    if not args.keep_html:
        html_path.unlink(missing_ok=True)
        css_path.unlink(missing_ok=True)

    size_kb = pdf_path.stat().st_size / 1024
    print(f"완료: {pdf_path} ({size_kb:,.0f} KB)")
    print(f"생성 기록: {evidence_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
