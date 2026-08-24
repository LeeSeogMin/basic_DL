"""실습 12.2 - AI가 만든 코드를 실행하기 전에 위험 신호를 찾는다.

=========================  이 도구의 명세  =========================

[무엇을 하는가]
  파이썬 파일을 **텍스트로만 읽어** 위험 신호를 찾고, 위험도별로 정리한다.

[입력]
  같은 폴더에 있는 파이썬 파일 세 개.
    _sample_safe.py       위험 신호가 거의 없는 합성 예제
    _sample_risky.py      위험 신호를 일부러 넣은 합성 예제
    12-1-spec-to-tool.py  실습 12.1에서 우리가 직접 만든 도구

[출력]
  1. 찾은 위험 신호를 줄 번호와 함께 표로 출력한다.
  2. 같은 표를 results/safety_findings.csv 로 저장한다.
  3. 파일별·신호별 건수를 results/safety_report.png 막대그래프로 그린다.

[하면 안 되는 일]
  - 검사 대상 파일을 **실행하지 않는다. import 도 하지 않는다.** read_text 로 읽을 뿐이다.
    특히 _sample_risky.py 는 한 줄도 실행하지 않는다.
  - 검사 대상 파일을 고치거나 지우지 않는다.
  - 찾은 비밀키 문자열을 그대로 출력하지 않는다. 값을 가리고 위치만 알린다.
  - 인터넷에 접속하지 않는다.

[실패 시 동작]
  - 검사 대상 파일이 없으면 목록에 "파일 없음"으로 남기고 나머지 파일 검사를 이어간다.
  - 위험 신호를 하나도 못 찾아도 정상 종료한다. "0건"은 오류가 아니다.

===================================================================

이 도구는 사람을 대신해 판단하지 않는다. 판단할 자리를 표시해 줄 뿐이다.
"높음"이 붙었다고 그 코드가 반드시 악성인 것도 아니고,
"0건"이라고 안전이 증명된 것도 아니다. 마지막 판단은 실행 버튼을 누르는 사람이 한다.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

import _krfont

# ===================== 학생이 바꿔 볼 값 =====================
SCAN_TARGETS = [                      # 검사할 파일 목록
    "_sample_safe.py",
    "_sample_risky.py",
    "12-1-spec-to-tool.py",
]
SECRET_WORDS = [                      # 비밀키로 의심할 변수 이름 조각
    "api[_-]?key", "secret", "password", "passwd", "token", "credential",
]
SNIPPET_CHARS = 44                    # 표에 보여줄 코드 조각 길이
# ============================================================

CODE_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = CODE_DIR.parent
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"

RISK_ORDER = {"높음": 0, "중간": 1, "낮음": 2}
RISK_COLOR = {"높음": RED, "중간": ORANGE, "낮음": BLUE}

SECRET_PATTERN = (
    r"(?i)(" + "|".join(SECRET_WORDS) + r")\w*\s*=\s*[\"'][^\"']{6,}[\"']"
)

RULES = [
    {
        "이름": "파일 삭제",
        "위험도": "높음",
        "패턴": [r"\bos\.(remove|unlink|rmdir)\s*\(", r"\bshutil\.rmtree\s*\("],
        "왜": "지운 파일은 되돌릴 수 없다. 어느 경로를 지우는지 사람이 확인해야 한다",
    },
    {
        "이름": "외부 전송",
        "위험도": "높음",
        "패턴": [
            r"\brequests\.(post|put|patch)\s*\(",
            r"\burllib\.request\.urlopen\s*\(",
            r"\bsocket\.socket\s*\(",
        ],
        "왜": "무엇이 밖으로 나가는지 코드만 봐서는 모른다. 개인정보·비밀키가 실려 나갈 수 있다",
    },
    {
        "이름": "셸 명령 실행",
        "위험도": "높음",
        "패턴": [
            r"\bos\.system\s*\(",
            r"shell\s*=\s*True",
            r"(?<![\w.])eval\s*\(",
            r"(?<![\w.])exec\s*\(",
        ],
        "왜": "문자열을 붙여 만든 명령은 붙인 내용에 따라 다른 명령까지 실행한다",
    },
    {
        "이름": "하드코딩된 비밀키",
        "위험도": "중간",
        "패턴": [SECRET_PATTERN, r"[\"']sk-[A-Za-z0-9_-]{12,}[\"']"],
        "왜": "코드에 적힌 키는 코드를 복사하는 모든 사람에게 함께 복사된다",
    },
    {
        "이름": "멈추지 않는 반복",
        "위험도": "중간",
        "패턴": [],          # 아래 find_endless_loops 가 따로 찾는다
        "왜": "멈추는 조건이 없으면 사람이 강제로 끄기 전까지 돈다(멈추는 문장이 없으면 높음)",
    },
    {
        "이름": "파일 쓰기",
        "위험도": "낮음",
        "패턴": [
            r"open\s*\([^)]*[\"'][wax]b?[\"']",
            r"\.to_csv\s*\(",
            r"\.savefig\s*\(",
            r"\.write_text\s*\(",
            r"\.write_bytes\s*\(",
        ],
        "왜": "어느 경로에 쓰는지 확인한다. 자기 결과 폴더에 쓰는 것이면 문제가 아니다",
    },
]

STOP_WORDS = ("break", "return", "raise", "sys.exit", "SystemExit")


# ---------------------------------------------------------------- 표 출력 도우미
def _width(text: str) -> int:
    """한글은 두 칸, 영문·숫자는 한 칸으로 세어 글자 폭을 구한다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def print_table(headers: list[str], rows: list[list[str]], right: set[int] | None = None) -> None:
    """칸을 맞춰 표를 출력한다. right 에 든 열 번호는 오른쪽 정렬한다."""
    right = right or set()
    widths = [max([_width(headers[i])] + [_width(r[i]) for r in rows])
              for i in range(len(headers))]

    def line(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            pad = " " * (widths[i] - _width(cell))
            out.append(pad + cell if i in right else cell + pad)
        return "  ".join(out).rstrip()

    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))


# ---------------------------------------------------------------- 검사기 본체
def mask_secrets(code_line: str) -> str:
    """따옴표 안의 긴 문자열을 가린다. 찾은 비밀키를 로그에 그대로 남기지 않는다."""
    return re.sub(r"[\"'][^\"']{6,}[\"']", '"***"', code_line)


def snippet(code_line: str, rule_name: str) -> str:
    """표에 넣을 코드 조각을 만든다. 앞뒤 공백을 지우고 길이를 자른다."""
    text = code_line.strip()
    if rule_name == "하드코딩된 비밀키":
        text = mask_secrets(text)
    if len(text) > SNIPPET_CHARS:
        text = text[: SNIPPET_CHARS - 1] + "…"
    return text


def indent_of(code_line: str) -> int:
    return len(code_line) - len(code_line.lstrip())


def find_endless_loops(lines: list[str]) -> list[tuple[int, str, str]]:
    """`while True:` 를 찾고, 그 안에 멈추는 문장이 있는지 본다.

    멈추는 문장이 없으면 위험도를 '높음'으로 올린다.
    """
    hits = []
    for i, line in enumerate(lines):
        if not re.match(r"\s*while\s+True\s*:", line):
            continue
        base = indent_of(line)
        has_stop = False
        for follow in lines[i + 1:]:
            if follow.strip() and indent_of(follow) <= base:
                break
            if any(word in follow for word in STOP_WORDS):
                has_stop = True
                break
        risk = "중간" if has_stop else "높음"
        note = "멈추는 문장 있음" if has_stop else "멈추는 문장 없음"
        hits.append((i + 1, risk, note))
    return hits


def scan_file(path: Path) -> list[dict]:
    """파일 하나를 텍스트로 읽어 위험 신호를 찾는다. 실행하지도 import 하지도 않는다."""
    lines = path.read_text(encoding="utf-8").splitlines()
    findings = []
    seen = set()

    for rule in RULES:
        for pattern in rule["패턴"]:
            for i, line in enumerate(lines):
                if line.lstrip().startswith("#"):
                    continue                      # 주석 줄은 세지 않는다
                if not re.search(pattern, line):
                    continue
                key = (rule["이름"], i + 1)
                if key in seen:
                    continue                      # 같은 줄에서 같은 신호는 한 번만 센다
                seen.add(key)
                findings.append({
                    "파일": path.name,
                    "줄": i + 1,
                    "위험도": rule["위험도"],
                    "신호": rule["이름"],
                    "코드": snippet(line, rule["이름"]),
                })

    for line_no, risk, note in find_endless_loops(lines):
        findings.append({
            "파일": path.name,
            "줄": line_no,
            "위험도": risk,
            "신호": "멈추지 않는 반복",
            "코드": f"{lines[line_no - 1].strip()}  ({note})",
        })

    findings.sort(key=lambda f: (RISK_ORDER[f["위험도"]], f["줄"]))
    return findings


def draw_report(per_file: pd.DataFrame, per_rule: pd.DataFrame, ko: bool, out_path: Path) -> None:
    """왼쪽은 파일별 위험도 누적 막대, 오른쪽은 신호별 건수 막대."""
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # 왼쪽: 파일별 위험도 누적
    ax = axes[0]
    files = list(per_file.index)[::-1]
    left = [0.0] * len(files)
    for risk in ["높음", "중간", "낮음"]:
        values = [float(per_file.loc[f, risk]) for f in files]
        ax.barh(files, values, left=left, color=RISK_COLOR[risk],
                label=L(risk, {"높음": "high", "중간": "medium", "낮음": "low"}[risk], ko))
        for y, (base, value) in enumerate(zip(left, values)):
            if value > 0:
                ax.text(base + value / 2, y, f"{int(value)}",
                        ha="center", va="center", fontsize=11, color="white")
        left = [b + v for b, v in zip(left, values)]

    ax.set_xlim(0, max(max(left), 1) * 1.15)
    ax.set_xlabel(L("발견 건수", "findings", ko), fontsize=11)
    ax.set_title(L("파일별 위험 신호 건수", "findings per file", ko), fontsize=13, pad=10)
    handles, labels = ax.get_legend_handles_labels()

    # 오른쪽: 신호 종류별 건수
    ax = axes[1]
    rules = list(per_rule["신호"])[::-1]
    counts = list(per_rule["건수"])[::-1]
    colors = [RISK_COLOR[r] for r in list(per_rule["위험도"])[::-1]]
    ax.barh(rules, counts, color=colors)
    ax.set_xlim(0, max(max(counts), 1) * 1.25)
    ax.set_xlabel(L("발견 건수", "findings", ko), fontsize=11)
    ax.set_title(L("신호 종류별 건수 (색 = 위험도)", "findings per signal (color = risk)", ko),
                 fontsize=13, pad=10)
    for y, value in enumerate(counts):
        ax.text(value + max(counts) * 0.04, y, str(int(value)),
                va="center", fontsize=11, color="#333333")

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # 건수는 정수다
        ax.tick_params(labelsize=11, length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(
        L("실행 전 코드 점검 결과 (합성 예제 파일 — 위험 예제는 읽기만 했다)",
          "Pre-run code check (synthetic samples; the risky one was only read)", ko),
        fontsize=14,
    )
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    print("=== 실습 12.2 실행 전 코드 안전 점검 ===")
    _krfont.report(ko)
    print("검사 대상은 합성 예제 파일이다. 실제 프로젝트 코드가 아니다.")
    print("이 도구는 대상 파일을 텍스트로 읽기만 한다. 실행하지도 import 하지도 않는다.")
    print("특히 _sample_risky.py 는 한 줄도 실행하지 않는다.")
    print()

    all_findings: list[dict] = []
    scanned: list[str] = []

    print("=== 검사 대상 ===")
    rows = []
    for name in SCAN_TARGETS:
        path = CODE_DIR / name
        if not path.exists():
            rows.append([name, "파일 없음", "-"])
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        found = scan_file(path)
        all_findings.extend(found)
        scanned.append(name)
        rows.append([name, f"{line_count}줄", f"{len(found)}건"])
    print_table(["파일", "길이", "발견"], rows, right={1, 2})
    print()

    if not scanned:
        print("검사할 파일이 하나도 없다.")
        return

    frame = pd.DataFrame(all_findings)

    print("=== 찾은 위험 신호 ===")
    if frame.empty:
        print("위험 신호 0건. 다만 0건이 안전을 증명하지는 않는다.")
    else:
        order = frame.assign(_r=frame["위험도"].map(RISK_ORDER)).sort_values(
            ["_r", "파일", "줄"]
        )
        print_table(
            ["위험도", "신호", "파일", "줄", "코드"],
            [[r["위험도"], r["신호"], r["파일"], str(r["줄"]), r["코드"]]
             for _i, r in order.iterrows()],
            right={3},
        )
    print()

    # 파일별 위험도 집계
    levels = ["높음", "중간", "낮음"]
    per_file = pd.DataFrame(0, index=scanned, columns=levels)
    for f in all_findings:
        per_file.loc[f["파일"], f["위험도"]] += 1
    per_file["합계"] = per_file[levels].sum(axis=1)

    print("=== 파일별 집계 ===")
    print_table(
        ["파일"] + levels + ["합계", "판정"],
        [[name] + [str(int(per_file.loc[name, c])) for c in levels]
         + [str(int(per_file.loc[name, "합계"])),
            "실행 전 사람이 읽어야 함" if per_file.loc[name, "높음"] > 0 else "높음 신호 없음"]
         for name in scanned],
        right={1, 2, 3, 4},
    )
    print()

    # 신호 종류별 집계
    counts = {rule["이름"]: 0 for rule in RULES}
    worst = {rule["이름"]: rule["위험도"] for rule in RULES}
    for f in all_findings:
        counts[f["신호"]] += 1
        if RISK_ORDER[f["위험도"]] < RISK_ORDER[worst[f["신호"]]]:
            worst[f["신호"]] = f["위험도"]      # 실제로 나온 것 중 가장 높은 위험도를 쓴다
    per_rule = pd.DataFrame([
        {"신호": rule["이름"], "위험도": worst[rule["이름"]],
         "건수": counts[rule["이름"]], "왜 위험한가": rule["왜"]}
        for rule in RULES
    ])

    print("=== 신호 종류별 집계 ===")
    print_table(
        ["신호", "위험도", "건수", "왜 위험한가"],
        [[r["신호"], r["위험도"], str(r["건수"]), r["왜 위험한가"]]
         for _i, r in per_rule.iterrows()],
        right={2},
    )
    print()

    csv_path = RESULTS_DIR / "safety_findings.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    png_path = RESULTS_DIR / "safety_report.png"
    draw_report(per_file, per_rule, ko, png_path)

    print("=== 저장된 결과 ===")
    print(f"발견 목록: results/{csv_path.name}")
    print(f"막대그래프: results/{png_path.name}")
    print()
    high = int(per_file["높음"].sum())
    print(f"'높음' 신호 {high}건. 이 줄들은 실행 전에 사람이 직접 읽어야 한다.")
    print("이 도구는 판단하지 않는다. 판단할 자리를 표시할 뿐이다.")


if __name__ == "__main__":
    main()
