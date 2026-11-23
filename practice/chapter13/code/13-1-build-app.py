"""실습 13.1 - 모델을 파일 한 개짜리 앱으로 내보낸다.

지금까지 실습 코드는 정확도 한 줄을 출력하고 끝났다. 그 결과는 코드를 실행한
사람만 볼 수 있다. 이 코드는 학습한 모델의 숫자를 꺼내 results/my_app.html 파일
하나에 담는다. 그 파일을 브라우저로 열면 파이썬이 없는 컴퓨터에서도 앱이 동작한다.

[입력]  합성 날씨 데이터(_bikedata.py 가 만든다. 실제 대여 기록이 아니다).
        아래 MY_INPUTS 에 적은 시험 입력 6건.
[출력]  1. results/my_app.html  브라우저로 여는 앱. 인터넷 연결 없이 동작한다.
        2. results/app_check.csv  시험 입력을 앱 규칙으로 처리한 결과.
        3. results/app_check.png  같은 결과를 그린 그림.
[하면 안 되는 일]
        - 이름·연락처 같은 개인정보를 입력으로 받지 않는다.
        - 앱 파일에서 인터넷으로 아무것도 보내지 않는다.
[실패 시 동작]
        - 내보낸 숫자로 계산한 확률이 원래 모델과 0.001 넘게 다르면 오류를 내고 멈춘다.

앱이 하는 일은 다섯 가지다.
  1. 입력 받기   무엇을 넣는지 화면에 적는다
  2. 입력 검사   있을 수 없는 값이면 계산하지 않고 이유를 말한다
  3. 계산        확률을 계산한다
  4. 결과 보이기 판정, 확률, 각 값이 어느 쪽으로 얼마나 밀었는지 함께 쓴다
  5. 한계 알리기 배운 범위 밖 입력이면 경고하고, 한계 안내문을 항상 적는다
"""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import _bikedata
import _krfont

# ===================== 학생이 바꿔 볼 값 =====================
APP_TITLE = "오늘은 공공자전거 대여가 많은 날일까"

# 앱 화면 아래에 항상 적을 한계 안내문. 자기 문장으로 고쳐 쓴다.
MODEL_LIMITS = [
    "컴퓨터로 만든 합성 데이터로 배웠다. 실제 대여 기록으로 시험한 적이 없다.",
    "기온·강수량·풍속 세 숫자만 본다. 요일, 행사, 미세먼지는 보지 않는다.",
    "기온 3~26도인 날만 보고 배웠다. 그보다 춥거나 더운 날의 답은 믿기 어렵다.",
]

# 앱에 넣어 볼 시험 입력. (설명, 기온, 강수량, 풍속) 순서다.
MY_INPUTS = [
    ("맑은 봄날", 21.0, 0.0, 2.0),
    ("비 오는 가을날", 15.0, 12.0, 4.0),
    ("쌀쌀하고 바람 부는 날", 8.0, 0.0, 7.0),
    ("흐리고 비가 조금 온 날", 18.0, 4.0, 3.0),
    ("한여름 폭염", 36.0, 0.0, 2.0),
    ("잘못 입력한 예", 20.0, -5.0, 2.0),     # 강수량이 음수다 -> 받지 않는다
]
# ============================================================

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

OK, WARN, REJECT = "답함", "범위 밖 경고", "입력 거절"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


# ---------------------------------------------------------------- 표 출력 도우미
def _width(text: str) -> int:
    """한글은 두 칸, 영문·숫자는 한 칸으로 세어 글자 폭을 구한다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def print_table(headers: list[str], rows: list[list[str]], right: set[int] | None = None) -> None:
    """칸을 맞춰 표를 출력한다. right 에 든 열 번호는 오른쪽 정렬한다."""
    right = right or set()
    widths = [max(_width(headers[i]), *(_width(r[i]) for r in rows)) for i in range(len(headers))]

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


# ---------------------------------------------------------------- 모델 내보내기
def export_spec(model, x_train: np.ndarray) -> dict:
    """모델에서 앱이 쓸 숫자를 꺼낸다. 앱 파일에는 이 숫자만 들어간다."""
    scaler, clf = model[0], model[1]
    ranges = _bikedata.learned_ranges(x_train)
    inputs = []
    for i, (name, unit, low, high, step) in enumerate(_bikedata.FEATURES):
        inputs.append({
            "name": name, "unit": unit, "min": low, "max": high, "step": step,
            "learned_min": ranges[i][0], "learned_max": ranges[i][1],
            "mean": round(float(scaler.mean_[i]), 6),       # 표준화에 쓰는 평균
            "scale": round(float(scaler.scale_[i]), 6),     # 표준화에 쓰는 표준편차
            "weight": round(float(clf.coef_[0][i]), 6),     # 모델이 배운 무게
        })
    return {
        "title": APP_TITLE,
        "labels": _bikedata.LABELS,
        "bias": round(float(clf.intercept_[0]), 6),
        "inputs": inputs,
        "limits": MODEL_LIMITS,
    }


def app_answer(spec: dict, values: list[float]) -> dict:
    """앱 규칙으로 입력 한 건을 처리한다. my_app.html 안의 계산과 같은 순서다."""
    # 2. 입력 검사 - 있을 수 없는 값이면 계산하지 않는다
    for item, value in zip(spec["inputs"], values):
        if not item["min"] <= value <= item["max"]:
            reason = f'{item["name"]}은(는) {item["min"]:g}~{item["max"]:g} {item["unit"]} 사이여야 한다'
            return {"status": REJECT, "reason": reason}

    # 3. 계산 - 표준화한 값에 무게를 곱해 더하고, 확률로 바꾼다
    pushes = [item["weight"] * (value - item["mean"]) / item["scale"]
              for item, value in zip(spec["inputs"], values)]
    prob = 1.0 / (1.0 + math.exp(-(spec["bias"] + sum(pushes))))

    # 5. 한계 알리기 - 배운 범위 밖 입력을 찾는다
    outside = [f'{item["name"]} {value:g}{item["unit"]}'
               f'(배운 범위 {item["learned_min"]:g}~{item["learned_max"]:g})'
               for item, value in zip(spec["inputs"], values)
               if not item["learned_min"] <= value <= item["learned_max"]]

    return {
        "status": WARN if outside else OK,
        "label": spec["labels"][1 if prob >= 0.5 else 0],
        "prob": prob,
        "pushes": pushes,
        "outside": outside,
    }


# ---------------------------------------------------------------- 앱 파일
HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
         max-width: 560px; margin: 24px auto; padding: 0 16px; color: #222; line-height: 1.6; }
  h1 { font-size: 1.3rem; }
  .notice { background: #eef3f9; border-left: 4px solid #4C78A8; padding: 8px 12px; font-size: 0.9rem; }
  label { display: block; margin-top: 14px; font-weight: 600; }
  .hint { font-weight: 400; color: #666; font-size: 0.85rem; }
  input { font-size: 1rem; padding: 6px 8px; width: 120px; }
  .card { border: 1px solid #ccc; border-radius: 8px; padding: 12px 16px; margin-top: 20px; }
  .reject { border-color: #79706E; background: #f1f1f1; }
  .warn { background: #fdf0e0; border-left: 4px solid #F58518; padding: 8px 12px; margin-top: 10px; }
  .bar { height: 14px; background: #ddd; border-radius: 7px; overflow: hidden; }
  .bar div { height: 100%; background: #4C78A8; }
  table { border-collapse: collapse; margin-top: 8px; font-size: 0.9rem; }
  td { padding: 2px 12px 2px 0; }
  ul { padding-left: 20px; font-size: 0.9rem; }
</style>
</head>
<body>
<h1 id="title"></h1>
<p class="notice">이 화면의 답은 AI 모델이 계산한 예측이다. 사람이 확인한 답이 아니다.</p>

<div id="form"></div>
<div id="result" class="card"></div>

<h2 style="font-size:1rem">이 모델이 틀릴 수 있는 경우</h2>
<ul id="limits"></ul>

<script id="spec" type="application/json">__SPEC__</script>
<script>
// 이 파일은 인터넷으로 아무것도 보내지 않는다. 계산은 전부 이 브라우저 안에서 한다.
const spec = JSON.parse(document.getElementById("spec").textContent);
document.getElementById("title").textContent = spec.title;

function add(parent, tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  parent.appendChild(el);
  return el;
}

// 1. 입력 받기 - 무엇을 넣는지와 받는 범위를 화면에 적는다
const form = document.getElementById("form");
const fields = spec.inputs.map(function (item) {
  const label = add(form, "label", item.name + " (" + item.unit + ") ");
  add(label, "span", "받는 범위 " + item.min + "~" + item.max, "hint");
  const input = add(form, "input");
  input.type = "number";
  input.step = item.step;
  input.value = Math.round(item.mean);          // 처음 값은 훈련 데이터의 평균
  input.addEventListener("input", update);
  return input;
});

for (const line of spec.limits) add(document.getElementById("limits"), "li", line);

function update() {
  const box = document.getElementById("result");
  box.textContent = "";
  box.className = "card";
  const values = fields.map(function (f) { return f.value === "" ? NaN : Number(f.value); });

  // 2. 입력 검사 - 있을 수 없는 값이면 계산하지 않는다
  for (let i = 0; i < spec.inputs.length; i++) {
    const item = spec.inputs[i];
    if (!(values[i] >= item.min && values[i] <= item.max)) {
      box.className = "card reject";
      add(box, "strong", "계산하지 않았다");
      add(box, "div", item.name + "은(는) " + item.min + "~" + item.max + " " + item.unit + " 사이여야 한다");
      return;
    }
  }

  // 3. 계산 - 표준화한 값에 무게를 곱해 더하고, 확률로 바꾼다
  const pushes = spec.inputs.map(function (item, i) {
    return item.weight * (values[i] - item.mean) / item.scale;
  });
  const total = pushes.reduce(function (a, b) { return a + b; }, spec.bias);
  const prob = 1 / (1 + Math.exp(-total));

  // 4. 결과 보이기 - 판정, 확률, 각 값이 민 크기
  add(box, "strong", "예측: " + spec.labels[prob >= 0.5 ? 1 : 0]);
  add(box, "div", spec.labels[1] + "일 확률 " + prob.toFixed(3));
  const bar = add(box, "div", undefined, "bar");
  add(bar, "div").style.width = (prob * 100).toFixed(1) + "%";
  add(box, "div", "각 값이 민 크기 (양수는 '" + spec.labels[1] + "' 쪽)", "hint");
  const table = add(box, "table");
  spec.inputs.forEach(function (item, i) {
    const row = add(table, "tr");
    add(row, "td", item.name + " " + values[i] + item.unit);
    add(row, "td", (pushes[i] >= 0 ? "+" : "") + pushes[i].toFixed(2));
  });

  // 5. 한계 알리기 - 배운 범위 밖 입력이면 경고한다
  const outside = spec.inputs.filter(function (item, i) {
    return values[i] < item.learned_min || values[i] > item.learned_max;
  });
  if (outside.length > 0) {
    const names = outside.map(function (item) {
      return item.name + "(배운 범위 " + item.learned_min + "~" + item.learned_max + item.unit + ")";
    });
    add(box, "div", "배운 범위 밖 입력: " + names.join(", ") + ". 위 확률은 믿기 어렵다.", "warn");
  }
}
update();
</script>
</body>
</html>
"""


def write_app(spec: dict, out_path: Path) -> None:
    """모델 숫자를 HTML 파일 한 개에 담는다."""
    spec_text = json.dumps(spec, ensure_ascii=False, indent=1).replace("</", "<\\/")
    title = spec["title"].replace("&", "&amp;").replace("<", "&lt;")
    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__SPEC__", spec_text)
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- 그림
def draw_check(cases: pd.DataFrame, p_model: np.ndarray, p_app: np.ndarray,
               ko: bool, out_path: Path) -> None:
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    colors = {OK: BLUE, WARN: ORANGE, REJECT: GRAY}
    english = {OK: "answered", WARN: "outside learned range", REJECT: "rejected"}
    names = [L(row["설명"], f"input {i + 1}", ko) for i, row in cases.iterrows()]
    y_pos = list(range(len(cases)))[::-1]
    for y, (_, row) in zip(y_pos, cases.iterrows()):
        if row["처리"] == REJECT:
            ax.text(0.02, y, L("계산하지 않음", "not computed", ko), va="center",
                    fontsize=11, color=GRAY)
            continue
        ax.barh(y, row["확률"], color=colors[row["처리"]], height=0.6)
        inside = row["확률"] > 0.85          # 막대가 길면 숫자를 막대 안에 쓴다
        ax.text(row["확률"] - 0.02 if inside else row["확률"] + 0.02, y, f'{row["확률"]:.3f}',
                va="center", ha="right" if inside else "left", fontsize=11,
                color="white" if inside else "#333333")
    ax.axvline(0.5, color="#999999", linewidth=1, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(cases) - 0.4)
    ax.set_xlabel(L("대여가 많은 날일 확률", "probability of a busy day", ko), fontsize=11)
    ax.set_title(L("시험 입력 6건을 앱이 처리한 결과", "how the app handled the 6 test inputs", ko),
                 fontsize=13, pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in (OK, WARN, REJECT)]
    ax.legend(handles, [L(k, english[k], ko) for k in (OK, WARN, REJECT)], fontsize=9.5,
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=1)
    ax.scatter(p_model, p_app, s=14, color=GREEN, alpha=0.7)
    ax.set_xlabel(L("원래 모델이 계산한 확률", "probability from the original model", ko), fontsize=11)
    ax.set_ylabel(L("앱 파일의 숫자로 계산한 확률", "probability from the exported numbers", ko),
                  fontsize=11)
    ax.set_title(L(f"테스트 {len(p_model)}일: 두 계산이 같은가",
                   f"{len(p_model)} test days: do both agree?", ko), fontsize=13, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for ax in axes:
        ax.tick_params(labelsize=10, length=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(L("앱 점검 결과 (합성 데이터)", "app check (synthetic data)", ko), fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    model, x_train, x_test, y_train, y_test = _bikedata.train_model()
    print("=== 1. 모델 ===")
    print("데이터: 컴퓨터로 만든 합성 날씨 데이터(실제 대여 기록 아님)")
    print(f"난수 seed: {_bikedata.RANDOM_SEED}")
    print(f"훈련 {len(y_train)}일, 테스트 {len(y_test)}일")
    print(f"훈련 데이터 중 대여가 많은 날: {int(y_train.sum())}일")
    print(f"테스트 정확도: {accuracy_score(y_test, model.predict(x_test)):.3f}")
    print()

    spec = export_spec(model, x_train)
    print("=== 2. 앱 파일에 담는 숫자 ===")
    print_table(
        ["입력", "단위", "앱이 받는 범위", "모델이 배운 범위", "평균", "표준편차", "무게"],
        [[i["name"], i["unit"], f'{i["min"]:g}~{i["max"]:g}',
          f'{i["learned_min"]:g}~{i["learned_max"]:g}',
          f'{i["mean"]:.3f}', f'{i["scale"]:.3f}', f'{i["weight"]:+.3f}'] for i in spec["inputs"]],
        right={4, 5, 6},
    )
    print(f'절편: {spec["bias"]:+.3f}')
    print(f'담은 숫자: 입력 {len(spec["inputs"])}개 x 3 + 절편 1 = {len(spec["inputs"]) * 3 + 1}개')
    print()

    # 내보낸 숫자로 계산한 확률이 원래 모델과 같은지 테스트 데이터 전체로 확인한다
    p_model = model.predict_proba(x_test)[:, 1]
    p_app = np.array([app_answer(spec, list(row))["prob"] for row in x_test])
    gap = float(np.abs(p_model - p_app).max())
    print("=== 3. 내보낸 숫자 확인 ===")
    print(f"테스트 {len(y_test)}일에서 원래 모델과 앱 계산의 확률 차이 최댓값: {gap:.6f}")
    if gap > 0.001:
        raise RuntimeError("내보낸 숫자로 계산한 확률이 원래 모델과 다르다. 앱 파일을 만들지 않는다.")
    print("판정이 달라진 날: "
          f"{int(((p_model >= 0.5) != (p_app >= 0.5)).sum())}일")
    print()

    print("=== 4. 시험 입력 처리 ===")
    rows, records = [], []
    for number, (note, *values) in enumerate(MY_INPUTS, start=1):
        answer = app_answer(spec, values)
        shown = [f"{v:g}" for v in values]
        if answer["status"] == REJECT:
            rows.append([str(number), note, *shown, REJECT, "-", "-"])
            records.append({"번호": number, "설명": note, "처리": REJECT, "판정": "", "확률": np.nan,
                            "비고": answer["reason"]})
            continue
        rows.append([str(number), note, *shown, answer["status"], answer["label"],
                     f'{answer["prob"]:.3f}'])
        records.append({"번호": number, "설명": note, "처리": answer["status"],
                        "판정": answer["label"], "확률": round(answer["prob"], 3),
                        "비고": ", ".join(answer["outside"])})
    print_table(["번호", "설명", "기온", "강수량", "풍속", "처리", "판정", "확률"], rows,
                right={0, 2, 3, 4, 7})
    print()
    for record in records:
        if record["비고"]:
            print(f'{record["번호"]}번 {record["처리"]}: {record["비고"]}')
    print()

    # 근거 예시 한 건: 각 값이 어느 쪽으로 얼마나 밀었는지
    note, *values = MY_INPUTS[1]
    answer = app_answer(spec, values)
    print(f"=== 5. 근거 예시: 2번 {note} ===")
    for item, value, push in zip(spec["inputs"], values, answer["pushes"]):
        print(f'{item["name"]} {value:g}{item["unit"]}: {push:+.2f}')
    print(f'절편: {spec["bias"]:+.2f}')
    print(f'합계 {spec["bias"] + sum(answer["pushes"]):+.2f} -> 확률 {answer["prob"]:.3f}')
    print("(양수는 '대여가 많은 날' 쪽, 음수는 '대여가 적은 날' 쪽으로 민 크기다)")
    print()

    cases = pd.DataFrame(records)
    statuses = cases["처리"].value_counts()
    print("=== 6. 집계 ===")
    print(f"시험 입력 {len(cases)}건: {OK} {int(statuses.get(OK, 0))}건, "
          f"{WARN} {int(statuses.get(WARN, 0))}건, {REJECT} {int(statuses.get(REJECT, 0))}건")
    print()

    app_path = RESULTS_DIR / "my_app.html"
    write_app(spec, app_path)
    csv_path = RESULTS_DIR / "app_check.csv"
    cases.to_csv(csv_path, index=False, encoding="utf-8-sig")
    png_path = RESULTS_DIR / "app_check.png"
    draw_check(cases, p_model, p_app, ko, png_path)
    print(f"저장: results/{app_path.name}  ({app_path.stat().st_size / 1024:.1f} KB, 파일 1개)")
    print(f"저장: results/{csv_path.name}")
    print(f"저장: results/{png_path.name}")
    print()
    print("results/my_app.html 을 브라우저로 열면 앱이 실행된다. 인터넷 연결은 필요 없다.")


if __name__ == "__main__":
    main()
