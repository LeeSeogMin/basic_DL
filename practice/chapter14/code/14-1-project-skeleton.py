"""실습 14.1 - 기말 프로젝트의 뼈대를 한 파일로 만든다.

4장부터 배운 것을 한 줄로 이어 붙인 최소 파이프라인이다.

  데이터 불러오기 -> 훈련/테스트 나누기 -> 기준선 만들기
  -> 모델 학습 -> 정확도와 혼동행렬 -> 틀린 사례 저장

학생은 `load_data()` 하나만 자기 데이터로 바꿔 끼우면 된다.
아무것도 바꾸지 않으면 합성 데이터로 그대로 실행된다.

주의: 기본 데이터는 실제 리뷰가 아니라 컴퓨터로 만든 합성 데이터다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.
DATA_SOURCE = "synthetic"   # "synthetic" 이면 합성 데이터, "csv" 이면 내 파일
CSV_PATH = "my_project_data.csv"   # DATA_SOURCE = "csv" 일 때 읽을 파일
LABEL_COLUMN = "평점"        # 정답이 들어 있는 열 이름
N_SAMPLES = 300             # 합성 데이터일 때 만들 줄 수
TEST_RATIO = 0.3            # 테스트로 떼어 둘 비율
HIDDEN_UNITS = 8            # 작은 신경망의 은닉층 뉴런 수
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GRAY = "#79706E"


# ===========================================================================
# 1단계. 데이터 불러오기 - 자기 프로젝트로 바꿀 때 이 함수만 고친다
# ===========================================================================
def load_data() -> tuple[pd.DataFrame, pd.Series, str]:
    """특징 표와 정답을 돌려준다.

    Returns:
        (특징 DataFrame, 정답 Series, 데이터 출처 설명 한 줄)
    """
    if DATA_SOURCE == "csv":
        return load_from_csv()
    return make_synthetic_reviews()


def load_from_csv() -> tuple[pd.DataFrame, pd.Series, str]:
    """내 CSV 파일을 읽는다. 숫자 열을 특징으로, LABEL_COLUMN 을 정답으로 쓴다."""
    path = Path(CSV_PATH)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / CSV_PATH
    if not path.exists():
        raise SystemExit(
            f"[중단] CSV 파일을 찾지 못했습니다: {path}\n"
            f"      CSV_PATH 를 고치거나 DATA_SOURCE 를 \"synthetic\" 으로 되돌리세요."
        )

    table = pd.read_csv(path)
    if LABEL_COLUMN not in table.columns:
        raise SystemExit(
            f"[중단] 정답 열 \"{LABEL_COLUMN}\" 이 파일에 없습니다.\n"
            f"      파일의 열 이름: {list(table.columns)}"
        )

    y = table[LABEL_COLUMN]
    x = table.drop(columns=[LABEL_COLUMN]).select_dtypes(include="number")
    if x.shape[1] == 0:
        raise SystemExit("[중단] 숫자로 된 특징 열이 하나도 없습니다.")
    return x, y, f"내 CSV 파일({path.name})"


def make_synthetic_reviews() -> tuple[pd.DataFrame, pd.Series, str]:
    """합성 리뷰 데이터를 만든다.

    특징은 네 개다. 긍정어와 부정어의 개수가 평점을 대부분 정하고,
    느낌표는 조금 거들며, 글자수는 평점과 아무 관계가 없다.
    글자수를 일부러 섞은 이유는 쓸모없는 특징이 섞였을 때
    모델이 어떻게 되는지 확인하기 위해서다.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = N_SAMPLES

    positive_words = rng.poisson(1.2, n)
    negative_words = rng.poisson(1.2, n)
    exclamations = rng.poisson(0.8, n)
    length = rng.integers(20, 300, n)

    hidden = (
        1.1 * positive_words
        - 1.1 * negative_words
        + 0.25 * exclamations
        + rng.normal(0.0, 1.0, n)
    )
    label = np.where(hidden > 0, "좋음", "나쁨")

    x = pd.DataFrame({
        "긍정어수": positive_words,
        "부정어수": negative_words,
        "느낌표수": exclamations,
        "글자수": length,
    })
    y = pd.Series(label, name=LABEL_COLUMN)
    return x, y, "컴퓨터로 만든 합성 데이터(실제 리뷰 아님)"


# ===========================================================================
# 그림 그리기
# ===========================================================================
def draw_scores(ax, names: list[str], scores: list[float], ko: bool) -> None:
    """세 방법의 테스트 정확도를 막대로 그린다."""
    L = _krfont.label
    colors = [GRAY, BLUE, GREEN]
    bars = ax.bar(range(len(names)), scores, color=colors, width=0.58)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.02,
                f"{score:.3f}", ha="center", fontsize=12, color="#222222")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel(L("테스트 정확도", "test accuracy", ko), fontsize=11)
    ax.set_title(L("기준선과 견주기", "Compare against the baseline", ko),
                 fontsize=12.5, pad=10)
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_confusion(ax, matrix: np.ndarray, class_names: list[str],
                   title: str, ko: bool) -> None:
    """혼동행렬 한 장을 그린다. 맞힌 칸은 초록, 틀린 칸은 빨강이다."""
    pred_word = _krfont.label("예측", "pred", ko)
    true_word = _krfont.label("정답", "true", ko)
    size = len(class_names)
    for row in range(size):
        for col in range(size):
            color = GREEN if row == col else RED
            ax.add_patch(plt.Rectangle((col, size - 1 - row), 1, 1,
                                       facecolor=color, alpha=0.85))
            ax.text(col + 0.5, size - 0.5 - row, str(matrix[row, col]),
                    ha="center", va="center", fontsize=21, color="white")

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(size)])
    ax.set_yticks([size - 0.5 - i for i in range(size)])
    ax.set_xticklabels([f"{pred_word}: {name}" for name in class_names], fontsize=10)
    ax.set_yticklabels([f"{true_word}: {name}" for name in class_names], fontsize=10)
    ax.set_title(title, fontsize=12.5, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def safe_name(name: str, index: int, ko: bool) -> str:
    """한글 폰트가 없는 환경에서 한글 클래스 이름이 깨지지 않게 바꾼다."""
    if ko or name.isascii():
        return name
    return f"class{index + 1}"


def save_figure(
    ko: bool,
    scores: list[float],
    matrix: np.ndarray,
    class_names: list[str],
    model_acc: float,
    out_path: Path,
) -> None:
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    method_names = [
        L("기준선", "baseline", ko),
        L("로지스틱 회귀", "logistic", ko),
        L(f"신경망({HIDDEN_UNITS})", f"network({HIDDEN_UNITS})", ko),
    ]
    draw_scores(axes[0], method_names, scores, ko)
    draw_confusion(
        axes[1], matrix,
        [safe_name(name, i, ko) for i, name in enumerate(class_names)],
        L(f"작은 신경망의 혼동행렬 (정확도 {model_acc:.3f})",
          f"Confusion matrix, small network (acc {model_acc:.3f})", ko),
        ko,
    )

    fig.suptitle(
        L("실습 14.1 프로젝트 뼈대 — 기준선, 정확도, 혼동행렬을 한 번에",
          "Practice 14.1 project skeleton - baseline, accuracy, confusion matrix", ko),
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ===========================================================================
# 본체
# ===========================================================================
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()

    # 1단계. 데이터 불러오기
    x, y, source_note = load_data()
    class_names = sorted(y.unique().tolist())
    feature_names = list(x.columns)

    # 2단계. 훈련/테스트 나누기
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y,
    )

    # 3단계. 기준선 - 훈련 데이터에서 가장 많았던 쪽으로만 답한다
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(x_test))

    # 4단계. 모델 학습
    # StandardScaler 는 열마다 다른 숫자 크기를 비슷하게 맞춘다.
    # 글자수(20~300)와 느낌표수(0~5)를 그대로 넣으면 큰 쪽이 계산을 지배한다.
    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=RANDOM_SEED),
    )
    linear.fit(x_train, y_train)
    linear_train_acc = accuracy_score(y_train, linear.predict(x_train))
    linear_acc = accuracy_score(y_test, linear.predict(x_test))

    network = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(HIDDEN_UNITS,),
            max_iter=3000,
            random_state=RANDOM_SEED,
        ),
    )
    network.fit(x_train, y_train)
    network_train_acc = accuracy_score(y_train, network.predict(x_train))
    network_pred = network.predict(x_test)
    network_acc = accuracy_score(y_test, network_pred)

    # 5단계. 정확도와 혼동행렬
    matrix = confusion_matrix(y_test, network_pred, labels=class_names)

    # 6단계. 틀린 사례 저장
    wrong_mask = network_pred != y_test.to_numpy()
    wrong = x_test[wrong_mask].copy()
    wrong["실제"] = y_test.to_numpy()[wrong_mask]
    wrong["모델예측"] = network_pred[wrong_mask]
    wrong = wrong.reset_index(drop=True).head(5)
    wrong_path = RESULTS_DIR / "project_wrong_examples.csv"
    wrong.to_csv(wrong_path, index=False, encoding="utf-8-sig")

    # 학생이 자기 CSV를 만들 때 참고할 모양을 남긴다
    sample = x.copy()
    sample[LABEL_COLUMN] = y
    sample_path = RESULTS_DIR / "project_input_example.csv"
    sample.head(10).to_csv(sample_path, index=False, encoding="utf-8-sig")

    plot_path = RESULTS_DIR / "project_result.png"
    save_figure(
        ko,
        [baseline_acc, linear_acc, network_acc],
        matrix, class_names, network_acc, plot_path,
    )

    # ------------------------------------------------------------------
    print("=== 실습 14.1 프로젝트 뼈대 ===")
    _krfont.report(ko)
    print(f"데이터: {source_note}")
    print(f"전체 {len(x)}줄, 특징 {len(feature_names)}개: {', '.join(feature_names)}")
    print(f"정답 열 이름: {LABEL_COLUMN}, 값의 종류: {', '.join(class_names)}")
    counts = y.value_counts().sort_index()
    print("정답 분포: " + ", ".join(f"{name} {int(counts[name])}줄" for name in class_names))
    print(f"훈련 {len(x_train)}줄 + 테스트 {len(x_test)}줄 (테스트 비율 {TEST_RATIO})")
    print()

    print("=== 세 방법의 정확도 ===")
    print(f"{'방법':<16}{'훈련':>10}{'테스트':>10}")
    print(f"{'기준선':<16}{'-':>10}{baseline_acc:>10.3f}")
    print(f"{'로지스틱 회귀':<16}{linear_train_acc:>10.3f}{linear_acc:>10.3f}")
    print(f"{f'신경망({HIDDEN_UNITS})':<16}{network_train_acc:>10.3f}{network_acc:>10.3f}")
    print()

    print("=== 작은 신경망의 혼동행렬 ===")
    header = "".join(f"{'예측:' + name:>12}" for name in class_names)
    print(" " * 12 + header)
    for i, name in enumerate(class_names):
        row = "".join(f"{matrix[i, j]:>12}" for j in range(len(class_names)))
        print(f"{'정답:' + name:<12}" + row)
    print()

    print("=== 모델이 틀린 사례 (최대 5줄) ===")
    if wrong.empty:
        print("틀린 사례가 없습니다.")
    else:
        print(wrong.to_string(index=False))
    print()

    print("=== 저장된 결과 ===")
    print(f"그림: results/{plot_path.name}")
    print(f"틀린 사례: results/{wrong_path.name}")
    print(f"입력 예시: results/{sample_path.name}")
    print()
    print("자기 데이터로 바꾸려면 CSV_PATH 에 파일 이름을 적고 "
          "DATA_SOURCE 를 \"csv\" 로 바꾼다.")
    print("정답 열 이름은 LABEL_COLUMN 에 적는다. 나머지 숫자 열이 특징이 된다.")


if __name__ == "__main__":
    main()
