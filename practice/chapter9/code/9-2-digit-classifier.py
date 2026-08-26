"""실습 9.2 - 손글씨 숫자 분류기를 만들고, 틀린 이미지를 모아 실패 갤러리를 만든다.

데이터는 실습 9.1과 같은 load_digits()다. 8x8 손글씨 숫자 1797장이다.
8x8 = 64개의 픽셀값을 한 줄로 펴서 신경망에 넣는다. 이 방식의 한계는 강의자료 3절에 있다.

torch나 tensorflow를 쓰지 않는다. scikit-learn의 MLPClassifier로 학습한다.
CNN을 직접 학습시키지 않는다. 이 실습의 목표는 "합성곱 없이도 어디까지 되는가"와
"어디서 틀리는가"를 확인하는 것이다.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.

HIDDEN_SIZE = 32       # 은닉층 뉴런 수. 크게 하면 정확도가 오르지만 느려진다.
MAX_ITER = 60          # 학습을 몇 바퀴 돌 것인가. 작게 두어 몇 초 안에 끝나게 했다.
TEST_RATIO = 0.3       # 테스트로 떼어 둘 비율
GALLERY_SIZE = 12      # 실패 갤러리에 넣을 이미지 장수
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"


def figure_confusion(cm: np.ndarray, ko: bool, out_path: Path) -> None:
    """혼동행렬 10x10을 그린다."""
    L = _krfont.label
    fig, ax = plt.subplots(figsize=(7.6, 6.6))

    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            count = int(cm[r, c])
            if count == 0:
                text, color = ".", "#BBBBBB"
            elif r == c:
                text, color = str(count), "#FFFFFF"
            else:
                text, color = str(count), "#C0392B"
            ax.text(c, r, text, ha="center", va="center", fontsize=10, color=color)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel(L("모델의 예측", "predicted", ko), fontsize=12)
    ax.set_ylabel(L("실제 정답", "true label", ko), fontsize=12)
    ax.set_title(
        L("혼동행렬 - 대각선은 맞힌 것, 나머지는 틀린 것",
          "Confusion matrix - diagonal is correct, the rest is wrong", ko),
        fontsize=13, pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_failure_gallery(images: np.ndarray, y_true: np.ndarray,
                           y_pred: np.ndarray, wrong_idx: np.ndarray,
                           ko: bool, out_path: Path) -> int:
    """모델이 틀린 이미지를 모아 갤러리로 만든다. 실제로 그린 장수를 돌려준다."""
    L = _krfont.label
    n_show = min(GALLERY_SIZE, len(wrong_idx))
    if n_show == 0:
        return 0

    ncols = 4
    nrows = int(np.ceil(n_show / ncols))
    # 이미지 축은 정사각형이라 tight_layout이 잡은 자리보다 위아래로 더 차지한다.
    # 그대로 두면 아랫줄 제목이 윗줄 그림에 겹친다. 줄 간격을 직접 벌린다.
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for slot, ax in enumerate(axes):
        if slot >= n_show:
            ax.axis("off")
            continue
        i = wrong_idx[slot]
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=16)
        ax.set_title(
            L(f"정답 {y_true[i]} → 예측 {y_pred[i]}",
              f"true {y_true[i]} -> pred {y_pred[i]}", ko),
            fontsize=12, color="#C0392B", pad=6,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        L("실패 갤러리 - 모델이 틀린 이미지", "Failure gallery - what the model got wrong", ko),
        fontsize=14,
    )
    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.88, bottom=0.03, wspace=0.15, hspace=0.35
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return n_show


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    digits = load_digits()
    images = digits.images
    x = digits.data / 16.0          # 0~16을 0~1로 줄인다. 학습이 안정된다.
    y = digits.target

    x_train, x_test, y_train, y_test, _img_train, img_test = train_test_split(
        x, y, images,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print("=== 데이터 ===")
    print("scikit-learn load_digits() - 실제 손글씨를 스캔해 만든 공개 데이터")
    print(f"전체 {len(x)}장 = 훈련 {len(x_train)}장 + 테스트 {len(x_test)}장")
    print(f"입력: 8x8 이미지를 한 줄로 펴서 숫자 {x.shape[1]}개")
    print(f"정답 종류: {len(np.unique(y))}가지 (숫자 0~9)")
    print()

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(x_test))

    print("=== 모델 설정 ===")
    print(f"은닉층 뉴런 수: {HIDDEN_SIZE}")
    print(f"학습 반복 횟수(max_iter): {MAX_ITER}")
    print(f"난수 seed: {RANDOM_SEED}")
    print()

    model = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_SIZE,),
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
    )

    # max_iter를 작게 잡았으므로 수렴 경고가 뜰 수 있다.
    # 경고를 없애지 않고 붙잡아서 로그에 그대로 적는다.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)

    print("=== 학습 중 나온 경고 ===")
    if caught:
        for w in caught:
            print(f"[{w.category.__name__}] {str(w.message).strip()}")
        print()
        print("max_iter를 작게 잡아서 나온 경고다. 학습을 더 돌리면 정확도가 조금 더 오른다.")
        print("이 실습은 수업 시간 안에 끝나는 것을 우선해 경고를 감수한다.")
    else:
        print("경고 없음")
    print()

    y_pred = model.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, model.predict(x_train))

    print("=== 정확도 ===")
    print(f"{'기준선(가장 많은 숫자로만 답하기)':<28}{baseline_acc:>8.3f}")
    print(f"{'훈련 데이터':<28}{train_acc:>8.3f}")
    print(f"{'테스트 데이터':<28}{test_acc:>8.3f}")
    print(f"학습 반복 실제 횟수: {model.n_iter_}회")
    print()

    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))

    print("=== 혼동행렬 (가로: 예측, 세로: 정답) ===")
    header = "정답\\예측" + "".join(f"{c:>5}" for c in range(10))
    print(header)
    for r in range(10):
        row = "".join(f"{cm[r, c]:>5}" for c in range(10))
        print(f"{r:>8}{row}")
    print()

    # 대각선 밖에서 큰 값을 찾는다. 자주 헷갈린 짝이다.
    pairs = []
    for r in range(10):
        for c in range(10):
            if r != c and cm[r, c] > 0:
                pairs.append((int(cm[r, c]), r, c))
    pairs.sort(reverse=True)

    print("=== 자주 헷갈린 짝 ===")
    if pairs:
        print(f"{'정답':>5}{'예측':>7}{'건수':>7}")
        for count, r, c in pairs[:6]:
            print(f"{r:>5}{c:>7}{count:>7}")
    else:
        print("틀린 사례가 없다.")
    print()

    wrong_idx = np.where(y_pred != y_test)[0]
    print(f"테스트 {len(y_test)}장 중 틀린 것 {len(wrong_idx)}장")
    print()

    wrong_table = pd.DataFrame({
        "테스트순번": wrong_idx,
        "정답": y_test[wrong_idx],
        "모델예측": y_pred[wrong_idx],
    })
    csv_path = RESULTS_DIR / "misclassified_digits.csv"
    wrong_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    cm_path = RESULTS_DIR / "confusion_matrix.png"
    gallery_path = RESULTS_DIR / "failure_gallery.png"
    figure_confusion(cm, ko, cm_path)
    n_drawn = figure_failure_gallery(
        img_test, y_test, y_pred, wrong_idx, ko, gallery_path
    )

    print("=== 저장된 결과 ===")
    print(f"혼동행렬 그림: results/{cm_path.name}")
    print(f"실패 갤러리({n_drawn}장): results/{gallery_path.name}")
    print(f"틀린 목록: results/{csv_path.name}")
    print()
    print("HIDDEN_SIZE와 MAX_ITER를 바꿔 다시 실행하면 정확도와 실패 갤러리가 달라진다.")


if __name__ == "__main__":
    main()
