"""실습 9.1 - 이미지를 숫자 64개로 펴서 분류하고, 그 방식의 한계를 측정한다.

하는 일은 넷이다.
  1. 손글씨 한 장의 8x8 픽셀값을 표로 찍는다. 이미지가 숫자라는 것을 확인한다.
  2. 8x8을 한 줄로 편 64개 숫자로 숫자 0~9를 분류한다. 틀린 이미지를 모아 실패 갤러리를 만든다.
  3. 64칸의 순서를 통째로 섞어 새로 학습한다. 정확도가 거의 그대로인지 본다.
     정확도가 그대로라면, 이 모델은 어느 칸이 어느 칸 옆에 있는지를 쓰지 않는다는 뜻이다.
  4. 학습이 끝난 모델에 테스트 이미지를 한두 칸 민 것을 넣는다. 정확도가 얼마나 떨어지는지 본다.

데이터는 scikit-learn에 들어 있는 load_digits()다. 내려받지 않고 바로 쓴다.
1797장의 8x8 손글씨 숫자이고, 픽셀값은 0(검정)부터 16(흰색)까지다.
합성 데이터가 아니라 실제로 사람이 쓴 숫자를 스캔해 만든 공개 데이터다.

torch나 tensorflow를 쓰지 않는다. scikit-learn의 MLPClassifier로 학습한다.
이 모델은 CNN이 아니다. 합성곱 층이 없다.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.

SAMPLE_INDEX = 17      # 픽셀 표로 볼 손글씨 번호. 0부터 1796까지 넣을 수 있다.
HIDDEN_SIZE = 32       # 은닉층 뉴런 수. 크게 하면 정확도가 오르지만 느려진다.
MAX_ITER = 60          # 학습을 몇 바퀴 돌 것인가. 작게 두어 몇 초 안에 끝나게 했다.
TEST_RATIO = 0.3       # 테스트로 떼어 둘 비율
GALLERY_SIZE = 12      # 실패 갤러리에 넣을 이미지 장수
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

# 테스트 이미지를 미는 방향과 칸 수. (이름, 가로 이동, 세로 이동)
# 가로 이동이 양수면 오른쪽, 세로 이동이 양수면 아래쪽이다.
SHIFTS = [
    ("오른쪽 1칸", "right 1", 1, 0),
    ("왼쪽 1칸", "left 1", -1, 0),
    ("아래 1칸", "down 1", 0, 1),
    ("오른쪽 2칸", "right 2", 2, 0),
]

# 평탄화 그림에서 강조할 두 칸 (행, 열). 위아래로 붙어 있는 칸이다.
UPPER_CELL = (4, 2)
LOWER_CELL = (5, 2)

ORANGE = "#F58518"
BLUE = "#4C78A8"
RED = "#C0392B"
GRAY = "#79706E"


def shift_images(images: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """이미지를 가로 dx칸, 세로 dy칸 민다. 비는 칸은 0(검정)으로 채운다.

    밀려서 이미지 밖으로 나간 칸은 버린다. load_digits()는 가장자리가 대부분 0이라서
    한두 칸 밀어도 획이 잘려 나가는 경우는 드물다.
    """
    out = np.zeros_like(images)
    h, w = images.shape[1], images.shape[2]
    src_r = slice(max(0, -dy), h - max(0, dy))
    dst_r = slice(max(0, dy), h - max(0, -dy))
    src_c = slice(max(0, -dx), w - max(0, dx))
    dst_c = slice(max(0, dx), w - max(0, -dx))
    out[:, dst_r, dst_c] = images[:, src_r, src_c]
    return out


def flatten(images: np.ndarray) -> np.ndarray:
    """8x8 이미지를 64개 숫자 한 줄로 펴고, 0~16을 0~1로 줄인다."""
    return images.reshape(len(images), -1) / 16.0


def print_pixel_table(image: np.ndarray) -> None:
    """8x8 픽셀값을 표로 찍는다."""
    header = "     " + "".join(f"{c:>5}" for c in range(image.shape[1]))
    print(header)
    print("     " + "-" * (5 * image.shape[1]))
    for r in range(image.shape[0]):
        row = "".join(f"{image[r, c]:>5.0f}" for c in range(image.shape[1]))
        print(f"{r:>3} |{row}")


def new_model() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(HIDDEN_SIZE,),
        max_iter=MAX_ITER,
        random_state=RANDOM_SEED,
    )


def figure_pixels(image: np.ndarray, digit_label: int, ko: bool,
                  out_path: Path) -> None:
    """왼쪽에 흑백 그림, 오른쪽에 같은 값의 숫자 표를 나란히 그린다."""
    L = _krfont.label
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))

    ax = axes[0]
    ax.imshow(image, cmap="gray", vmin=0, vmax=16)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.tick_params(labelsize=9)
    ax.set_title(
        L(f"사람 눈에 보이는 그림 (정답 {digit_label})",
          f"What we see (label {digit_label})", ko),
        fontsize=13, pad=10,
    )

    ax = axes[1]
    ax.imshow(image, cmap="gray", vmin=0, vmax=16)
    for r in range(8):
        for c in range(8):
            value = image[r, c]
            ax.text(
                c, r, f"{value:.0f}",
                ha="center", va="center", fontsize=10,
                color="#111111" if value > 8 else "#FFFFFF",
            )
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.tick_params(labelsize=9)
    ax.set_title(
        L("컴퓨터가 받는 숫자 (0~16)", "What the computer gets (0-16)", ko),
        fontsize=13, pad=10,
    )

    fig.suptitle(
        L("이미지는 숫자다 - 같은 자료를 두 가지로 보여준 것",
          "An image is numbers - the same data shown two ways", ko),
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_flatten(image: np.ndarray, ko: bool, out_path: Path) -> None:
    """8x8을 64칸 한 줄로 펴면, 위아래로 붙은 두 칸이 8칸 떨어진다는 것을 그린다."""
    L = _krfont.label
    fig = plt.figure(figsize=(12.0, 6.4))
    ax_img = fig.add_axes([0.33, 0.40, 0.34, 0.52])
    ax_row = fig.add_axes([0.03, 0.10, 0.94, 0.12])

    (ur, uc), (lr, lc) = UPPER_CELL, LOWER_CELL
    ax_img.imshow(image, cmap="gray", vmin=0, vmax=16)
    for r in range(8):
        for c in range(8):
            v = image[r, c]
            ax_img.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="#111111" if v > 8 else "#FFFFFF")
    for (r, c), color in [((ur, uc), ORANGE), ((lr, lc), BLUE)]:
        ax_img.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor=color, linewidth=3))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(
        L("8 x 8 그림: 주황 칸과 파랑 칸은 위아래로 붙어 있다",
          "8 x 8 image: the orange and blue cells touch vertically", ko),
        fontsize=12, pad=8,
    )

    flat = image.reshape(-1)
    ax_row.imshow(flat.reshape(1, -1), cmap="gray", vmin=0, vmax=16, aspect="auto")
    i_up = ur * 8 + uc
    i_low = lr * 8 + lc
    for i, color in [(i_up, ORANGE), (i_low, BLUE)]:
        ax_row.add_patch(Rectangle((i - 0.5, -0.5), 1, 1, fill=False,
                                   edgecolor=color, linewidth=3))
    for k in range(1, 8):
        ax_row.axvline(k * 8 - 0.5, color="#E45756", linewidth=1.0, alpha=0.7)
    ax_row.set_yticks([])
    ax_row.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 63])
    ax_row.tick_params(labelsize=9)
    ax_row.set_xlabel(
        L("한 줄로 편 64개 숫자 (빨간 세로선은 원래 줄이 바뀌던 곳)",
          "The 64 numbers in one row (red lines mark where image rows ended)", ko),
        fontsize=11,
    )
    ax_row.annotate(
        "", xy=(i_low, 1.25), xytext=(i_up, 1.25),
        xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="<->", color=GRAY, linewidth=1.6),
    )
    ax_row.text(
        (i_up + i_low) / 2, 1.55,
        L(f"{i_up}번 칸과 {i_low}번 칸: {i_low - i_up}칸 떨어졌다",
          f"cells {i_up} and {i_low}: {i_low - i_up} apart", ko),
        ha="center", fontsize=11, color="#333333",
        transform=ax_row.get_xaxis_transform(),
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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
                text, color = str(count), RED
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
            fontsize=12, color=RED, pad=6,
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


def figure_shift_test(example: np.ndarray, example_shifted: np.ndarray,
                      rows: list, baseline_acc: float, ko: bool,
                      out_path: Path) -> None:
    """왼쪽에 원본과 1칸 민 이미지, 오른쪽에 조건별 테스트 정확도 막대를 그린다."""
    L = _krfont.label
    fig = plt.figure(figsize=(12.0, 4.8))
    ax_a = fig.add_axes([0.02, 0.14, 0.15, 0.66])
    ax_b = fig.add_axes([0.19, 0.14, 0.15, 0.66])
    ax_bar = fig.add_axes([0.56, 0.16, 0.41, 0.66])

    ax_a.imshow(example, cmap="gray", vmin=0, vmax=16)
    ax_a.set_title(L("원본", "original", ko), fontsize=12)
    ax_b.imshow(example_shifted, cmap="gray", vmin=0, vmax=16)
    ax_b.set_title(L("오른쪽으로 1칸", "shifted right by 1", ko), fontsize=12)
    for ax in (ax_a, ax_b):
        ax.set_xticks([])
        ax.set_yticks([])

    names = [L(r[0], r[1], ko) for r in rows]
    accs = [r[2] for r in rows]
    colors = [BLUE, BLUE] + [RED] * (len(rows) - 2)
    y_pos = np.arange(len(rows))[::-1]
    ax_bar.barh(y_pos, accs, color=colors, height=0.6)
    for y, a in zip(y_pos, accs):
        ax_bar.text(a + 0.01, y, f"{a:.3f}", va="center", fontsize=10.5)
    ax_bar.axvline(baseline_acc, color=GRAY, linestyle="--", linewidth=1.2)
    ax_bar.text(baseline_acc + 0.01, y_pos[0] + 0.5,
                L(f"기준선 {baseline_acc:.3f}", f"baseline {baseline_acc:.3f}", ko),
                fontsize=9.5, color=GRAY)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=10.5)
    ax_bar.set_xlim(0, 1.12)
    ax_bar.set_ylim(-0.6, len(rows) - 0.1)
    ax_bar.set_xlabel(L("테스트 정확도", "test accuracy", ko), fontsize=11)
    ax_bar.set_title(
        L("파랑: 칸 순서만 바꿈 / 빨강: 그림을 밀었음",
          "blue: only the cell order / red: the picture is shifted", ko),
        fontsize=11.5,
    )
    fig.suptitle(
        L("64개 숫자로 편 모델은 칸 순서는 상관없고, 위치가 밀리면 무너진다",
          "A flattened model ignores cell order but breaks when the picture shifts", ko),
        fontsize=13,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    digits = load_digits()
    images = digits.images          # (1797, 8, 8)
    y = digits.target

    print("=== 데이터 ===")
    print("scikit-learn load_digits() - 실제 손글씨를 스캔해 만든 공개 데이터")
    print(f"이미지 장수: {len(images)}장")
    print(f"이미지 한 장의 크기: {images.shape[1]} x {images.shape[2]} 픽셀")
    print("채널 수: 1 (흑백이므로 숫자 판이 한 장이다)")
    print(f"픽셀값 범위: {images.min():.0f} ~ {images.max():.0f}")
    print(f"이미지 한 장이 숫자 {images.shape[1] * images.shape[2]}개다")
    print()

    # ------------------------------------------------------------------
    # 1. 이미지 한 장을 숫자로 본다
    index = SAMPLE_INDEX % len(images)
    image = images[index]
    digit_label = int(y[index])
    print("=== 고른 이미지 ===")
    print(f"SAMPLE_INDEX = {SAMPLE_INDEX} (전체 {len(images)}장 중 {index}번)")
    print(f"이 이미지의 정답: {digit_label}")
    print()
    print("=== 8x8 픽셀값 ===")
    print_pixel_table(image)
    print()
    print("0이 검정, 16이 흰색이다. 위 숫자가 이미지의 전부다.")
    print()

    (ur, uc), (lr, lc) = UPPER_CELL, LOWER_CELL
    print("=== 한 줄로 펴면 이웃이 어디로 가는가 ===")
    print(f"그림에서 위아래로 붙은 두 칸: ({ur}행 {uc}열, 값 {image[ur, uc]:.0f})"
          f"과 ({lr}행 {lc}열, 값 {image[lr, lc]:.0f})")
    print(f"한 줄로 편 뒤의 자리: {ur * 8 + uc}번과 {lr * 8 + lc}번"
          f" -> {(lr * 8 + lc) - (ur * 8 + uc)}칸 떨어진다")
    print()

    # ------------------------------------------------------------------
    # 2. 64개 숫자로 분류한다
    x = flatten(images)
    idx_train, idx_test = train_test_split(
        np.arange(len(images)),
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    x_train, x_test = x[idx_train], x[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    img_test = images[idx_test]

    print("=== 분류 데이터 ===")
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
    print(f"첫 층의 가중치 수: 64 x {HIDDEN_SIZE} = {64 * HIDDEN_SIZE}개")
    print()

    model = new_model()
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

    # 학습이 끝난 모델에서 층마다 가중치 행렬의 모양을 직접 꺼내 본다.
    print("=== 학습한 가중치의 모양 ===")
    print(f"hidden_layer_sizes = {model.hidden_layer_sizes}")
    layer_sizes = [x.shape[1]] + list(model.hidden_layer_sizes) + [model.n_outputs_]
    print("층별 뉴런 수: " + " -> ".join(str(n) for n in layer_sizes)
          + "  (입력 -> 은닉층 -> 출력)")
    n_weights = 0
    for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_), start=1):
        print(f"{i}번째 연결: 가중치 {w.shape[0]} x {w.shape[1]} = {w.size}개, "
              f"편향 {b.size}개")
        n_weights += w.size + b.size
    print(f"가중치와 편향을 합한 전체 개수: {n_weights}개")
    print()

    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    print("=== 혼동행렬 (가로: 예측, 세로: 정답) ===")
    print("정답\\예측" + "".join(f"{c:>5}" for c in range(10)))
    for r in range(10):
        print(f"{r:>8}" + "".join(f"{cm[r, c]:>5}" for c in range(10)))
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

    # ------------------------------------------------------------------
    # 3. 64칸의 순서를 섞어 새로 학습한다
    perm = np.random.default_rng(RANDOM_SEED).permutation(x.shape[1])
    model_perm = new_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_perm.fit(x_train[:, perm], y_train)
    perm_acc = accuracy_score(y_test, model_perm.predict(x_test[:, perm]))

    print("=== 실험 1. 64칸의 순서를 섞어 새로 학습 ===")
    print("모든 이미지에 같은 순서 섞기를 적용한다. 훈련과 테스트 모두 같은 순서다.")
    print(f"섞은 순서의 앞 10칸: {perm[:10].tolist()}")
    print(f"{'원래 순서로 학습':<20}{test_acc:>8.3f}")
    print(f"{'섞은 순서로 학습':<20}{perm_acc:>8.3f}")
    print(f"차이: {perm_acc - test_acc:+.3f}")
    print("수렴 경고는 위 원래 모델과 같은 이유로 나오므로 여기서는 다시 적지 않는다.")
    print()

    # ------------------------------------------------------------------
    # 4. 학습이 끝난 모델에 민 이미지를 넣는다
    print("=== 실험 2. 테스트 이미지를 밀어서 넣기 (다시 학습하지 않는다) ===")
    print(f"{'조건':<14}{'테스트 정확도':>12}")
    print(f"{'원본':<14}{test_acc:>12.3f}")
    shift_rows = []
    for name_ko, name_en, dx, dy in SHIFTS:
        shifted = shift_images(img_test, dx, dy)
        acc = accuracy_score(y_test, model.predict(flatten(shifted)))
        shift_rows.append((name_ko, name_en, acc))
        print(f"{name_ko:<14}{acc:>12.3f}")
    print(f"기준선: {baseline_acc:.3f}")
    print()
    lost = (img_test.sum(axis=(1, 2)) - shift_images(img_test, 1, 0).sum(axis=(1, 2)))
    print(f"오른쪽 1칸 밀 때 맨 오른쪽 열이 잘려 나간 밝기 합: "
          f"이미지 한 장 평균 {lost.mean():.1f} (전체 밝기 합 평균 "
          f"{img_test.sum(axis=(1, 2)).mean():.1f})")
    print()

    # ------------------------------------------------------------------
    # 결과 저장
    wrong_table = pd.DataFrame({
        "테스트순번": wrong_idx,
        "정답": y_test[wrong_idx],
        "모델예측": y_pred[wrong_idx],
    })
    csv_path = RESULTS_DIR / "misclassified_digits.csv"
    wrong_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    pixels_path = RESULTS_DIR / "digit_pixels.png"
    flatten_path = RESULTS_DIR / "flatten_neighbors.png"
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    gallery_path = RESULTS_DIR / "failure_gallery.png"
    shift_path = RESULTS_DIR / "shift_test.png"

    figure_pixels(image, digit_label, ko, pixels_path)
    figure_flatten(image, ko, flatten_path)
    figure_confusion(cm, ko, cm_path)
    n_drawn = figure_failure_gallery(img_test, y_test, y_pred, wrong_idx, ko, gallery_path)
    bar_rows = [("원래 순서", "original order", test_acc),
                ("순서 섞어 새로 학습", "shuffled order, retrained", perm_acc)]
    bar_rows += [(f"{name_ko} 밀기", f"shift {name_en}", acc)
                 for name_ko, name_en, acc in shift_rows]
    figure_shift_test(image, shift_images(image[None], 1, 0)[0], bar_rows,
                      baseline_acc, ko, shift_path)

    print("=== 저장된 결과 ===")
    print(f"픽셀값 그림: results/{pixels_path.name}")
    print(f"평탄화 그림: results/{flatten_path.name}")
    print(f"혼동행렬 그림: results/{cm_path.name}")
    print(f"실패 갤러리({n_drawn}장): results/{gallery_path.name}")
    print(f"밀기 실험 그림: results/{shift_path.name}")
    print(f"틀린 목록: results/{csv_path.name}")
    print()
    print("HIDDEN_SIZE와 MAX_ITER를 바꿔 다시 실행하면 정확도와 실패 갤러리가 달라진다.")


if __name__ == "__main__":
    main()
