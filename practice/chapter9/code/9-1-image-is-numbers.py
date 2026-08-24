"""실습 9.1 - 이미지가 숫자라는 것을 눈으로 확인한다.

손글씨 숫자 한 장을 골라 8x8 픽셀값을 표로 찍고, 같은 값을 흑백 그림으로 나란히 본다.
그다음 필터 네 개를 numpy로 직접 만들어 합성곱을 적용하고, 필터마다 무엇이 도드라지는지 본다.

데이터는 scikit-learn에 들어 있는 load_digits()다. 내려받지 않고 바로 쓴다.
1797장의 8x8 손글씨 숫자이고, 픽셀값은 0(검정)부터 16(흰색)까지다.
합성 데이터가 아니라 실제로 사람이 쓴 숫자를 스캔해 만든 공개 데이터다.

torch나 tensorflow를 쓰지 않는다. 합성곱은 아래 convolve2d()에서 직접 계산한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

import _krfont

# ---------------------------------------------------------------------------
# 여기를 바꿔가며 실습한다.

# 몇 번째 손글씨를 볼 것인가. 0부터 1796까지 넣을 수 있다.
SAMPLE_INDEX = 17

# 내가 만드는 필터. 3x3 숫자를 마음대로 바꿔 본다.
# 아래 값은 가운데를 크게, 주변을 작게 만들어 테두리를 도드라지게 하는 필터다.
MY_FILTER = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 4.0, -1.0],
    [0.0, -1.0, 0.0],
])
MY_FILTER_NAME = "내 필터"
# ---------------------------------------------------------------------------

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

# 손계산을 따라갈 3x3 창의 왼쪽 위 좌표 (행, 열)
TRACE_ROW = 2
TRACE_COL = 2

# 기본으로 만들어 쓰는 필터 세 개
VERTICAL_FILTER = np.array([
    [-1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
])
HORIZONTAL_FILTER = np.array([
    [-1.0, -1.0, -1.0],
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
])
BLUR_FILTER = np.ones((3, 3)) / 9.0


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """합성곱을 직접 계산한다.

    커널을 이미지 위에서 한 칸씩 옮기며, 겹친 숫자끼리 곱해서 전부 더한다.
    이미지 밖으로 나가는 자리는 계산하지 않는다(valid 방식).
    그래서 8x8 이미지에 3x3 필터를 쓰면 결과는 6x6이 된다.

    참고: 딥러닝 라이브러리가 '합성곱'이라고 부르는 연산도 이 계산과 같다.
    수학 교과서의 합성곱은 커널을 뒤집지만, 필터를 학습으로 찾는 상황에서는
    뒤집으나 안 뒤집으나 결과가 같아서 라이브러리도 뒤집지 않는다.
    """
    kh, kw = kernel.shape
    ih, iw = image.shape
    out_h = ih - kh + 1
    out_w = iw - kw + 1
    out = np.zeros((out_h, out_w))

    for r in range(out_h):
        for c in range(out_w):
            window = image[r:r + kh, c:c + kw]
            out[r, c] = float(np.sum(window * kernel))

    return out


def print_pixel_table(image: np.ndarray) -> None:
    """8x8 픽셀값을 표로 찍는다."""
    header = "     " + "".join(f"{c:>5}" for c in range(image.shape[1]))
    print(header)
    print("     " + "-" * (5 * image.shape[1]))
    for r in range(image.shape[0]):
        row = "".join(f"{image[r, c]:>5.0f}" for c in range(image.shape[1]))
        print(f"{r:>3} |{row}")


def print_convolution_trace(image: np.ndarray, kernel: np.ndarray,
                            row: int, col: int) -> float:
    """3x3 창 하나에 대한 합성곱 계산을 손으로 따라갈 수 있게 출력한다."""
    window = image[row:row + 3, col:col + 3]
    products = window * kernel

    print(f"창 위치: 이미지의 {row}행 {col}열에서 시작하는 3x3 칸")
    print()
    print("  이미지 창          필터            곱한 값")
    for r in range(3):
        left = " ".join(f"{window[r, c]:>4.0f}" for c in range(3))
        mid = " ".join(f"{kernel[r, c]:>5.1f}" for c in range(3))
        right = " ".join(f"{products[r, c]:>6.1f}" for c in range(3))
        print(f"  {left}   x   {mid}   =   {right}")
    print()

    terms = " + ".join(f"({window[r, c]:.0f}x{kernel[r, c]:.0f})"
                       for r in range(3) for c in range(3))
    total = float(products.sum())
    print(f"  더하기: {terms}")
    print(f"        = {total:.1f}")
    print()
    print(f"  이 값 하나가 결과 이미지의 {row}행 {col}열에 들어간다.")
    return total


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


def figure_filter_gallery(image: np.ndarray, results: list, ko: bool,
                          out_path: Path) -> None:
    """원본과 필터 결과를 한 줄로 나란히 그린다."""
    L = _krfont.label
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 4.2))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=16)
    axes[0].set_title(L("원본 8x8", "Original 8x8", ko), fontsize=12, pad=8)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for i, (name_ko, name_en, _kernel, out) in enumerate(results, start=1):
        ax = axes[i]
        # 필터 결과는 음수도 나온다. 0을 회색 가운데에 두고 그린다.
        limit = max(abs(out.min()), abs(out.max()), 1e-9)
        ax.imshow(out, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(L(name_ko, name_en, ko), fontsize=12, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        L("같은 그림에 필터를 바꿔 가며 합성곱을 적용한 결과 (모두 6x6)",
          "Same image, different filters (all outputs are 6x6)", ko),
        fontsize=14,
    )
    fig.text(
        0.5, 0.045,
        L("빨간 곳은 값이 크고 파란 곳은 값이 작다. 흰 곳은 0에 가깝다.",
          "Red is a large value, blue is a small one, white is near zero.", ko),
        ha="center", fontsize=10.5, color="#333333",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    digits = load_digits()
    images = digits.images          # (1797, 8, 8)
    labels = digits.target

    print("=== 데이터 ===")
    print("scikit-learn load_digits() - 실제 손글씨를 스캔해 만든 공개 데이터")
    print(f"이미지 장수: {len(images)}장")
    print(f"이미지 한 장의 크기: {images.shape[1]} x {images.shape[2]} 픽셀")
    print(f"채널 수: 1 (흑백이므로 숫자 판이 한 장이다)")
    print(f"픽셀값 범위: {images.min():.0f} ~ {images.max():.0f}")
    print(f"이미지 한 장이 숫자 {images.shape[1] * images.shape[2]}개다")
    print()

    index = SAMPLE_INDEX % len(images)
    image = images[index]
    digit_label = int(labels[index])

    print("=== 고른 이미지 ===")
    print(f"SAMPLE_INDEX = {SAMPLE_INDEX} (전체 {len(images)}장 중 {index}번)")
    print(f"이 이미지의 정답: {digit_label}")
    print()

    print("=== 8x8 픽셀값 ===")
    print_pixel_table(image)
    print()
    print("0이 검정, 16이 흰색이다. 위 숫자가 이미지의 전부다.")
    print()

    print("=== 합성곱 한 번 손으로 따라가기 ===")
    print("세로선 검출 필터:")
    for r in range(3):
        print("   " + " ".join(f"{VERTICAL_FILTER[r, c]:>5.1f}" for c in range(3)))
    print()
    print_convolution_trace(image, VERTICAL_FILTER, TRACE_ROW, TRACE_COL)
    print()

    filters = [
        ("세로선 검출", "vertical edges", VERTICAL_FILTER),
        ("가로선 검출", "horizontal edges", HORIZONTAL_FILTER),
        ("흐리게", "blur", BLUR_FILTER),
        (MY_FILTER_NAME, "my filter", MY_FILTER),
    ]

    results = []
    for name_ko, name_en, kernel in filters:
        out = convolve2d(image, kernel)
        results.append((name_ko, name_en, kernel, out))

    print("=== 필터 네 개를 적용한 결과 ===")
    print(f"{'필터':<14}{'출력 크기':>10}{'최솟값':>10}{'최댓값':>10}")
    for name_ko, _name_en, _kernel, out in results:
        size = f"{out.shape[0]}x{out.shape[1]}"
        print(f"{name_ko:<14}{size:>10}{out.min():>10.1f}{out.max():>10.1f}")
    print()
    print("8x8 이미지에 3x3 필터를 씌우면 6x6이 된다. 창이 이미지 밖으로 못 나가서다.")
    print()

    print("=== 세로선 검출 필터의 결과 6x6 ===")
    vertical_out = results[0][3]
    print_pixel_table(vertical_out)
    print()
    print("값이 큰 자리에서 왼쪽은 어둡고 오른쪽은 밝다. 세로 방향 경계가 있는 자리다.")
    print()

    print("=== 내 필터 ===")
    for r in range(3):
        print("   " + " ".join(f"{MY_FILTER[r, c]:>5.1f}" for c in range(3)))
    my_out = results[3][3]
    print(f"결과 최솟값 {my_out.min():.1f}, 최댓값 {my_out.max():.1f}")
    print()

    pixels_path = RESULTS_DIR / "digit_pixels.png"
    gallery_path = RESULTS_DIR / "filter_gallery.png"
    figure_pixels(image, digit_label, ko, pixels_path)
    figure_filter_gallery(image, results, ko, gallery_path)

    print("=== 저장된 결과 ===")
    print(f"픽셀값 그림: results/{pixels_path.name}")
    print(f"필터 갤러리: results/{gallery_path.name}")
    print()
    print("SAMPLE_INDEX를 바꾸면 다른 숫자가 나오고, MY_FILTER를 바꾸면 결과가 달라진다.")


if __name__ == "__main__":
    main()
