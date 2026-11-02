"""실습 9.2 - 필터와 pooling을 직접 계산한다.

하는 일은 다섯이다.
  1. 3x3 창 하나의 합성곱을 곱셈 아홉 개까지 풀어서 찍는다.
  2. 세로선 필터에 가장 크게, 가장 작게, 0에 가깝게 반응한 창을 찾아 나란히 본다.
     필터와 생김새가 닮은 창일수록 곱의 합이 크다는 것을 확인한다.
  3. 창을 한 칸씩 옮기며 6x6 결과가 채워지는 순서를 그린다.
  4. 필터 네 개(세로선·가로선·흐리게·내 필터)를 적용해 결과를 비교한다.
  5. 이미지를 한 칸 민 뒤 합성곱과 2x2 최대 pooling을 다시 계산해, 원본과 무엇이 같고 무엇이 다른지 본다.

데이터는 scikit-learn에 들어 있는 load_digits()다. 내려받지 않고 바로 쓴다.
1797장의 8x8 손글씨 숫자이고, 픽셀값은 0(검정)부터 16(흰색)까지다.
합성 데이터가 아니라 실제로 사람이 쓴 숫자를 스캔해 만든 공개 데이터다.

torch나 tensorflow를 쓰지 않는다. 합성곱은 convolve2d(), pooling은 max_pool2d()에서 직접 계산한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
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

# pooling 비교에서 이미지를 오른쪽으로 몇 칸 밀 것인가
SHIFT_PIXELS = 1
# ---------------------------------------------------------------------------

CHAPTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = CHAPTER_DIR / "results"

# 손계산을 따라갈 3x3 창의 왼쪽 위 좌표 (행, 열)
TRACE_ROW = 2
TRACE_COL = 2

# 창이 옮겨 가는 과정을 그릴 때 보여 줄 창 위치 (행, 열)
SLIDE_STEPS = [(0, 0), (0, 1), (0, 5), (5, 5)]

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

ORANGE = "#F58518"
GREEN = "#54A24B"
GRAY = "#79706E"


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


def max_pool2d(feature: np.ndarray, size: int = 2) -> np.ndarray:
    """size x size 덩어리마다 가장 큰 값 하나만 남긴다. 덩어리는 겹치지 않는다."""
    out_h = feature.shape[0] // size
    out_w = feature.shape[1] // size
    out = np.zeros((out_h, out_w))
    for r in range(out_h):
        for c in range(out_w):
            block = feature[r * size:(r + 1) * size, c * size:(c + 1) * size]
            out[r, c] = float(block.max())
    return out


def shift_right(image: np.ndarray, k: int) -> np.ndarray:
    """이미지를 오른쪽으로 k칸 민다. 왼쪽에 비는 칸은 0(검정)으로 채운다."""
    out = np.zeros_like(image)
    out[:, k:] = image[:, :image.shape[1] - k]
    return out


def print_table(values: np.ndarray) -> None:
    """숫자 판을 표로 찍는다."""
    print("     " + "".join(f"{c:>5}" for c in range(values.shape[1])))
    print("     " + "-" * (5 * values.shape[1]))
    for r in range(values.shape[0]):
        row = "".join(f"{values[r, c]:>5.0f}" for c in range(values.shape[1]))
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


def draw_numbers(ax, values, vmin, vmax, cmap, fontsize=9):
    """숫자 판을 색과 숫자로 함께 그린다."""
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    mid = (vmin + vmax) / 2
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            v = values[r, c]
            if np.isnan(v):
                continue
            if cmap == "gray":
                color = "#111111" if v > mid else "#FFFFFF"
            else:
                color = "#FFFFFF" if abs(v) > 0.6 * max(abs(vmin), abs(vmax)) else "#111111"
            ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                    fontsize=fontsize, color=color)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_similarity(kernel: np.ndarray, picks: list, ko: bool,
                      out_path: Path) -> None:
    """필터 모양과, 그 필터에 크게·작게·0 근처로 반응한 창 세 개를 나란히 그린다."""
    L = _krfont.label
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.0))

    ax = axes[0]
    ax.imshow(kernel, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{kernel[r, c]:.0f}", ha="center", va="center", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(L("세로선 필터\n(왼쪽 어둡게, 오른쪽 밝게)",
                   "Vertical-edge filter\n(dark left, bright right)", ko), fontsize=11.5)

    for ax, (title_ko, title_en, (r, c), window, value) in zip(axes[1:], picks):
        draw_numbers(ax, window, 0, 16, "gray", fontsize=12)
        ax.set_title(
            L(f"{title_ko}\n{r}행 {c}열 창, 결과 {value:.0f}",
              f"{title_en}\nwindow at ({r},{c}), result {value:.0f}", ko),
            fontsize=11.5,
        )
    fig.suptitle(
        L("필터와 생김새가 닮은 창일수록 결과가 크고, 반대로 생긴 창은 음수가 된다",
          "The more a window looks like the filter, the larger the result", ko),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_sliding(image: np.ndarray, out: np.ndarray, ko: bool,
                   out_path: Path) -> None:
    """창이 한 칸씩 옮겨 가며 6x6 결과를 채우는 과정을 네 장면으로 그린다."""
    L = _krfont.label
    n = len(SLIDE_STEPS)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7.6))
    limit = float(np.abs(out).max())
    order = [(r, c) for r in range(out.shape[0]) for c in range(out.shape[1])]

    for k, (r, c) in enumerate(SLIDE_STEPS):
        ax = axes[0, k]
        draw_numbers(ax, image, 0, 16, "gray", fontsize=7.5)
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 3, 3, fill=False,
                               edgecolor=ORANGE, linewidth=3))
        step_no = order.index((r, c)) + 1
        ax.set_title(L(f"{step_no}번째 창: {r}행 {c}열",
                       f"window {step_no}: row {r}, col {c}", ko), fontsize=11)

        filled = np.full(out.shape, np.nan)
        for rr, cc in order[:step_no]:
            filled[rr, cc] = out[rr, cc]
        ax = axes[1, k]
        ax.imshow(np.zeros(out.shape), cmap="Greys", vmin=0, vmax=1)
        shown = np.ma.masked_invalid(filled)
        ax.imshow(shown, cmap="RdBu_r", vmin=-limit, vmax=limit)
        for rr in range(out.shape[0]):
            for cc in range(out.shape[1]):
                v = filled[rr, cc]
                ax.text(cc, rr, "?" if np.isnan(v) else f"{v:.0f}",
                        ha="center", va="center", fontsize=8.5,
                        color="#AAAAAA" if np.isnan(v) else "#111111")
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                               edgecolor=GREEN, linewidth=3))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(L(f"결과 6x6: {step_no}/36칸 채움",
                       f"output 6x6: {step_no}/36 filled", ko), fontsize=11)

    fig.suptitle(
        L("주황 창 하나가 초록 칸 하나가 된다. 창의 왼쪽 위 모서리가 놓일 수 있는 자리가 6 x 6이다",
          "One orange window becomes one green cell; the window corner has 6 x 6 places to go", ko),
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0)
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
        # 필터 결과는 음수도 나온다. 0을 흰색 가운데에 두고 그린다.
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


def figure_pool_shift(rows: list, ko: bool, out_path: Path) -> None:
    """원본과 민 이미지 각각에 대해 이미지 -> 합성곱 6x6 -> pooling 3x3을 그린다."""
    L = _krfont.label
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0),
                             gridspec_kw={"width_ratios": [8, 6, 3]})
    limit = max(float(np.abs(conv).max()) for _n, _e, _img, conv, _pool in rows)
    for i, (name_ko, name_en, img, conv, pooled) in enumerate(rows):
        draw_numbers(axes[i, 0], img, 0, 16, "gray", fontsize=8)
        axes[i, 0].set_title(L(f"{name_ko} 8x8", f"{name_en} 8x8", ko), fontsize=11.5)
        draw_numbers(axes[i, 1], conv, -limit, limit, "RdBu_r", fontsize=9)
        axes[i, 1].set_title(L("세로선 필터 결과 6x6", "vertical filter 6x6", ko), fontsize=11.5)
        for k in (1.5, 3.5):
            axes[i, 1].axhline(k, color="#333333", linewidth=1.4)
            axes[i, 1].axvline(k, color="#333333", linewidth=1.4)
        draw_numbers(axes[i, 2], pooled, -limit, limit, "RdBu_r", fontsize=11)
        axes[i, 2].set_title(L("2x2 최대 pooling 3x3", "2x2 max pool 3x3", ko), fontsize=11.5)
    fig.suptitle(
        L("그림을 한 칸 밀면 합성곱 결과도 같은 숫자 그대로 한 칸 밀린다 (검은 선은 pooling 덩어리)",
          "Shift the image by one and the conv output shifts by one with the same numbers", ko),
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ko = _krfont.setup()
    _krfont.report(ko)

    digits = load_digits()
    images = digits.images          # (1797, 8, 8)
    labels = digits.target

    index = SAMPLE_INDEX % len(images)
    image = images[index]
    digit_label = int(labels[index])

    print("=== 고른 이미지 ===")
    print("scikit-learn load_digits() - 실제 손글씨를 스캔해 만든 공개 데이터")
    print(f"SAMPLE_INDEX = {SAMPLE_INDEX} (전체 {len(images)}장 중 {index}번)")
    print(f"이 이미지의 정답: {digit_label}")
    print()
    print_table(image)
    print()

    # ------------------------------------------------------------------
    # 1. 창 하나를 손으로 따라간다
    print("=== 합성곱 한 번 손으로 따라가기 ===")
    print("세로선 검출 필터:")
    for r in range(3):
        print("   " + " ".join(f"{VERTICAL_FILTER[r, c]:>5.1f}" for c in range(3)))
    print()
    print_convolution_trace(image, VERTICAL_FILTER, TRACE_ROW, TRACE_COL)
    print()

    vertical_out = convolve2d(image, VERTICAL_FILTER)

    # ------------------------------------------------------------------
    # 2. 가장 크게·가장 작게·0 근처로 반응한 창
    flat_order = np.argsort(vertical_out, axis=None)
    pos_max = tuple(int(v) for v in np.unravel_index(flat_order[-1], vertical_out.shape))
    pos_min = tuple(int(v) for v in np.unravel_index(flat_order[0], vertical_out.shape))
    pos_zero = tuple(int(v) for v in np.unravel_index(
        np.argmin(np.abs(vertical_out)), vertical_out.shape))

    def window_at(pos):
        r, c = pos
        return image[r:r + 3, c:c + 3]

    print("=== 세로선 필터는 무엇과 닮은 창에 크게 반응하는가 ===")
    picks = []
    for name_ko, name_en, pos in [
        ("가장 큰 결과", "largest", pos_max),
        ("가장 작은 결과", "smallest", pos_min),
        ("0에 가장 가까운 결과", "closest to 0", pos_zero),
    ]:
        w = window_at(pos)
        value = float(vertical_out[pos])
        left, right = w[:, 0].sum(), w[:, 2].sum()
        print(f"{name_ko}: {pos[0]}행 {pos[1]}열 창, 결과 {value:.0f}")
        print(f"  왼쪽 열 합 {left:.0f}, 오른쪽 열 합 {right:.0f}, "
              f"오른쪽 - 왼쪽 = {right - left:.0f}")
        for r in range(3):
            print("    " + " ".join(f"{w[r, c]:>4.0f}" for c in range(3)))
        picks.append((name_ko, name_en, pos, w, value))
    print()
    print("이 필터의 결과는 (오른쪽 열 합) - (왼쪽 열 합)과 같다. 가운데 열에는 0을 곱하기 때문이다.")
    print()

    # ------------------------------------------------------------------
    # 3. 필터 네 개
    filters = [
        ("세로선 검출", "vertical edges", VERTICAL_FILTER),
        ("가로선 검출", "horizontal edges", HORIZONTAL_FILTER),
        ("흐리게", "blur", BLUR_FILTER),
        (MY_FILTER_NAME, "my filter", MY_FILTER),
    ]
    results = []
    for name_ko, name_en, kernel in filters:
        results.append((name_ko, name_en, kernel, convolve2d(image, kernel)))

    print("=== 필터 네 개를 적용한 결과 ===")
    print(f"{'필터':<14}{'출력 크기':>10}{'최솟값':>10}{'최댓값':>10}")
    for name_ko, _name_en, _kernel, out in results:
        size = f"{out.shape[0]}x{out.shape[1]}"
        print(f"{name_ko:<14}{size:>10}{out.min():>10.1f}{out.max():>10.1f}")
    print()
    print("8x8 이미지에 3x3 필터를 씌우면 6x6이 된다. 창이 이미지 밖으로 못 나가서다.")
    print()

    print("=== 세로선 검출 필터의 결과 6x6 ===")
    print_table(vertical_out)
    print()

    print("=== 내 필터 ===")
    for r in range(3):
        print("   " + " ".join(f"{MY_FILTER[r, c]:>5.1f}" for c in range(3)))
    my_out = results[3][3]
    print(f"결과 최솟값 {my_out.min():.1f}, 최댓값 {my_out.max():.1f}")
    print()

    # ------------------------------------------------------------------
    # 4. 한 칸 밀고 합성곱과 pooling을 다시 계산한다
    k = SHIFT_PIXELS
    shifted = shift_right(image, k)
    shifted_out = convolve2d(shifted, VERTICAL_FILTER)
    pooled = max_pool2d(vertical_out)
    shifted_pooled = max_pool2d(shifted_out)

    print(f"=== 이미지를 오른쪽으로 {k}칸 밀었을 때 ===")
    print("밀린 이미지의 세로선 필터 결과 6x6:")
    print_table(shifted_out)
    print()
    same = np.array_equal(shifted_out[:, k:], vertical_out[:, :vertical_out.shape[1] - k])
    print(f"밀린 결과의 {k}열부터 끝까지 == 원본 결과의 0열부터 {5 - k}열까지: "
          f"{'같다' if same else '다르다'}")
    print("같은 필터를 모든 자리에 똑같이 쓰므로, 무늬가 옮겨 가면 반응도 같은 값 그대로 옮겨 간다.")
    print()
    print("원본 결과의 2x2 최대 pooling 3x3:")
    print_table(pooled)
    print("밀린 결과의 2x2 최대 pooling 3x3:")
    print_table(shifted_pooled)
    n_same = int(np.sum(pooled == shifted_pooled))
    print(f"3x3 아홉 칸 중 값이 같은 칸: {n_same}칸")
    print("2x2 덩어리 하나로는 한 칸 이동을 다 흡수하지 못한다. 값이 덩어리 경계를 넘으면 옆 덩어리로 옮겨 간다.")
    print(f"가장 큰 반응: 원본 {vertical_out.max():.0f}, 밀린 이미지 {shifted_out.max():.0f}")
    print()

    # ------------------------------------------------------------------
    similarity_path = RESULTS_DIR / "filter_similarity.png"
    sliding_path = RESULTS_DIR / "sliding_window.png"
    gallery_path = RESULTS_DIR / "filter_gallery.png"
    pool_path = RESULTS_DIR / "pooling_shift.png"
    figure_similarity(VERTICAL_FILTER, picks, ko, similarity_path)
    figure_sliding(image, vertical_out, ko, sliding_path)
    figure_filter_gallery(image, results, ko, gallery_path)
    figure_pool_shift(
        [("원본", "original", image, vertical_out, pooled),
         (f"오른쪽 {k}칸 밀기", f"shifted right {k}", shifted, shifted_out, shifted_pooled)],
        ko, pool_path,
    )

    print("=== 저장된 결과 ===")
    print(f"닮은 정도 그림: results/{similarity_path.name}")
    print(f"창 옮기기 그림: results/{sliding_path.name}")
    print(f"필터 갤러리: results/{gallery_path.name}")
    print(f"pooling 비교 그림: results/{pool_path.name}")
    print()
    print("SAMPLE_INDEX를 바꾸면 다른 숫자가 나오고, MY_FILTER를 바꾸면 결과가 달라진다.")


if __name__ == "__main__":
    main()
