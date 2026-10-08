import math

import numpy as np

# Values used by Hahn & Marcel (IEEE T-BIOM 2022) / Tham số theo Hahn & Marcel (IEEE T-BIOM 2022)
M = 5
COEF_RANGE = 50


def generate_key(seed: int, m: int = M) -> tuple[np.ndarray, np.ndarray]:
    """Generate the user-specific key (C, E).
    Sinh key riêng cho từng người (C, E).

    C: m distinct non-zero integers in [-50, 50] / m số nguyên khác 0, đôi một khác nhau, trong [-50, 50]
    E: a random permutation of 1..m / một hoán vị ngẫu nhiên của 1..m
    """
    rng = np.random.default_rng(seed)
    candidates = np.array([c for c in range(-COEF_RANGE, COEF_RANGE + 1) if c != 0])
    coeffs = rng.choice(candidates, size=m, replace=False)
    exps = rng.permutation(np.arange(1, m + 1))
    return coeffs, exps


def template_length(n: int, overlap: int, m: int = M) -> int:
    """Number of protected elements for an n-dim embedding.
    Độ dài template sau bảo vệ với embedding n chiều.
    """
    _check_overlap(overlap, m)
    step = m - overlap
    return math.ceil((n - m) / step) + 1


def polyprotect(embedding: np.ndarray, coeffs: np.ndarray, exps: np.ndarray, overlap: int) -> np.ndarray:
    """Map each window of m embedding values to one number: p = sum(c_i * v_i ** e_i).
    Ánh xạ mỗi nhóm m phần tử của embedding thành một số: p = tổng(c_i * v_i ** e_i).
    """
    v = np.asarray(embedding, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    exps = np.asarray(exps)
    m = coeffs.size
    if exps.size != m:
        raise ValueError("C and E must have the same length")

    step = m - overlap
    k = template_length(v.size, overlap, m)

    # Zero-pad so the last window is complete / Đệm số 0 để nhóm cuối đủ m phần tử
    padded = np.zeros((k - 1) * step + m)
    padded[: v.size] = v

    # Consecutive windows share `overlap` values / Hai nhóm liền nhau dùng chung `overlap` phần tử
    windows = np.lib.stride_tricks.sliding_window_view(padded, m)[::step]
    return (coeffs * windows**exps).sum(axis=1)


def _check_overlap(overlap: int, m: int) -> None:
    if not 0 <= overlap < m:
        raise ValueError(f"overlap must be in [0, {m - 1}], got {overlap}")
