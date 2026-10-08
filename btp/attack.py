from collections.abc import Callable, Iterator

import numpy as np


def hill_climb(
    score_fn: Callable[[np.ndarray], float],
    dim: int = 128,
    iterations: int = 300,
    sigma: float = 0.2,
    seed: int = 0,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Hill-climbing attack that only sees the matcher's score; yields (iteration, best_score, best_probe).
    Tấn công hill-climbing chỉ nhìn thấy điểm của matcher; trả lần lượt (vòng lặp, điểm tốt nhất, probe tốt nhất).

    One matcher query per iteration / Mỗi vòng lặp gửi 1 truy vấn tới matcher.
    """
    rng = np.random.default_rng(seed)

    # Start from a random probe with no knowledge of the victim / Bắt đầu từ probe ngẫu nhiên, không biết gì về nạn nhân
    probe = rng.standard_normal(dim)
    best = score_fn(probe)
    yield 0, best, probe

    for i in range(1, iterations + 1):
        # Add small noise and keep the candidate only if the score improves / Thêm nhiễu nhỏ, chỉ giữ nếu điểm tăng
        candidate = probe + sigma * rng.standard_normal(dim)
        score = score_fn(candidate)
        if score > best:
            probe, best = candidate, score
        yield i, best, probe
