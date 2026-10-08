import numpy as np


def biohash(embedding: np.ndarray, key: int, n_bits: int = 128) -> np.ndarray:
    """BioHashing (Teoh, Ngo & Goh 2004): project onto key-seeded orthonormal vectors, then take the sign.
    BioHashing (Teoh, Ngo & Goh 2004): chiếu lên các vector trực chuẩn sinh từ key, rồi lấy dấu.

    Returns a uint8 array of n_bits values in {0, 1}.
    Trả về mảng uint8 gồm n_bits giá trị 0/1.
    """
    v = np.asarray(embedding, dtype=float)
    if n_bits > v.size:
        raise ValueError(f"n_bits ({n_bits}) cannot exceed embedding size ({v.size})")

    # The key is the only source of randomness, so the same key always gives the same matrix
    # / Key là nguồn ngẫu nhiên duy nhất nên cùng key luôn ra cùng ma trận
    rng = np.random.default_rng(key)
    random_matrix = rng.standard_normal((v.size, n_bits))

    # QR gives the same orthonormal basis as Gram-Schmidt in the original paper
    # / QR cho cùng cơ sở trực chuẩn như Gram-Schmidt trong bài báo gốc
    basis, _ = np.linalg.qr(random_matrix)

    # Threshold tau = 0 / Ngưỡng tau = 0
    return (v @ basis > 0).astype(np.uint8)
