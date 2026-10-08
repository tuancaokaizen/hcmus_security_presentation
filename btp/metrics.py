import numpy as np

# DeepFace default for Facenet is cosine distance 0.40, i.e. similarity >= 0.60
# (deepface/config/threshold.py) / Mặc định của DeepFace cho Facenet: khoảng cách cosine 0.40, tức độ tương đồng >= 0.60
DEFAULT_THRESHOLD = 0.60


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1].
    Độ tương đồng cosine trong khoảng [-1, 1].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    # Zero vector has no direction / Vector 0 không có hướng nên coi như không giống
    if denom == 0:
        return 0.0
    return float(a @ b / denom)


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of equal bits in [0, 1]; 0.5 means unrelated.
    Tỉ lệ bit trùng trong khoảng [0, 1]; 0.5 nghĩa là không liên quan.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Bit strings differ in length: {a.shape} vs {b.shape}")
    return float(np.mean(a == b))
