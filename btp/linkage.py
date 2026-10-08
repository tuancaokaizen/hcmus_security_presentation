import numpy as np

from btp.biohash import biohash
from btp.metrics import cosine_similarity, hamming_similarity
from btp.polyprotect import generate_key, polyprotect

N_KEY_PAIRS = 200
LINKABLE_LEVEL = 0.5


def cross_key_scores(
    ea: np.ndarray, eb: np.ndarray, key_a: int, key_b: int, overlap: int, n_pairs: int = N_KEY_PAIRS
) -> tuple[np.ndarray, np.ndarray]:
    """BioHash (Hamming) and PolyProtect (cosine) cross-key scores over many random key pairs.
    Điểm khác key của BioHash (Hamming) và PolyProtect (cosine) trên nhiều cặp key ngẫu nhiên.

    The pairs are seeded by (key_a, key_b) so results are reproducible.
    Các cặp key được seed từ (key_a, key_b) nên kết quả lặp lại được.
    """
    seeds = np.random.default_rng([key_a, key_b]).integers(0, 2**31, size=(n_pairs, 2))
    bio, poly = [], []
    for sa, sb in seeds:
        bio.append(hamming_similarity(biohash(ea, sa), biohash(eb, sb)))
        poly.append(cosine_similarity(polyprotect(ea, *generate_key(sa), overlap),
                                      polyprotect(eb, *generate_key(sb), overlap)))
    return np.array(bio), np.array(poly)
