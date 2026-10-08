"""Download Facenet weights, cache embeddings of images/, and print the numbers for the slides.
Tải weights Facenet, cache embedding của ảnh trong images/, và in số liệu cho slide.

Run once at home with internet: python scripts/precompute.py
Chạy một lần ở nhà khi có mạng: python scripts/precompute.py
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from btp.biohash import biohash
from btp.embedding import IMAGES_DIR, get_embedding, list_images
from btp.linkage import LINKABLE_LEVEL, N_KEY_PAIRS, cross_key_scores
from btp.metrics import DEFAULT_THRESHOLD, cosine_similarity, hamming_similarity
from btp.polyprotect import generate_key, polyprotect

# Same defaults as Tab 4 / Cùng giá trị mặc định với Tab 4
KEY_A, KEY_B, OVERLAP = 42, 7, 2


def main() -> int:
    images = list_images()
    if not images:
        print(f"Chưa có ảnh trong {IMAGES_DIR}. Đặt ảnh theo dạng nguoiA_1.jpg, nguoiA_2.jpg, ...")
        return 1

    # Compute (and cache) every embedding; report faces that cannot be detected
    # / Tính (và cache) mọi embedding; báo ảnh không tìm thấy mặt
    embeddings = {}
    for path in images:
        try:
            embeddings[path.name] = get_embedding(path)
            print(f"OK   {path.name}")
        except ValueError as exc:
            print(f"LỖI  {exc}")
    if len(embeddings) < 2:
        return 1

    ca, ea = generate_key(KEY_A)
    cb, eb = generate_key(KEY_B)
    print(f"\nNgưỡng = {DEFAULT_THRESHOLD:.2f} | BioHash/PolyProtect key {KEY_A} vs {KEY_B} | overlap {OVERLAP}")
    header = (f"{'Ảnh 1':<16}{'Ảnh 2':<16}{'Cùng người':<12}{'Cosine':>8}{'BioH khác key':>15}"
              f"{'Poly cùng key':>15}{'Poly khác key':>15}{f'Poly TB {N_KEY_PAIRS}':>14}{'Poly |s|>0.5':>14}")
    print(header)
    print("-" * len(header))
    for (n1, v1), (n2, v2) in combinations(embeddings.items(), 2):
        same = n1.rsplit("_", 1)[0] == n2.rsplit("_", 1)[0]
        poly_a = polyprotect(v1, ca, ea, OVERLAP)
        _, poly_many = cross_key_scores(v1, v2, KEY_A, KEY_B, OVERLAP)
        print(
            f"{n1:<16}{n2:<16}{'có' if same else 'không':<12}"
            f"{cosine_similarity(v1, v2):>8.3f}"
            f"{hamming_similarity(biohash(v1, KEY_A), biohash(v2, KEY_B)):>15.3f}"
            f"{cosine_similarity(poly_a, polyprotect(v2, ca, ea, OVERLAP)):>15.3f}"
            f"{cosine_similarity(poly_a, polyprotect(v2, cb, eb, OVERLAP)):>15.3f}"
            f"{poly_many.mean():>14.3f}"
            f"{np.mean(np.abs(poly_many) > LINKABLE_LEVEL):>14.0%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
