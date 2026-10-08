"""Export the demo charts and numbers used by slides/build_slides.py.
Xuất biểu đồ và số liệu của demo cho slides/build_slides.py.

Run after putting the team photos in images/: python scripts/export_figures.py
Chạy sau khi bỏ ảnh của nhóm vào images/: python scripts/export_figures.py
"""

import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from matplotlib.figure import Figure

import demo_app as app
from btp.attack import hill_climb
from btp.biohash import biohash
from btp.embedding import ROOT, get_embedding, list_images
from btp.linkage import LINKABLE_LEVEL, N_KEY_PAIRS, cross_key_scores
from btp.metrics import DEFAULT_THRESHOLD, cosine_similarity, hamming_similarity
from btp.polyprotect import generate_key, polyprotect, template_length

OUT_DIR = ROOT / "slides" / "figures"
BIO_KEYS = (42, 99)
POLY_KEYS = (42, 7)
OVERLAP = 2
ATTACK_SEEDS = range(5)
DPI = 200


def _person(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def _similarity_matrix_figure(names: list[str], matrix: np.ndarray) -> Figure:
    """Cosine matrix of all photos: same-person blocks should be bright.
    Ma trận cosine giữa mọi ảnh: các ô cùng người phải sáng.
    """
    fig = Figure(figsize=(5.2, 4.4))
    ax = fig.subplots()
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
    fig.tight_layout()
    return fig


def _powers_figure(typical_abs: float) -> Figure:
    """Size of |v|^e for e = 1..5 at a typical embedding value (log scale).
    Độ lớn |v|^e với e = 1..5 tại giá trị embedding điển hình (thang log).
    """
    exps = np.arange(1, 6)
    fig = Figure(figsize=(5.2, 3.2))
    ax = fig.subplots()
    ax.bar([f"$v^{e}$" for e in exps], typical_abs**exps, color=["#27ae60"] + ["#95a5a6"] * 4)
    ax.set_yscale("log")
    ax.set_ylabel("Độ lớn (thang log)")
    ax.set_title(f"|v| điển hình = {typical_abs:.2f}".replace(".", ","))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> int:
    images = list_images()
    if len(images) < 4:
        print("Cần ít nhất 4 ảnh (2 người × 2 ảnh) trong images/.")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = {p: get_embedding(p) for p in images}
    names = [p.stem for p in images]
    vectors = list(embeddings.values())
    matrix = np.array([[cosine_similarity(a, b) for b in vectors] for a in vectors])

    # Pick the first same-person pair as the running example / Lấy cặp cùng người đầu tiên làm ví dụ xuyên suốt
    pairs = list(combinations(images, 2))
    same_pairs = [(a, b) for a, b in pairs if _person(a) == _person(b)]
    diff_pairs = [(a, b) for a, b in pairs if _person(a) != _person(b)]
    if not same_pairs or not diff_pairs:
        print("Cần ít nhất 2 người, mỗi người 2 ảnh.")
        return 1
    img_a, img_b = same_pairs[0]
    same_scores = [cosine_similarity(embeddings[a], embeddings[b]) for a, b in same_pairs]
    diff_scores = [cosine_similarity(embeddings[a], embeddings[b]) for a, b in diff_pairs]

    # Tab 1 / Tab 1
    _similarity_matrix_figure(names, matrix).savefig(OUT_DIR / "tab1_matrix.png", dpi=DPI)

    # Tab 2: chart of seed 0 plus statistics over several seeds / Biểu đồ seed 0 và thống kê nhiều seed
    attacks = []
    for seed in ATTACK_SEEDS:
        target = embeddings[img_a]
        history = [s for _, s, _ in hill_climb(lambda x: cosine_similarity(x, target), seed=seed)]
        crossed = next((i for i, s in enumerate(history) if s >= DEFAULT_THRESHOLD), None)
        attacks.append({"seed": seed, "start": history[0], "final": history[-1], "crossed_at": crossed})
        if seed == 0:
            app._climb_figure(history, DEFAULT_THRESHOLD, len(history) - 1).savefig(OUT_DIR / "tab2_attack.png", dpi=DPI)

    # Longer run answers "what about 1000 iterations?" / Chạy dài hơn để trả lời câu hỏi "1000 vòng thì sao?"
    target = embeddings[img_a]
    long_run = [s for _, s, _ in hill_climb(lambda x: cosine_similarity(x, target), iterations=1000, seed=0)]

    # Tab 3 / Tab 3
    e = embeddings[img_a]
    bits_a, bits_b = biohash(e, BIO_KEYS[0]), biohash(e, BIO_KEYS[1])
    app._bits_figure(bits_a, bits_b, *BIO_KEYS).savefig(OUT_DIR / "tab3_bits.png", dpi=DPI)

    # Tab 4: same computation as the app / Cùng cách tính với app
    ea, eb = embeddings[img_a], embeddings[img_b]
    bio_many, poly_many = cross_key_scores(ea, eb, *POLY_KEYS, OVERLAP)
    unprotected = cosine_similarity(ea, eb)
    app._linkage_figure([unprotected, float(bio_many.mean()), float(poly_many.mean())]).savefig(
        OUT_DIR / "tab4_linkage.png", dpi=DPI)
    ca, exa = generate_key(POLY_KEYS[0])
    cb, exb = generate_key(POLY_KEYS[1])
    c99, ex99 = generate_key(BIO_KEYS[1])
    pa = polyprotect(ea, ca, exa, OVERLAP)

    # Finding slide: why one PolyProtect key pair stays linkable / Slide phát hiện: vì sao một cặp key vẫn liên kết được
    typical_abs = float(np.median(np.abs(ea)))
    _powers_figure(typical_abs).savefig(OUT_DIR / "finding_powers.png", dpi=DPI)

    numbers = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_images": len(images),
        "n_people": len({_person(p) for p in images}),
        "image_names": names,
        "threshold": DEFAULT_THRESHOLD,
        "example_pair": [img_a.stem, img_b.stem],
        "tab1": {
            "same_min": min(same_scores), "same_max": max(same_scores),
            "diff_min": min(diff_scores), "diff_max": max(diff_scores),
        },
        "tab2": {
            "iterations": 300, "sigma": 0.2, "runs": attacks,
            "seed0_at_300": long_run[300], "seed0_at_1000": long_run[1000],
        },
        "tab3": {
            "keys": list(BIO_KEYS),
            "same_key": hamming_similarity(bits_a, bits_a),
            "cross_key": hamming_similarity(bits_a, bits_b),
            "first32_a": "".join(map(str, bits_a[:32])),
            "first32_b": "".join(map(str, bits_b[:32])),
        },
        "tab4": {
            "keys": list(POLY_KEYS), "overlap": OVERLAP, "n_key_pairs": N_KEY_PAIRS,
            "template_length": template_length(ea.size, OVERLAP),
            "unprotected": unprotected,
            "bio_same_key": hamming_similarity(biohash(ea, POLY_KEYS[0]), biohash(eb, POLY_KEYS[0])),
            "bio_cross_mean": float(bio_many.mean()),
            "poly_same_key": cosine_similarity(pa, polyprotect(eb, ca, exa, OVERLAP)),
            "poly_cross": cosine_similarity(pa, polyprotect(eb, cb, exb, OVERLAP)),
            "poly_cross_mean": float(poly_many.mean()),
            "poly_linkable_rate": float(np.mean(np.abs(poly_many) > LINKABLE_LEVEL)),
            "poly_cross_42_99": cosine_similarity(pa, polyprotect(eb, c99, ex99, OVERLAP)),
            "key_42": {"C": ca.tolist(), "E": exa.tolist()},
            "key_99": {"C": c99.tolist(), "E": ex99.tolist()},
            "typical_abs_v": typical_abs,
            "max_abs_v": float(np.max(np.abs(ea))),
        },
    }
    (OUT_DIR / "numbers.json").write_text(json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(numbers, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
