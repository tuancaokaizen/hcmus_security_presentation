import numpy as np
import pytest

from btp.attack import hill_climb
from btp.biohash import biohash
from btp.linkage import N_KEY_PAIRS, cross_key_scores
from btp.metrics import DEFAULT_THRESHOLD, cosine_similarity, hamming_similarity
from btp.polyprotect import generate_key, polyprotect, template_length


def _unit_vectors(n: int, dim: int = 128, seed: int = 0) -> np.ndarray:
    """Random unit vectors standing in for L2-normalized embeddings.
    Vector đơn vị ngẫu nhiên thay cho embedding đã chuẩn hóa L2.
    """
    x = np.random.default_rng(seed).standard_normal((n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --- PolyProtect ---


def test_polyprotect_matches_worked_example_in_doc():
    # Table 17 of the team doc / Bảng 17 trong doc của nhóm
    v = np.array([0.3, -0.5, 0.8, 0.1, -0.2])
    coeffs = np.array([12, -7, 30, -45, 5])
    exps = np.array([3, 1, 5, 2, 4])
    p = polyprotect(v, coeffs, exps, overlap=0)
    assert p.shape == (1,)
    assert p[0] == pytest.approx(0.324 + 3.5 + 9.8304 - 0.45 + 0.008)


@pytest.mark.parametrize("overlap, expected", [(0, 26), (1, 32), (2, 42), (3, 63), (4, 124)])
def test_template_length_matches_paper_table(overlap, expected):
    assert template_length(128, overlap) == expected
    coeffs, exps = generate_key(1)
    assert polyprotect(_unit_vectors(1)[0], coeffs, exps, overlap).size == expected


@pytest.mark.parametrize("seed", range(50))
def test_generated_key_is_valid(seed):
    coeffs, exps = generate_key(seed)
    assert len(set(coeffs.tolist())) == 5
    assert all(c != 0 and -50 <= c <= 50 for c in coeffs)
    assert sorted(exps.tolist()) == [1, 2, 3, 4, 5]


def test_same_seed_gives_same_key():
    c1, e1 = generate_key(42)
    c2, e2 = generate_key(42)
    assert np.array_equal(c1, c2) and np.array_equal(e1, e2)


@pytest.mark.parametrize("overlap", [-1, 5])
def test_invalid_overlap_rejected(overlap):
    with pytest.raises(ValueError):
        template_length(128, overlap)


def test_polyprotect_cross_key_is_unlinkable():
    vs = _unit_vectors(300)
    scores = []
    for i, v in enumerate(vs):
        ca, ea = generate_key(2 * i)
        cb, eb = generate_key(2 * i + 1)
        scores.append(cosine_similarity(polyprotect(v, ca, ea, 2), polyprotect(v, cb, eb, 2)))
    assert abs(np.mean(scores)) < 0.1


def test_polyprotect_same_key_keeps_similar_inputs_similar():
    vs = _unit_vectors(100)
    noise = 0.02 * np.random.default_rng(1).standard_normal(vs.shape)
    coeffs, exps = generate_key(7)
    scores = [
        cosine_similarity(polyprotect(v, coeffs, exps, 2), polyprotect(v + n, coeffs, exps, 2))
        for v, n in zip(vs, noise)
    ]
    assert np.mean(scores) > 0.8


# --- BioHashing ---


def test_biohash_is_deterministic_per_key():
    v = _unit_vectors(1)[0]
    assert hamming_similarity(biohash(v, 42), biohash(v, 42)) == 1.0


def test_biohash_cross_key_is_near_random():
    scores = [hamming_similarity(biohash(v, 42), biohash(v, 99)) for v in _unit_vectors(200)]
    assert abs(np.mean(scores) - 0.5) < 0.05


def test_biohash_output_is_binary():
    bits = biohash(_unit_vectors(1)[0], 3)
    assert bits.shape == (128,)
    assert set(np.unique(bits).tolist()) <= {0, 1}


# --- Hill-climbing ---


def test_hill_climb_never_decreases_and_passes_threshold_on_average():
    finals = []
    for seed, target in enumerate(_unit_vectors(20, seed=5)):
        history = [s for _, s, _ in hill_climb(lambda x: cosine_similarity(x, target), seed=seed)]
        assert all(b >= a for a, b in zip(history, history[1:]))
        finals.append(history[-1])
    assert np.mean(finals) > DEFAULT_THRESHOLD


# --- Linkage over many key pairs ---


def test_cross_key_scores_average_to_random_level_and_are_reproducible():
    ea, eb = _unit_vectors(2, seed=9)
    bio, poly = cross_key_scores(ea, ea, 42, 99, overlap=2)
    assert bio.shape == poly.shape == (N_KEY_PAIRS,)
    assert abs(bio.mean() - 0.5) < 0.05
    assert abs(poly.mean()) < 0.1
    bio2, poly2 = cross_key_scores(ea, ea, 42, 99, overlap=2)
    assert np.array_equal(bio, bio2) and np.array_equal(poly, poly2)


# --- Metrics ---


def test_hamming_rejects_length_mismatch():
    with pytest.raises(ValueError):
        hamming_similarity(np.zeros(3), np.zeros(4))
