import csv
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from btp.attack import hill_climb
from btp.biohash import biohash
from btp.embedding import ROOT, get_embedding, list_images
from btp.linkage import LINKABLE_LEVEL, N_KEY_PAIRS, cross_key_scores
from btp.metrics import DEFAULT_THRESHOLD, cosine_similarity, hamming_similarity
from btp.polyprotect import generate_key, polyprotect

RESULTS_FILE = ROOT / "results" / "demo_results.csv"


# ---------- Helpers / Hàm phụ ----------


def _embed(image_path: str | None, label: str) -> np.ndarray:
    """Embedding for a UI image, with user-facing errors.
    Lấy embedding cho ảnh trên giao diện, báo lỗi dễ hiểu.
    """
    if not image_path:
        raise gr.Error(f"Hãy chọn {label}.")
    try:
        return get_embedding(image_path)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def _record(tab: str, **metrics: float) -> None:
    """Append real numbers to results/demo_results.csv for the slides.
    Ghi số thật vào results/demo_results.csv để đưa lên slide.
    """
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    is_new = not RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["time", "tab", "metric", "value"])
        now = datetime.now().isoformat(timespec="seconds")
        for name, value in metrics.items():
            writer.writerow([now, tab, name, f"{value:.4f}" if isinstance(value, float) else value])


def _person(path: Path) -> str:
    """Person id from file name, e.g. nguoiA_1.jpg -> nguoiA.
    Mã người lấy từ tên file, ví dụ nguoiA_1.jpg -> nguoiA.
    """
    return path.stem.rsplit("_", 1)[0]


def _pair_examples(same_person_only: bool = False) -> list[list[str]]:
    """Image pairs from images/: same-person pairs first, then different-person pairs.
    Các cặp ảnh trong images/: cặp cùng người trước, rồi cặp khác người.
    """
    pairs = list(combinations(list_images(), 2))
    same = [[str(a), str(b)] for a, b in pairs if _person(a) == _person(b)]
    if same_person_only:
        return same
    diff = [[str(a), str(b)] for a, b in pairs if _person(a) != _person(b)]
    return same + diff[:3]


def _single_examples() -> list[list[str]]:
    return [[str(p)] for p in list_images()]


def _verdict(score: float, threshold: float) -> str:
    return "✅ **MATCH**" if score >= threshold else "❌ **NO MATCH**"


# ---------- Tab 1: Enroll & Match ----------


def enroll_and_match(img_enroll: str, img_probe: str, threshold: float) -> str:
    """Compare two faces with cosine similarity on Facenet embeddings.
    So khớp hai khuôn mặt bằng cosine trên embedding Facenet.
    """
    e1 = _embed(img_enroll, "ảnh đăng ký")
    e2 = _embed(img_probe, "ảnh xác thực")
    score = cosine_similarity(e1, e2)
    _record("1_match", cosine=score, threshold=threshold)
    return (
        f"### Cosine similarity = `{score:.3f}`\n"
        f"Ngưỡng = `{threshold:.2f}` → {_verdict(score, threshold)}\n\n"
        f"Template lưu trong CSDL là vector 128 số thực (Facenet), **chưa được bảo vệ**."
    )


# ---------- Tab 2: Hill-climbing ----------


def _climb_figure(history: list[float], threshold: float, total: int) -> Figure:
    fig = Figure(figsize=(7, 3.6))
    ax = fig.subplots()
    ax.plot(range(len(history)), history, color="#c0392b", lw=2, label="Điểm tốt nhất của kẻ tấn công")
    ax.axhline(threshold, color="#2c3e50", ls="--", label=f"Ngưỡng matcher = {threshold:.2f}")
    ax.set_xlim(0, total)
    ax.set_ylim(-0.3, 1.0)
    ax.set_xlabel("Số vòng lặp (= số lần hỏi matcher)")
    ax.set_ylabel("Cosine similarity")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def run_attack(img_victim: str, iterations: int, sigma: float, seed: int, threshold: float):
    """Hill-climbing against an unprotected matcher; streams the chart.
    Tấn công hill-climbing vào matcher không bảo vệ; vẽ biểu đồ dần dần.
    """
    target = _embed(img_victim, "ảnh nạn nhân")
    iterations, seed = int(iterations), int(seed)

    # The attacker only receives the matcher's score, never the template itself
    # / Kẻ tấn công chỉ nhận điểm từ matcher, không bao giờ thấy template
    def matcher_score(probe: np.ndarray) -> float:
        return cosine_similarity(probe, target)

    history: list[float] = []
    crossed_at = None
    for i, best, _ in hill_climb(matcher_score, iterations=iterations, sigma=sigma, seed=seed):
        history.append(best)
        if crossed_at is None and best >= threshold:
            crossed_at = i
        # Stream every 10 iterations; the short pause lets the audience see the curve grow
        # / Cập nhật mỗi 10 vòng; nghỉ ngắn để khán giả thấy đường cong tăng dần
        if i % 10 == 0 and i < iterations:
            yield _climb_figure(history, threshold, iterations), f"Đang chạy… vòng {i}/{iterations}, điểm = `{best:.3f}`"
            time.sleep(0.05)

    final = history[-1]
    _record("2_attack", final_score=final, iterations=iterations, sigma=sigma, seed=seed,
            crossed_at=crossed_at if crossed_at is not None else -1)
    if crossed_at is not None:
        status = f"🚨 **Attack SUCCEEDED**: vượt ngưỡng ở vòng {crossed_at}"
    else:
        status = "🛡️ **Attack FAILED**: chưa vượt ngưỡng"
    yield _climb_figure(history, threshold, iterations), (
        f"### {status}\n"
        f"Điểm ban đầu `{history[0]:.3f}` → điểm cuối `{final:.3f}` sau {iterations} lần hỏi matcher."
    )


# ---------- Tab 3: BioHashing ----------


def _bits_figure(bits_a: np.ndarray, bits_b: np.ndarray, key_a: int, key_b: int) -> Figure:
    fig = Figure(figsize=(8, 1.9))
    ax = fig.subplots()
    ax.imshow(np.vstack([bits_a, bits_b]), cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set_yticks([0, 1], [f"key = {key_a}", f"key = {key_b}"])
    ax.set_xlabel("128 bit của template (đen = 1, trắng = 0)")
    fig.tight_layout()
    return fig


def _fmt_bits(bits: np.ndarray, n: int = 32) -> str:
    s = "".join(map(str, bits[:n]))
    return " ".join(s[i : i + 8] for i in range(0, n, 8))


def biohash_demo(img: str, key_a: int, key_b: int):
    """Same face, two keys: shows renewability and unlinkability of BioHashing.
    Cùng khuôn mặt, hai key: minh họa tính thu hồi và không liên kết của BioHashing.
    """
    e = _embed(img, "ảnh")
    key_a, key_b = int(key_a), int(key_b)
    bits_a, bits_b = biohash(e, key_a), biohash(e, key_b)
    score = hamming_similarity(bits_a, bits_b)
    _record("3_biohash", key_a=key_a, key_b=key_b, hamming=score)

    if key_a == key_b:
        note = "Cùng key → template giống hệt → **MATCH**."
    else:
        note = "Khác key → ≈ 0.5, tức **như đoán ngẫu nhiên**: không liên kết được hai CSDL, key cũ thu hồi an toàn."
    md = (
        f"**32 bit đầu (key = {key_a}):** `{_fmt_bits(bits_a)}`\n\n"
        f"**32 bit đầu (key = {key_b}):** `{_fmt_bits(bits_b)}`\n\n"
        f"### Hamming similarity = `{score:.3f}`\n{note}"
    )
    return _bits_figure(bits_a, bits_b, key_a, key_b), md


# ---------- Tab 4: BioHashing vs PolyProtect ----------


def _linkage_figure(values: list[float]) -> Figure:
    labels = ["Không bảo vệ\n(cosine)", "BioHashing\n(Hamming)", "PolyProtect\n(cosine)"]
    random_level = [None, 0.5, 0.0]
    colors = ["#c0392b", "#2980b9", "#27ae60"]
    fig = Figure(figsize=(7, 3.8))
    ax = fig.subplots()
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value, level in zip(bars, values, random_level):
        ax.text(bar.get_x() + bar.get_width() / 2, max(value, 0) + 0.03, f"{value:.2f}", ha="center", fontsize=11)
        # Dashed mark = score of two unrelated templates on that method's scale
        # / Vạch đứt = điểm của hai template không liên quan, theo thang của phương pháp đó
        if level is not None:
            ax.hlines(level, bar.get_x() - 0.05, bar.get_x() + bar.get_width() + 0.05, colors="black", linestyles="--")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylim(-0.3, 1.1)
    ax.set_ylabel("Điểm liên kết giữa 2 CSDL")
    ax.set_title("Cùng một người, hai hệ thống dùng hai key khác nhau")
    fig.tight_layout()
    return fig


def compare_methods(img_db_a: str, img_db_b: str, key_a: int, key_b: int, overlap: int):
    """Can an attacker link the same person across two databases protected with different keys?
    Kẻ tấn công có liên kết được cùng một người giữa hai CSDL dùng hai key khác nhau không?
    """
    ea = _embed(img_db_a, "ảnh CSDL A")
    eb = _embed(img_db_b, "ảnh CSDL B")
    key_a, key_b, overlap = int(key_a), int(key_b), int(overlap)
    if key_a == key_b:
        gr.Warning("Hai key đang giống nhau: đây là cùng một hệ thống, không phải hai CSDL độc lập.")

    unprotected = cosine_similarity(ea, eb)

    bio_same = hamming_similarity(biohash(ea, key_a), biohash(eb, key_a))
    bio_cross = hamming_similarity(biohash(ea, key_a), biohash(eb, key_b))

    ca, exa = generate_key(key_a)
    cb, exb = generate_key(key_b)
    pa = polyprotect(ea, ca, exa, overlap)
    poly_same = cosine_similarity(pa, polyprotect(eb, ca, exa, overlap))
    poly_cross = cosine_similarity(pa, polyprotect(eb, cb, exb, overlap))

    # A single key pair can be unlucky, so the chart shows the average over many pairs
    # / Một cặp key có thể "xui", nên biểu đồ dùng trung bình trên nhiều cặp key
    bio_many, poly_many = cross_key_scores(ea, eb, key_a, key_b, overlap)
    poly_linkable = float(np.mean(np.abs(poly_many) > LINKABLE_LEVEL))

    _record("4_compare", key_a=key_a, key_b=key_b, overlap=overlap, unprotected=unprotected,
            biohash_same_key=bio_same, biohash_cross_key=bio_cross,
            poly_same_key=poly_same, poly_cross_key=poly_cross,
            biohash_cross_mean=float(bio_many.mean()), poly_cross_mean=float(poly_many.mean()),
            poly_linkable_rate=poly_linkable)

    table = pd.DataFrame(
        [
            ["Không bảo vệ", "128 số thực", unprotected, unprotected, unprotected, "0 (cosine)"],
            ["BioHashing", "128 bit", bio_same, bio_cross, bio_many.mean(), "0.5 (Hamming)"],
            [f"PolyProtect (overlap {overlap})", f"{pa.size} số thực", poly_same, poly_cross,
             poly_many.mean(), "0 (cosine)"],
        ],
        columns=["Phương pháp", "Template", "Cùng key (xác thực)", f"Khác key ({key_a} vs {key_b})",
                 f"Khác key, TB {N_KEY_PAIRS} cặp", "Mức ngẫu nhiên"],
    ).round(3)

    caption = (
        f"Biểu đồ: trung bình trên {N_KEY_PAIRS} cặp key ngẫu nhiên. "
        "Không bảo vệ: không có key nên không thu hồi được, điểm cao → **liên kết được**. "
        "BioHashing ≈ 0.5 và PolyProtect ≈ 0 đều là **mức ngẫu nhiên** trên thang của mình. "
        "Cột *Cùng key* cho thấy hệ thống vẫn nhận ra đúng người.\n\n"
        f"Với PolyProtect, {poly_linkable:.0%} số cặp key ngẫu nhiên vẫn cho |cosine| > {LINKABLE_LEVEL} "
        "(xem Bảng 19 trong doc: chọn (C, E) ngẫu nhiên vs chọn chặt)."
    )
    if abs(poly_cross) > LINKABLE_LEVEL:
        caption += f"\n\n⚠️ Cặp key {key_a}/{key_b} rơi vào nhóm liên kết được (cosine {poly_cross:.2f})."
        # Explain only when the cause is actually present / Chỉ giải thích khi đúng là có nguyên nhân này
        if np.argmax(exa == 1) == np.argmax(exb == 1):
            caption += (
                " Hai key đặt số mũ 1 ở cùng vị trí; embedding đã chuẩn hóa có giá trị nhỏ (|v| < 1) "
                "nên hạng tử bậc 1 chi phối và hai template gần như tỉ lệ với nhau."
            )
    return _linkage_figure([unprotected, float(bio_many.mean()), float(poly_many.mean())]), table, caption


# ---------- UI / Giao diện ----------


def _examples(component_list, rows):
    if rows:
        gr.Examples(examples=rows, inputs=component_list, label="Ảnh mẫu trong images/")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="BTP Demo — Nhóm 8") as app:
        gr.Markdown(
            "# Biometric Template Protection — Demo\n"
            f"Facenet (DeepFace) · embedding 128 chiều · ngưỡng mặc định {DEFAULT_THRESHOLD:.2f} "
            "(mặc định DeepFace cho Facenet)"
        )

        with gr.Tab("1. Enroll & Match"):
            with gr.Row():
                t1_a = gr.Image(type="filepath", label="Ảnh đăng ký (enroll)", height=260)
                t1_b = gr.Image(type="filepath", label="Ảnh xác thực (probe)", height=260)
            t1_th = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.01, label="Ngưỡng cosine")
            t1_btn = gr.Button("So khớp", variant="primary")
            t1_out = gr.Markdown()
            _examples([t1_a, t1_b], _pair_examples())
            t1_btn.click(enroll_and_match, [t1_a, t1_b, t1_th], t1_out)

        with gr.Tab("2. Hill-climbing Attack"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2_img = gr.Image(type="filepath", label="Nạn nhân (template trong CSDL)", height=260)
                    t2_iter = gr.Slider(100, 1000, value=300, step=50, label="Số vòng lặp")
                    t2_sigma = gr.Slider(0.05, 0.5, value=0.2, step=0.05, label="Độ lớn nhiễu σ")
                    t2_seed = gr.Number(value=0, precision=0, label="Seed")
                    t2_th = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.01, label="Ngưỡng matcher")
                    t2_btn = gr.Button("Chạy Attack", variant="stop")
                with gr.Column(scale=2):
                    t2_plot = gr.Plot(label="Điểm theo vòng lặp")
                    t2_out = gr.Markdown()
            _examples([t2_img], _single_examples())
            t2_btn.click(run_attack, [t2_img, t2_iter, t2_sigma, t2_seed, t2_th], [t2_plot, t2_out])

        with gr.Tab("3. BioHashing"):
            with gr.Row():
                with gr.Column(scale=1):
                    t3_img = gr.Image(type="filepath", label="Ảnh", height=260)
                    with gr.Row():
                        t3_ka = gr.Number(value=42, precision=0, label="key_a (hiện tại)")
                        t3_kb = gr.Number(value=99, precision=0, label="key_b (sau khi thu hồi)")
                    t3_btn = gr.Button("Tạo protected templates", variant="primary")
                with gr.Column(scale=2):
                    t3_plot = gr.Plot(label="Hai template BioHash")
                    t3_out = gr.Markdown()
            _examples([t3_img], _single_examples())
            t3_btn.click(biohash_demo, [t3_img, t3_ka, t3_kb], [t3_plot, t3_out])

        with gr.Tab("4. BioHashing vs PolyProtect"):
            with gr.Row():
                t4_a = gr.Image(type="filepath", label="Ảnh trong CSDL A", height=240)
                t4_b = gr.Image(type="filepath", label="Ảnh trong CSDL B (cùng người, lần chụp khác)", height=240)
            with gr.Row():
                t4_ka = gr.Number(value=42, precision=0, label="Key hệ thống A")
                t4_kb = gr.Number(value=7, precision=0, label="Key hệ thống B")
                t4_ov = gr.Slider(0, 4, value=2, step=1, label="PolyProtect overlap")
            t4_btn = gr.Button("So sánh 3 phương pháp", variant="primary")
            t4_plot = gr.Plot(label="Mức liên kết giữa 2 CSDL")
            t4_table = gr.Dataframe(label="Số liệu chi tiết", interactive=False)
            t4_out = gr.Markdown()
            _examples([t4_a, t4_b], _pair_examples(same_person_only=True))
            t4_btn.click(compare_methods, [t4_a, t4_b, t4_ka, t4_kb, t4_ov], [t4_plot, t4_table, t4_out])

    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
