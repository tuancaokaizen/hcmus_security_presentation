import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
CACHE_FILE = ROOT / "cache" / "embeddings.npz"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Keep Facenet weights inside the project so the demo runs offline on any machine
# / Giữ weights Facenet trong thư mục dự án để demo chạy offline trên máy nào cũng được
os.environ.setdefault("DEEPFACE_HOME", str(ROOT / "models"))

MODEL_NAME = "Facenet"
_cache: dict[str, np.ndarray] | None = None


def list_images() -> list[Path]:
    """Demo images in images/, sorted by name.
    Ảnh demo trong images/, sắp theo tên.
    """
    if not IMAGES_DIR.exists():
        return []
    return sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def get_embedding(image_path: str | Path) -> np.ndarray:
    """128-dim L2-normalized Facenet embedding, cached by file content.
    Embedding Facenet 128 chiều đã chuẩn hóa L2, cache theo nội dung file.
    """
    data = Path(image_path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    cache = _load_cache()
    if digest in cache:
        return cache[digest]

    # Decode bytes ourselves: DeepFace rejects paths with non-ASCII characters such as "Tài liệu"
    # / Tự giải mã ảnh: DeepFace từ chối đường dẫn có ký tự không phải ASCII như "Tài liệu"
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh {Path(image_path).name}")

    # Import lazily because TensorFlow takes several seconds to load / Import muộn vì TensorFlow mất vài giây để nạp
    from deepface import DeepFace

    try:
        faces = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            enforce_detection=True,
            l2_normalize=True,
        )
    except ValueError as exc:
        # Translate only the "no face" error; other errors (e.g. broken install) must stay visible
        # / Chỉ dịch lỗi "không có mặt"; lỗi khác (ví dụ cài đặt hỏng) phải giữ nguyên để thấy được
        if "could not be detected" not in str(exc):
            raise
        raise ValueError(f"Không tìm thấy khuôn mặt trong ảnh {Path(image_path).name}") from exc

    # Use the largest face when the photo contains several / Ảnh có nhiều mặt thì lấy mặt lớn nhất
    face = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
    embedding = np.asarray(face["embedding"], dtype=float)

    cache[digest] = embedding
    _save_cache(cache)
    return embedding


def _load_cache() -> dict[str, np.ndarray]:
    global _cache
    if _cache is None:
        if CACHE_FILE.exists():
            with np.load(CACHE_FILE) as npz:
                _cache = {k: npz[k] for k in npz.files}
        else:
            _cache = {}
    return _cache


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_FILE, **cache)
