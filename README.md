# Biometric Template Protection — Demo (Nhóm 8, MTH042)

Demo Gradio 4 tab cho phần Người 4: so khớp khuôn mặt, tấn công hill-climbing, BioHashing, và so sánh BioHashing với PolyProtect.

## Cài đặt (một lần, cần mạng)

Yêu cầu Python 3.12.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Trên Windows (PowerShell hoặc cmd):

```bat
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Các lệnh bên dưới viết cho macOS/Linux. Trên Windows, thay `.venv/bin/python` bằng `.venv\Scripts\python`, ví dụ `.venv\Scripts\python demo_app.py`. Nếu `pip` báo không tìm thấy bản `tensorflow` phù hợp cho Windows, chạy demo trong WSL2 (Ubuntu) theo các lệnh macOS/Linux.

`opencv-python` được ghim ở bản 4.x vì OpenCV 5 không còn kèm file Haar cascade mà DeepFace dùng để tìm khuôn mặt.

## Chuẩn bị ảnh

### Ảnh mẫu có sẵn

Repo kèm sẵn 6 ảnh mẫu (3 người, mỗi người 2 ảnh) để chạy thử ngay sau khi clone. Ảnh lấy nguyên từ bộ ảnh test công khai của thư viện DeepFace ([serengil/deepface](https://github.com/serengil/deepface), thư mục `tests/unit/dataset`, giấy phép MIT):

| Ảnh trong repo | File gốc |
|---|---|
| `mauA_1.jpg`, `mauA_2.jpg` | `img1.jpg`, `img2.jpg` |
| `mauB_1.jpg`, `mauB_2.jpg` | `img13.jpg`, `img14.jpg` |
| `mauC_1.jpg`, `mauC_2.jpg` | `img18.jpg`, `img19.jpg` |

App liệt kê mọi ảnh trong `images/`. Khi dùng ảnh của nhóm, xóa `images/mau*.jpg` trên máy mình (đừng commit thao tác xóa) để ô Ảnh mẫu chỉ còn ảnh nhóm, rồi chạy lại `scripts/precompute.py`.

### Ảnh của nhóm

Đặt 6 ảnh vào `images/`, đặt tên theo dạng `<người>_<số>.jpg`:

```text
images/nguoiA_1.jpg  images/nguoiA_2.jpg
images/nguoiB_1.jpg  images/nguoiB_2.jpg
images/nguoiC_1.jpg  images/nguoiC_2.jpg
```

Ảnh chụp thẳng mặt, đủ sáng, mỗi ảnh một người. Phần trước dấu `_` cuối cùng được coi là mã người: app dùng nó để ghép sẵn cặp "cùng người" và "khác người" trong ô Ảnh mẫu.

Ảnh khuôn mặt là dữ liệu sinh trắc học nên `images/` nằm trong `.gitignore`: ảnh của nhóm không được đẩy lên GitHub. Chỉ các file `images/mau*.jpg` (ảnh mẫu công khai ở trên) được Git theo dõi, nên đừng đặt ảnh nhóm theo tên `mau*`.

## Chuẩn bị chạy offline

```bash
.venv/bin/python scripts/precompute.py
```

Lệnh này:

1. Tải weights Facenet (~92MB) vào `models/` (nằm trong thư mục dự án, không phải `~/.deepface`).
2. Tính embedding của 6 ảnh và lưu vào `cache/embeddings.npz`.
3. In bảng số cho slide: cosine từng cặp ảnh, BioHash khác key, PolyProtect cùng key / khác key / trung bình 200 cặp key.

Sau bước này app chạy được không cần mạng. Khi chuyển sang máy thuyết trình: chép `models/`, `cache/`, `images/` cùng code, rồi tạo lại `.venv` trên máy đó (venv không chép sang máy khác được).

## Chạy demo

```bash
.venv/bin/python demo_app.py
```

Mở http://127.0.0.1:7860.

| Tab | Chạy sau | Nội dung |
|---|---|---|
| 1. Enroll & Match | Người 1 | Cosine giữa 2 embedding Facenet, so với ngưỡng |
| 2. Hill-climbing Attack | Người 1 | Kẻ tấn công chỉ nhận điểm từ matcher, leo dần từ vector ngẫu nhiên |
| 3. BioHashing | Người 2 | Cùng ảnh, key_a vs key_b: 32 bit đầu và Hamming similarity |
| 4. BioHashing vs PolyProtect | Người 3 | Mức liên kết cùng một người giữa 2 CSDL dùng 2 key khác nhau |

Mỗi lần bấm nút, số liệu được ghi vào `results/demo_results.csv` để đưa lên slide.

## Tham số mặc định và nguồn

| Tham số | Giá trị | Nguồn |
|---|---|---|
| Mô hình | Facenet qua DeepFace, embedding 128 chiều, chuẩn hóa L2 | Schroff et al., CVPR 2015; DeepFace |
| Ngưỡng | cosine similarity ≥ 0.60 | `deepface/config/threshold.py`: Facenet cosine distance 0.40 |
| BioHashing | Chiếu lên 128 vector trực chuẩn sinh từ key, ngưỡng 0 | Teoh, Ngo & Goh, Pattern Recognition 2004 |
| PolyProtect | m = 5, C: 5 số nguyên khác 0, đôi một khác nhau trong [−50, 50]; E: hoán vị của 1..5; overlap 0–4 → độ dài 26/32/42/63/124 | Hahn & Marcel, IEEE T-BIOM 2022 (arXiv:2110.00434) |
| Hill-climbing | 300 vòng, mỗi vòng 1 truy vấn, nhiễu σ = 0.2, giữ nếu điểm tăng | Mô tả trong doc nhóm, Phần E |

## Lưu ý khi trình bày Tab 4

Biểu đồ Tab 4 hiển thị **trung bình trên 200 cặp key ngẫu nhiên**, còn bảng hiển thị thêm số của đúng cặp key đang nhập.

Với PolyProtect, một cặp key cụ thể có thể vẫn liên kết được: khi hai key đặt số mũ 1 ở cùng vị trí, vì embedding đã chuẩn hóa có giá trị nhỏ (|v| < 1) nên hạng tử bậc 1 chi phối. Cặp 42/99 (dùng ở Tab 3) rơi đúng vào trường hợp này, nên Tab 4 mặc định dùng 42/7. Nhập 42/99 ở Tab 4 nếu muốn minh họa trường hợp này. Trên ảnh thử, khoảng 15% số cặp key ngẫu nhiên có |cosine| > 0.5. Điều này khớp với Bảng 19 trong doc: chọn (C, E) ngẫu nhiên cho D↔sys 0,14–0,16, chọn chặt cho 0,03–0,04. App tự hiện cảnh báo khi gặp cặp key như vậy.

## Kiểm thử

```bash
.venv/bin/python -m pytest -q
```

Kiểm tra: ví dụ số Bảng 17 trong doc (p₁ ≈ 13,21), độ dài template theo overlap, tính hợp lệ của (C, E), BioHash cùng key = 1.0 và khác key ≈ 0.5, PolyProtect khác key trung bình ≈ 0, hill-climbing vượt ngưỡng sau 300 vòng.

## Cấu trúc

```text
demo_app.py            Giao diện Gradio 4 tab
btp/embedding.py       Embedding Facenet + cache theo nội dung file
btp/biohash.py         BioHashing
btp/polyprotect.py     PolyProtect
btp/attack.py          Hill-climbing
btp/linkage.py         Điểm khác key trên nhiều cặp key
btp/metrics.py         Cosine, Hamming, ngưỡng
scripts/precompute.py  Tải weights, cache embedding, in bảng số
tests/test_btp.py      Unit test
```
