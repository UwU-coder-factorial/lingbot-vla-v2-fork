# HƯỚNG DẪN TRIỂN KHAI VÀ HUẤN LUYỆN TRÊN MÁY MỚI (TURNKEY GUIDE)

> [!IMPORTANT]
> **DÀNH CHO NGƯỜI DÙNG MUỐN CHẠY NHANH (ALL-IN-ONE MASTER SCRIPT)**
> Toàn bộ pipeline đã được gói gọn trong script duy nhất: tự động kiểm tra thư viện (hỏi [y/n]), tính dung lượng model cần tải (hỏi [y/n]), nạp dữ liệu từ thư mục chỉ định, tiền xử lý và khởi chạy huấn luyện.

---

## ⚡ 1. KHỞI CHẠY NHANH: SCRIPT TỔNG ALL-IN-ONE (KHUYẾN NGHỊ)

Sau khi kéo repo về máy:
```bash
git clone https://github.com/UwU-coder-factorial/lingbot-vla-v2-fork.git
cd lingbot-vla-v2-fork
```

Chạy ngay một trong các lệnh sau:

### Trường hợp A: Dữ liệu ĐÃ CÓ SẴN trong một thư mục trên máy
```bash
# Trên Linux / Ubuntu:
bash scripts/train_all_in_one.sh --data-dir /duong_dan/toi/thu_muc_data_cua_ban

# Hoặc dùng Python trực tiếp (chạy được cả Windows / Linux / macOS):
python scripts/train_all_in_one.py --data-dir /duong_dan/toi/thu_muc_data_cua_ban

# Trên Windows PowerShell:
powershell -ExecutionPolicy Bypass -File scripts/train_all_in_one.ps1 -data-dir "D:\path\to\data"
```

### Trường hợp B: Muốn tăng tốc độ tải mạng (Dùng CDN Mirror Châu Á)
```bash
# Thêm cờ --use-mirror để tải model siêu tốc qua hf-mirror.com:
bash scripts/train_all_in_one.sh --data-dir /duong_dan/data --use-mirror
```

### Trường hợp C: Chạy tự động không cần hỏi (Tự động bấm 'y' mọi prompt)
```bash
bash scripts/train_all_in_one.sh --data-dir /duong_dan/data --use-mirror -y
```

### 📋 Quy trình 5 bước Script Tổng sẽ tự động thực hiện:
1. **Kiểm tra thư viện**: Quét toàn bộ package (`torch`, `h5py`, `transformers`, `peft`...). Nếu đã có sẵn trên máy $\to$ **BỎ QUA, không cài đè**. Nếu thiếu $\to$ Liệt kê tên các gói và dung lượng ước tính, prompt `[y/n]` để người dùng xác nhận cài đặt.
2. **Kiểm tra Model**: Quét thư mục `models/`. Nếu chưa có model $\to$ Báo dung lượng cụ thể (~25.6 GB cho LingBot 6B và ~30 MB cho Qwen3-VL tokenizer), prompt `[y/n]` trước khi bắt đầu tải.
3. **Nạp Dữ liệu**: Quét các file `.hdf5` / `.h5` / `.npz` từ đường dẫn `--data-dir` bạn cung cấp.
4. **Tiền xử lý & Chuẩn hóa**: Lọc bỏ NaN, chuẩn hóa vector 16-DoF, tính toán bảng phân phối góc khớp `assets/norm_stats/openarm_dual.json`.
5. **Khởi chạy Huấn luyện**: Tự động nạp cấu hình tối ưu 1x RTX 4090 (đóng băng VLM, VRAM 16–18 GB) và train model.

---

## 2. Kiểm Thử Hệ Thống Trước Khi Train (Tùy chọn)

Nếu muốn chạy kiểm thử smoke test (17/17 test cases bao phủ logic, kinematics 16-DoF, RAM budget, không dùng mock):

```bash
# 1. Tải 1 tập dữ liệu mẫu từ HuggingFace về test (nhẹ, ~300MB)
python tools/download_sample_data.py --num 1 --target_dir data/openarm_samples

# 2. Chạy test suite
python tests/test_openarm_suite.py
```

---

## 3. Chạy Thủ Công Từng Bước Độc Lập (Nâng cao)

Nếu không dùng script tổng mà muốn tự kiểm soát từng giai đoạn:

### Bước 3.1: Kiểm tra thư viện riêng lẻ
```bash
# Chỉ kiểm tra:
python tools/check_environment.py

# Hoặc tự cài các gói còn thiếu:
python tools/check_environment.py --install
```

### Bước 3.2: Tải Model riêng lẻ
```bash
python scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir models --use_mirror
python scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir models --use_mirror
```

### Bước 3.3: Tiền xử lý dữ liệu OpenArm
```bash
python tools/prepare_openarm_data.py \
    --input_dir "/duong_dan/data_tho" \
    --output_dir "data/openarm_processed" \
    --txt_path "assets/training_data/openarm_train.txt"
```

### Bước 3.4: Tính thống kê chuẩn hóa góc khớp (Norm Stats)
```bash
python scripts/compute_norm_stats.py \
    --config configs/vla/norm_compute/post_data.yaml \
    --data.train_path "assets/training_data/openarm_train.txt" \
    --data.robot_config_root configs/robot_configs \
    --data.robot_name openarm \
    --data.norm_path "assets/norm_stats/openarm_dual.json"
```

### Bước 3.5: Huấn luyện Action Expert
```bash
python tasks/vla/train_lingbotvla.py --config configs/vla/real_robot/openarm_expert_4090.yaml
```

---

## 4. Theo Dõi và Kết Quả

* **Giám sát VRAM**:
  ```bash
  nvidia-smi -l 1
  ```
* **Thư mục lưu Checkpoint**: `output/openarm_expert_4090/`
* **Đánh giá Open-Loop sau khi train**:
  ```bash
  python scripts/open_loop_eval.py --config configs/vla/real_robot/openarm_expert_4090.yaml
  ```
