# HƯỚNG DẪN TRIỂN KHAI VÀ HUẤN LUYỆN TRÊN MÁY MỚI (TURNKEY GUIDE)

Tài liệu này cung cấp đầy đủ các bước từ lúc clone repository đến khi hoàn tất huấn luyện Action Expert cho robot OpenArm trên máy mới (yêu cầu tối thiểu: 1x RTX 4090 24GB).

---

## 1. Clone Repository

Mở terminal trên máy mới và chạy:

```bash
git clone https://github.com/UwU-coder-factorial/lingbot-vla-v2-fork.git
cd lingbot-vla-v2-fork
```

---

## 2. Thiết Lập Môi Trường (Environment Setup)

### Cách A: Dùng script tự động (Khuyến nghị cho Linux / Ubuntu)
```bash
# Đảm bảo đã cài Conda / Miniconda
bash tools/create_train_env.sh --env-name lingbotvla
conda activate lingbotvla
```

### Cách B: Cài đặt thủ công qua Conda / Pip (Windows hoặc Linux)
```bash
conda create -n lingbotvla python=3.12 -y
conda activate lingbotvla

# Cài đặt PyTorch tương thích CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Cài đặt các thư viện bắt buộc cho robot data & training
pip install h5py numpy huggingface_hub tqdm opencv-python transformers peft
```

---

## 3. Kiểm Thử Hệ Thống (Smoke Test / Unit Tests)

Trước khi train, chạy bộ test case (không mock) để xác nhận hệ thống và định dạng HDF5 tương thích 100%:

```bash
# 1. Tải 1 tập dữ liệu mẫu từ HuggingFace về test (nhẹ, ~300MB)
python tools/download_sample_data.py --num 1 --target_dir data/openarm_samples

# 2. Chạy test suite (17/17 test cases bao phủ logic, kinematics 16-DoF, RAM budget)
python tests/test_openarm_suite.py
```

*Khi màn hình báo `OK (17/17 PASSED)` là hệ thống sẵn sàng huấn luyện.*

---

## 4. Huấn Luyện Thực Tế 1-Click (All-in-One Training)

Script tự động thực hiện từ A đến Z:
1. Tải Base Pretrained Model (`lingbot-vla-v2-6b` và `Qwen3-VL-4B-Instruct`).
2. Tải toàn bộ 45 episodes dataset OpenArm từ HuggingFace (`Tuyen062004/data_openarm_8_10`).
3. Tiền xử lý, lọc bỏ NaN, định dạng HDF5 chuẩn.
4. Tính bảng thống kê chuẩn hóa góc khớp (`assets/norm_stats/openarm_dual.json`).
5. Khởi chạy huấn luyện Action Expert (khóa VLM backbone, tối ưu VRAM trong ngưỡng 16–18 GB).

### Trên Linux:
```bash
bash scripts/train_remote_openarm.sh
```

### Trên Windows PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_remote_openarm.ps1
```

---

## 5. Theo Dõi và Kết Quả

* **Giám sát VRAM**:
  ```bash
  nvidia-smi -l 1
  ```
* **Thư mục lưu Checkpoint**: `output/openarm_expert_4090/`
* **Đánh giá Open-Loop sau khi train**:
  ```bash
  python scripts/open_loop_eval.py --config configs/vla/real_robot/openarm_expert_4090.yaml
  ```

