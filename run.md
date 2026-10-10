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

## 4. Huấn Luyện Thực Tế (Training Pipeline)

### Trường hợp 1: Data ĐÃ CÓ SẴN trên máy (Bỏ qua tải HuggingFace)
Truyền trực tiếp đường dẫn thư mục chứa các file `.hdf5`:

* **Trên Linux**:
  ```bash
  bash scripts/train_remote_openarm.sh --data-dir /duong_dan/toi/thu_muc_data_co_san
  # hoặc truyền tắt:
  bash scripts/train_remote_openarm.sh /duong_dan/toi/thu_muc_data_co_san
  ```

* **Trên Windows PowerShell**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/train_remote_openarm.ps1 -DataDir "D:\path\to\data_co_san"
  ```

### Trường hợp 2: Tải data tự động từ HuggingFace qua Mirror siêu tốc
Nếu mạng chậm, dùng cờ `--use-mirror` để tải qua CDN hf-mirror.com / ModelScope:

* **Trên Linux**:
  ```bash
  bash scripts/train_remote_openarm.sh --use-mirror
  ```

* **Trên Windows PowerShell**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/train_remote_openarm.ps1 -UseMirror
  ```

> [!NOTE]
> **Tối ưu dung lượng model:** Script tải đã được tối ưu:
> 1. Chỉ tải Tokenizer & Config của `Qwen3-VL-4B-Instruct` (~30MB), **bỏ qua 9GB weights thừa**.
> 2. Bỏ qua các teacher thừa (`depth/`, `dino_video/`) của LingBot, tiết kiệm thêm ~4GB.

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


