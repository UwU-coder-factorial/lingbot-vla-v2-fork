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

### Cách A: Kiểm tra nhanh môi trường có sẵn (Tránh cài đè / cài trùng)
Nếu máy train đã có sẵn Conda hoặc môi trường Python:
```bash
# 1. Kiểm tra xem đã có đủ thư viện chưa
python tools/check_environment.py

# 2. Hoặc tự động cài CHỈ các gói còn thiếu (không cài lại gói đã có)
python tools/check_environment.py --install
```

### Cách B: Tạo môi trường mới từ đầu (Nếu máy chưa có gì)
```bash
# Linux (Script tự động):
bash tools/create_train_env.sh --env-name lingbotvla
conda activate lingbotvla

# Hoặc thủ công:
conda create -n lingbotvla python=3.12 -y
conda activate lingbotvla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install h5py numpy huggingface_hub tqdm opencv-python transformers peft einops safetensors
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

## 4. Huấn Luyện Thực Tế (Training Pipeline)

### Cách 1: Chạy 1-Click tự động (Khuyến nghị)

Script tự động nhận diện data/model, tiền xử lý và chạy train:

#### TH 1.1: Data ĐÃ CÓ SẴN trên máy (Bỏ qua tải HuggingFace)
Truyền trực tiếp đường dẫn thư mục chứa các file `.hdf5` / `.h5`:

* **Trên Linux**:
  ```bash
  bash scripts/train_remote_openarm.sh --data-dir /duong_dan/toi/thu_muc_data_co_san
  # hoặc truyền tắt:
  bash scripts/train_remote_openarm.sh /duong_dan/toi/thu_muc_data_co_san
  ```

* **Trên Windows PowerShell**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/train_remote_openarm.ps1 -DataDir "D:\duong_dan\data_co_san"
  ```

#### TH 1.2: Cần tải data tự động từ HuggingFace qua Mirror siêu tốc
Nếu mạng chậm, dùng cờ `--use-mirror` để tải qua CDN hf-mirror.com / ModelScope:

* **Trên Linux**:
  ```bash
  bash scripts/train_remote_openarm.sh --use-mirror
  ```

* **Trên Windows PowerShell**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/train_remote_openarm.ps1 -UseMirror
  ```

#### TH 1.3: Model weights đã lưu sẵn ở thư mục khác (ổ cứng ngoài / SSD)
* **Linux**:
  ```bash
  bash scripts/train_remote_openarm.sh --data-dir /path/to/data --model-dir /path/to/custom_models
  ```
* **Windows**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/train_remote_openarm.ps1 -DataDir "D:\data" -ModelDir "E:\models"
  ```

---

### Cách 2: Chạy thủ công từng bước độc lập (Manual Steps)

Nếu muốn tự kiểm soát từng bước:

#### Bước 4.1: Tải Model Base (Đã tối ưu dung lượng)
```bash
# Tải LingBot-VLA 2.0 (tự động bỏ qua teacher thừa để giảm ~4GB)
python scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir models --use_mirror

# Tải Qwen3-VL Tokenizer/Config (~30MB, bỏ qua 9GB weights thừa)
python scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir models --use_mirror
```

#### Bước 4.2: Tiền xử lý & lọc dữ liệu
```bash
python tools/prepare_openarm_data.py \
    --input_dir "/duong_dan/data_tho" \
    --output_dir "data/openarm_processed" \
    --txt_path "assets/training_data/openarm_train.txt"
```

#### Bước 4.3: Tính thống kê chuẩn hóa góc khớp (Norm Stats)
```bash
python scripts/compute_norm_stats.py \
    --config configs/vla/norm_compute/post_data.yaml \
    --data.train_path "assets/training_data/openarm_train.txt" \
    --data.robot_config_root configs/robot_configs \
    --data.robot_name openarm \
    --data.norm_path "assets/norm_stats/openarm_dual.json"
```

#### Bước 4.4: Khởi chạy huấn luyện
```bash
python tasks/vla/train_lingbotvla.py --config configs/vla/real_robot/openarm_expert_4090.yaml
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
