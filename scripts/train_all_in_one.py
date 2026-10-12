"""
SCRIPT ALL-IN-ONE MASTER: KIỂM TRA MÔI TRƯỜNG -> TẢI MODEL -> XỬ LÝ DATA -> HUẤN LUYỆN
Hỗ trợ chạy trực tiếp trên cả Linux và Windows.
"""

import os
import sys
import argparse
import subprocess
import importlib
from pathlib import Path

# Đảm bảo UTF-8 cho terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Danh sách thư viện bắt buộc và dung lượng ước tính khi cài (MB)
REQUIRED_PACKAGES = {
    "torch": ("torch", 2400),
    "torchvision": ("torchvision", 10),
    "numpy": ("numpy", 30),
    "h5py": ("h5py", 15),
    "transformers": ("transformers", 15),
    "huggingface_hub": ("huggingface_hub", 5),
    "tqdm": ("tqdm", 1),
    "cv2": ("opencv-python", 60),
    "peft": ("peft", 5),
    "einops": ("einops", 2),
    "safetensors": ("safetensors", 2),
}

# Thông tin models cần tải
MODELS_INFO = {
    "lingbot-vla-v2-6b": {
        "repo_id": "robbyant/lingbot-vla-v2-6b",
        "size_gb": 25.6,
        "desc": "Base model LingBot-VLA 2.0 (MoE 6B parameters)",
        "check_files": ["model-00001-of-00006.safetensors", "model-00006-of-00006.safetensors"],
    },
    "Qwen3-VL-4B-Instruct": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "size_gb": 0.03,  # ~30MB tokenizer & configs
        "desc": "Tokenizer & Processor của Qwen3-VL (đã cắt 9GB weights thừa)",
        "check_files": ["tokenizer.json", "vocab.json"],
    },
}


def ask_user_confirm(prompt_msg: str) -> bool:
    """Prompt hỏi xác nhận người dùng y/n."""
    while True:
        try:
            choice = input(f"{prompt_msg} [y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n[!] Đã hủy thao tác.")
            return False
        if choice in ["y", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False
        else:
            print("Vui lòng nhập 'y' để đồng ý hoặc 'n' để hủy.")


# ==============================================================================
# BƯỚC 1 & 2: KIỂM TRA THƯ VIỆN & PROMPT XÁC NHẬN CÀI ĐẶT
# ==============================================================================
def step_check_and_install_packages(yes_all: bool = False):
    print("\n" + "=" * 65)
    print("  BƯỚC 1: KIỂM TRA THƯ VIỆN VÀ MÔI TRƯỜNG PYTHON")
    print("=" * 65)

    missing_packages = []
    est_download_mb = 0

    for mod_name, (pip_name, est_mb) in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "ok")
            print(f"  [+] {pip_name:20s}: ĐÃ CÓ ({ver})")
        except ImportError:
            print(f"  [-] {pip_name:20s}: CHƯA CÓ (Ước tính: ~{est_mb} MB)")
            missing_packages.append(pip_name)
            est_download_mb += est_mb

    if not missing_packages:
        print("\n[+] Toàn bộ thư viện cần thiết ĐÃ ĐẦY ĐỦ 100%.")
        return True

    print("\n" + "-" * 65)
    print(f"[!] Phát hiện {len(missing_packages)} thư viện còn thiếu:")
    for pkg in missing_packages:
        print(f"    - {pkg}")
    
    if est_download_mb >= 1000:
        size_str = f"{est_download_mb / 1024:.2f} GB"
    else:
        size_str = f"{est_download_mb} MB"
    print(f"[*] Tổng dung lượng ước tính cần tải: ~{size_str}")

    if not yes_all:
        confirmed = ask_user_confirm(f"[?] Bạn có muốn cài đặt {len(missing_packages)} thư viện trên không?")
        if not confirmed:
            print("[X] Người dùng hủy cài đặt thư viện. Dừng chương trình.")
            sys.exit(0)

    print("\n[*] Đang tiến hành cài đặt...")
    cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
    ret = subprocess.call(cmd)
    if ret != 0:
        print("[X] Lỗi trong quá trình cài đặt thư viện!")
        sys.exit(ret)
    print("[+] Cài đặt thư viện hoàn tất thành công!")
    return True


# ==============================================================================
# BƯỚC 3: KIỂM TRA MODEL CẦN TẢI & PROMPT XÁC NHẬN TẢI MODEL
# ==============================================================================
def step_check_and_download_models(models_dir: str, use_mirror: bool = False, yes_all: bool = False):
    print("\n" + "=" * 65)
    print("  BƯỚC 2: KIỂM TRA TRỌNG SỐ PRETRAINED MODELS")
    print("=" * 65)

    os.makedirs(models_dir, exist_ok=True)
    missing_models = []
    total_model_gb = 0.0

    for model_name, info in MODELS_INFO.items():
        model_path = Path(models_dir) / model_name
        is_complete = False
        if model_path.exists() and model_path.is_dir():
            # Kiểm tra các file then chốt
            files_exist = [ (model_path / f).exists() for f in info["check_files"] ]
            if all(files_exist):
                is_complete = True

        if is_complete:
            print(f"  [+] {model_name:25s}: ĐÃ CÓ tại {model_path}")
        else:
            print(f"  [-] {model_name:25s}: CHƯA CÓ ({info['size_gb']} GB) - {info['desc']}")
            missing_models.append((model_name, info))
            total_model_gb += info["size_gb"]

    if not missing_models:
        print("\n[+] Toàn bộ Pretrained Models ĐÃ ĐẦY ĐỦ trong thư mục:", models_dir)
        return True

    print("\n" + "-" * 65)
    print(f"[!] Cần tải {len(missing_models)} model về thư mục '{models_dir}':")
    for m_name, info in missing_models:
        print(f"    - {m_name}: ~{info['size_gb']} GB")
    print(f"[*] Tổng dung lượng model cần tải: ~{total_model_gb:.2f} GB")

    if not yes_all:
        confirmed = ask_user_confirm(f"[?] Bạn có muốn tải các model trên (~{total_model_gb:.2f} GB) không?")
        if not confirmed:
            print("[X] Người dùng hủy tải model. Dừng chương trình.")
            sys.exit(0)

    # Thực hiện tải
    from scripts.download_hf_model import download_model
    for m_name, info in missing_models:
        print(f"\n[*] Đang tải {m_name}...")
        download_model(
            repo_id=info["repo_id"],
            local_dir=models_dir,
            use_mirror=use_mirror,
            source="huggingface",
        )

    print("[+] Hoàn tất tải toàn bộ models!")
    return True


# ==============================================================================
# BƯỚC 4 & 5: LẤY DỮ LIỆU -> TIỀN XỬ LÝ -> NORM STATS -> HUẤN LUYỆN
# ==============================================================================
def step_process_data_and_train(raw_data_dir: str, models_dir: str, yes_all: bool = False):
    print("\n" + "=" * 65)
    print("  BƯỚC 3: KIỂM TRA DỮ LIỆU & TIỀN XỬ LÝ")
    print("=" * 65)

    raw_path = Path(raw_data_dir)
    print(f"[*] Thư mục dữ liệu chỉ định: {raw_path.resolve()}")

    # Tìm các file episode hợp lệ
    data_files = []
    if raw_path.exists():
        data_files = list(raw_path.glob("**/*.hdf5")) + list(raw_path.glob("**/*.h5")) + list(raw_path.glob("**/*.npz"))

    if not data_files:
        print(f"[!] CẢNH BÁO: Không tìm thấy file .hdf5 / .h5 / .npz nào trong: {raw_data_dir}")
        print("    Bạn có thể:")
        print("    1. Kiểm tra lại đường dẫn truyền vào tham số --data-dir")
        print("    2. Tải 45 episodes từ HuggingFace (Tuyen062004/data_openarm_8_10, ~16 GB)")
        download_hf = ask_user_confirm("[?] Bạn có muốn tải dataset từ HuggingFace về thư mục này không?")
        if download_hf:
            os.makedirs(raw_data_dir, exist_ok=True)
            try:
                from huggingface_hub import snapshot_download
                print("[*] Đang tải dataset từ HuggingFace...")
                snapshot_download(
                    repo_id="Tuyen062004/data_openarm_8_10",
                    repo_type="dataset",
                    local_dir=raw_data_dir,
                )
                data_files = list(Path(raw_data_dir).glob("**/*.hdf5"))
            except Exception as e:
                print(f"[X] Lỗi khi tải dataset: {e}")
                sys.exit(1)
        else:
            print("[X] Không có dữ liệu để huấn luyện. Dừng chương trình.")
            sys.exit(1)

    print(f"[+] Tìm thấy {len(data_files)} files episode hợp lệ.")

    # Tiền xử lý
    processed_dir = PROJECT_ROOT / "data" / "openarm_processed"
    train_txt_path = PROJECT_ROOT / "assets" / "training_data" / "openarm_train.txt"

    print("\n[*] Tiến hành tiền xử lý, lọc bỏ NaN và chuẩn hóa dữ liệu...")
    from tools.prepare_openarm_data import process_dataset
    process_dataset(str(raw_path), str(processed_dir), str(train_txt_path))

    # Môi trường chạy subprocess (luôn kèm PYTHONPATH trỏ về PROJECT_ROOT)
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + run_env["PYTHONPATH"] if "PYTHONPATH" in run_env else "")

    # Tính norm stats nếu chưa có
    norm_stats_path = PROJECT_ROOT / "assets" / "norm_stats" / "openarm_dual.json"
    if not norm_stats_path.exists():
        print("\n[*] Đang tính toán bảng thống kê chuẩn hóa góc khớp (Normalization Stats)...")
        norm_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compute_norm_stats.py"),
            "--config", str(PROJECT_ROOT / "configs" / "vla" / "norm_compute" / "post_data.yaml"),
            "--data.train_path", str(train_txt_path),
            "--data.robot_config_root", str(PROJECT_ROOT / "configs" / "robot_configs"),
            "--data.robot_name", "openarm_dual",
            "--data.norm_path", str(norm_stats_path),
        ]
        ret = subprocess.call(norm_cmd, env=run_env)
        if ret != 0:
            print("[X] Lỗi khi tính toán Normalization Stats!")
            sys.exit(ret)
    else:
        print(f"[+] Đã có bảng chuẩn hóa: {norm_stats_path}")

    # BƯỚC CUỐI CÙNG: KHỞI CHẠY HUẤN LUYỆN
    print("\n" + "=" * 65)
    print("  BƯỚC 4: KHỞI CHẠY HUẤN LUYỆN ACTION EXPERT (RTX 4090)")
    print("=" * 65)
    train_config = PROJECT_ROOT / "configs" / "vla" / "real_robot" / "openarm_expert_4090.yaml"
    print(f"[*] Cấu hình huấn luyện: {train_config}")
    print("    - VLM Backbone: FROZEN (tiết kiệm VRAM)")
    print("    - Action Expert: ACTIVE TRAINING (bfloat16 + FP32 heads)")
    print("    - VRAM ước tính: 16 - 18 GB")

    train_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tasks" / "vla" / "train_lingbotvla.py"),
        "--config", str(train_config),
    ]

    print("\n[*] Bắt đầu quá trình huấn luyện...")
    ret = subprocess.call(train_cmd, env=run_env)
    if ret == 0:
        print("\n[+] HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    else:
        print(f"\n[!] Quá trình huấn luyện kết thúc với mã lỗi: {ret}")
    sys.exit(ret)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Script All-in-One: Kiểm tra thư viện -> Tải model -> Xử lý dữ liệu -> Huấn luyện",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=str,
        default="data/openarm_raw",
        help="Đường dẫn đến thư mục chứa dữ liệu thô (.hdf5 / .h5 / .npz)",
    )
    parser.add_argument(
        "-m", "--model-dir",
        type=str,
        default="models",
        help="Thư mục lưu trữ hoặc chứa sẵn pretrained models",
    )
    parser.add_argument(
        "--use-mirror",
        action="store_true",
        help="Sử dụng mirror (hf-mirror.com) để tăng tốc độ tải mạng",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Tự động đồng ý (y) mọi prompt cài đặt/tải xuống mà không cần hỏi lại",
    )

    args = parser.parse_args()

    # 1. Kiểm tra & cài thư viện
    step_check_and_install_packages(yes_all=args.yes)

    # 2. Kiểm tra & tải models
    step_check_and_download_models(models_dir=args.model_dir, use_mirror=args.use_mirror, yes_all=args.yes)

    # 3. Xử lý data & train
    step_process_data_and_train(raw_data_dir=args.data_dir, models_dir=args.model_dir, yes_all=args.yes)


if __name__ == "__main__":
    main()

