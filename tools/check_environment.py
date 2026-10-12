"""
Script kiểm tra môi trường & các thư viện cần thiết trước khi train.
Tránh cài lại các gói đã có sẵn.
"""

import sys
import importlib
import subprocess
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


REQUIRED_PACKAGES = {
    "torch": "torch",
    "numpy": "numpy",
    "h5py": "h5py",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
    "tqdm": "tqdm",
    "cv2": "opencv-python",
    "peft": "peft",
    "einops": "einops",
    "safetensors": "safetensors",
}

OPTIONAL_PACKAGES = {
    "flash_attn": "flash-attn",
}


def check_env(auto_install=False):
    print("=" * 60)
    print("  KIỂM TRA MÔI TRƯỜNG & THƯ VIỆN CHO LINGBOT-VLA")
    print("=" * 60)

    print(f"[*] Python: {sys.version.split()[0]} ({sys.executable})")

    missing_required = []
    installed_packages = []

    # 1. Kiểm tra các thư viện bắt buộc
    for mod_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [+] {pip_name:20s}: Đã cài ({ver})")
            installed_packages.append(pip_name)
        except ImportError:
            print(f"  [-] {pip_name:20s}: CHƯA CÓ")
            missing_required.append(pip_name)

    # 2. Kiểm tra PyTorch CUDA
    cuda_ready = False
    try:
        import torch
        if torch.cuda.is_available():
            cuda_ready = True
            print(f"\n[+] CUDA: Sẵn sàng | GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("\n[!] CẢNH BÁO: PyTorch chưa bật CUDA (đang chạy trên CPU)!")
    except ImportError:
        pass

    # 3. Kiểm tra các thư viện tối ưu (tùy chọn)
    print("\n[*] Thư viện tối ưu (Optional):")
    for mod_name, pip_name in OPTIONAL_PACKAGES.items():
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [+] {pip_name:20s}: Đã cài ({ver})")
        except ImportError:
            print(f"  [-] {pip_name:20s}: Chưa cài (Không bắt buộc, flash-attn giúp tăng tốc attention)")

    # 4. Đánh giá & Xử lý
    print("-" * 60)
    if not missing_required:
        print("[+] KẾT QUẢ: Môi trường đã ĐẦY ĐỦ 100% các thư viện cần thiết!")
        print("[+] Không cần cài đặt thêm gì.")
        return 0
    else:
        print(f"[!] KẾT QUẢ: Thiếu {len(missing_required)} thư viện: {', '.join(missing_required)}")
        install_cmd = f"pip install {' '.join(missing_required)}"
        print(f"\n[*] Lệnh cài đặt nhanh chỉ các gói còn thiếu:")
        print(f"    {install_cmd}")

        if auto_install:
            print("\n[*] Đang tự động cài các gói còn thiếu...")
            res = subprocess.call([sys.executable, "-m", "pip", "install"] + missing_required)
            return res
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra dependencies của LingBot-VLA")
    parser.add_argument("--install", action="store_true", help="Tự động cài đặt đúng các gói còn thiếu")
    args = parser.parse_args()

    exit_code = check_env(auto_install=args.install)
    sys.exit(exit_code)
