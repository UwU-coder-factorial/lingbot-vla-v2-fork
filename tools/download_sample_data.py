"""
Tải 1-2 episode ngẫu nhiên từ HuggingFace Tuyen062004/data_openarm_8_10 về máy local để chạy test suite.
"""

import os
import sys
import random
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


REPO_ID = "Tuyen062004/data_openarm_8_10"
SAMPLE_CANDIDATES = [
    "episode_20261008_152119.hdf5",
    "episode_20261008_152230.hdf5",
    "episode_20261008_152334.hdf5",
    "episode_20261008_154211.hdf5",
]


def download_samples(num_samples=1, target_dir="data/openarm_samples"):
    os.makedirs(target_dir, exist_ok=True)
    selected = random.sample(SAMPLE_CANDIDATES, min(num_samples, len(SAMPLE_CANDIDATES)))

    try:
        from huggingface_hub import hf_hub_download
        print(f"[*] Đang tải {len(selected)} file mẫu từ {REPO_ID} về '{target_dir}'...")
        for filename in selected:
            dest = os.path.join(target_dir, filename)
            if os.path.exists(dest):
                print(f"[+] File đã tồn tại: {dest}")
                continue
            print(f"[*] Đang tải {filename}...")
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=target_dir,
            )
            print(f"[+] Đã tải: {dest}")
    except ImportError:
        # Dự phòng bằng urllib nếu chưa cài huggingface_hub
        import urllib.request
        print(f"[*] huggingface_hub chưa cài, dùng urllib tải trực tiếp...")
        for filename in selected:
            dest = os.path.join(target_dir, filename)
            if os.path.exists(dest):
                print(f"[+] File đã tồn tại: {dest}")
                continue
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{filename}"
            print(f"[*] Đang tải từ: {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"[+] Đã tải: {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải dữ liệu mẫu OpenArm để chạy test.")
    parser.add_argument("--num", type=int, default=1, help="Số lượng file cần tải (mặc định: 1)")
    parser.add_argument("--target_dir", type=str, default="data/openarm_samples", help="Thư mục lưu")
    args = parser.parse_args()
    download_samples(args.num, args.target_dir)

