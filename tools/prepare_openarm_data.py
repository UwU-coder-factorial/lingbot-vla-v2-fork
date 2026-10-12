"""
Script tiền xử lý dữ liệu OpenArm (Dual-arm 16 DoF + Camera Chest)
Hỗ trợ chuyển đổi từ nhiều nguồn: HDF5 (.h5, .hdf5), thư mục mảng Numpy (.npy, .npz), LeRobot.
Lọc bỏ mẫu lỗi / NaN và chuẩn hóa về cấu trúc dataset cho LingBot-VLA.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


try:
    import cv2
except ImportError:
    cv2 = None


def inspect_and_extract_hdf5(file_path):
    """Đọc dữ liệu từ file HDF5."""
    with h5py.File(file_path, "r") as f:
        # Hỗ trợ cả hai dạng: lồng observations/... hoặc phẳng
        chest_rgb = None
        chest_depth = None
        qpos = None
        action = None

        def search_key(keys):
            for k in keys:
                if k in f:
                    return np.array(f[k])
            return None

        # RGB
        for k in ["observations/images/chest_rgb", "observations/chest_rgb", "chest_rgb", "images/chest_rgb"]:
            if k in f:
                chest_rgb = np.array(f[k])
                break

        # Depth (tùy chọn)
        for k in ["observations/images/chest_depth", "observations/chest_depth", "chest_depth", "images/chest_depth"]:
            if k in f:
                chest_depth = np.array(f[k])
                break

        # Qpos / State
        for k in ["observations/qpos", "observations/state", "qpos", "state"]:
            if k in f:
                qpos = np.array(f[k])
                break

        # Action
        for k in ["action", "actions", "observations/qpos"]:
            if k in f:
                action = np.array(f[k])
                break

        return chest_rgb, chest_depth, qpos, action


def inspect_and_extract_npz(file_path):
    """Đọc dữ liệu từ file NPZ / thư mục NPY."""
    data = np.load(file_path, allow_pickle=True)
    chest_rgb = data.get("observations/images/chest_rgb", data.get("chest_rgb", None))
    chest_depth = data.get("observations/images/chest_depth", data.get("chest_depth", None))
    qpos = data.get("observations/qpos", data.get("qpos", None))
    action = data.get("action", data.get("actions", qpos))
    return chest_rgb, chest_depth, qpos, action


def process_dataset(input_dir, output_dir, txt_list_path, task_name="openarm_manipulation", target_img_size=None):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(txt_list_path), exist_ok=True)

    input_path = Path(input_dir)
    h5_files = sorted(list(input_path.glob("**/*.h5")) + list(input_path.glob("**/*.hdf5")))
    npz_files = sorted(list(input_path.glob("**/*.npz")))

    raw_files = []
    file_type = "h5"
    if h5_files:
        raw_files = h5_files
        file_type = "h5"
    elif npz_files:
        raw_files = npz_files
        file_type = "npz"
    else:
        print(f"[!] Không tìm thấy file .h5 hoặc .npz trong {input_dir}. Sẽ quét thư mục con.")
        return

    print(f"[*] Tìm thấy {len(raw_files)} episodes ({file_type}). Bắt đầu chuẩn hóa...")

    valid_episodes = []

    for ep_idx, fpath in enumerate(tqdm(raw_files, desc="Processing episodes")):
        try:
            if file_type == "h5":
                rgb, depth, qpos, action = inspect_and_extract_hdf5(fpath)
            else:
                rgb, depth, qpos, action = inspect_and_extract_npz(fpath)

            if rgb is None or qpos is None:
                print(f"[Warn] Bỏ qua {fpath.name}: thiếu RGB hoặc Qpos.")
                continue

            if action is None:
                action = qpos.copy()

            # Kiểm tra NaN
            if np.isnan(qpos).any() or np.isnan(action).any():
                print(f"[Warn] Bỏ qua {fpath.name}: chứa giá trị NaN.")
                continue

            # Kiểm tra khớp: 16 DoF
            if qpos.shape[-1] != 16 or action.shape[-1] != 16:
                print(f"[Warn] Bỏ qua {fpath.name}: số DoF={qpos.shape[-1]} != 16.")
                continue

            # Resize ảnh RGB nếu yêu cầu
            if target_img_size is not None and cv2 is not None:
                h, w = target_img_size
                resized_rgb = []
                for img in rgb:
                    resized_rgb.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
                rgb = np.array(resized_rgb, dtype=np.uint8)

            # Lưu ra file HDF5 chuẩn hóa
            out_file = Path(output_dir) / f"episode_{len(valid_episodes):06d}.h5"
            with h5py.File(out_file, "w") as out_f:
                out_f.create_dataset("observations/images/chest_rgb", data=rgb, compression="gzip")
                if depth is not None:
                    out_f.create_dataset("observations/images/chest_depth", data=depth, compression="gzip")
                out_f.create_dataset("observation.qpos", data=qpos.astype(np.float32))
                out_f.create_dataset("observation.state", data=qpos.astype(np.float32))
                out_f.create_dataset("action", data=action.astype(np.float32))

            valid_episodes.append(str(out_file.resolve()))
        except Exception as e:
            print(f"[Error] Lỗi khi xử lý {fpath.name}: {e}")

    # Ghi file danh sách training (.txt)
    with open(txt_list_path, "w", encoding="utf-8") as f:
        for ep_path in valid_episodes:
            f.write(f"openarm {ep_path}\n")

    print(f"[+] Hoàn tất! Đã lưu {len(valid_episodes)} episodes chuẩn hóa tại: {output_dir}")
    print(f"[+] File danh sách training: {txt_list_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiền xử lý và chuẩn hóa dataset OpenArm")
    parser.add_argument("--input_dir", type=str, required=True, help="Thư mục chứa raw dataset (HDF5 / NPZ)")
    parser.add_argument("--output_dir", type=str, default="data/openarm_processed", help="Thư mục xuất HDF5 chuẩn")
    parser.add_argument("--txt_path", type=str, default="assets/training_data/openarm_train.txt", help="File txt danh sách")
    parser.add_argument("--resize_h", type=int, default=None, help="Resize chiều cao ảnh (tùy chọn)")
    parser.add_argument("--resize_w", type=int, default=None, help="Resize chiều rộng ảnh (tùy chọn)")
    args = parser.parse_args()

    size = (args.resize_h, args.resize_w) if args.resize_h and args.resize_w else None
    process_dataset(args.input_dir, args.output_dir, args.txt_path, target_img_size=size)

