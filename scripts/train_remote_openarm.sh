#!/bin/bash
# ==============================================================================
# SCRIPT TỰ ĐỘNG TỪ A-Z CHO MÁY TRAIN (ALL-IN-ONE PIPELINE)
# 1. Tự động tải dataset từ HuggingFace (hỗ trợ resume khi rớt mạng)
# 2. Tiền xử lý, lọc NaN, đóng gói HDF5 chuẩn
# 3. Tính toán Normalization Statistics
# 4. Khởi chạy Finetune Action Expert (Tối ưu 1x RTX 4090 - VRAM 16-18GB)
# ==============================================================================

set -e

REPO_ID="Tuyen062004/data_openarm_8_10"
RAW_DATA_DIR="data/openarm_raw"
PROCESSED_DATA_DIR="data/openarm_processed"
TRAIN_TXT="assets/training_data/openarm_train.txt"
NORM_STATS="assets/norm_stats/openarm_dual.json"
TRAIN_CONFIG="configs/vla/real_robot/openarm_expert_4090.yaml"
MODELS_DIR="models"

echo "=========================================================="
echo "  0. KIỂM TRA & TẢI BASE PRETRAINED MODELS"
echo "=========================================================="
mkdir -p "$MODELS_DIR"

if [ ! -d "$MODELS_DIR/lingbot-vla-v2-6b" ]; then
    echo "[*] Đang tải lingbot-vla-v2-6b..."
    python scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir "$MODELS_DIR"
else
    echo "[+] Đã tồn tại $MODELS_DIR/lingbot-vla-v2-6b"
fi

if [ ! -d "$MODELS_DIR/Qwen3-VL-4B-Instruct" ]; then
    echo "[*] Đang tải Qwen3-VL-4B-Instruct..."
    python scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir "$MODELS_DIR"
else
    echo "[+] Đã tồn tại $MODELS_DIR/Qwen3-VL-4B-Instruct"
fi

echo "=========================================================="
echo "  1. TẢI DATASET TỪ HUGGINGFACE: $REPO_ID"
echo "=========================================================="
mkdir -p "$RAW_DATA_DIR"


if command -v huggingface-cli &> /dev/null; then
    echo "[*] Sử dụng huggingface-cli download..."
    huggingface-cli download --repo-type dataset "$REPO_ID" --local-dir "$RAW_DATA_DIR"
else
    echo "[*] huggingface-cli không tìm thấy, sử dụng python snapshot_download..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$REPO_ID', repo_type='dataset', local_dir='$RAW_DATA_DIR')"
fi

echo "=========================================================="
echo "  2. TIỀN XỬ LÝ & ĐÓNG GÓI DỮ LIỆU CHUẨN HÓA"
echo "=========================================================="
python tools/prepare_openarm_data.py \
    --input_dir "$RAW_DATA_DIR" \
    --output_dir "$PROCESSED_DATA_DIR" \
    --txt_path "$TRAIN_TXT"

echo "=========================================================="
echo "  3. TÍNH TOÁN NORMALIZATION STATISTICS"
echo "=========================================================="
if [ ! -f "$NORM_STATS" ]; then
    python scripts/compute_norm_stats.py \
        --config configs/vla/norm_compute/post_data.yaml \
        --data.train_path "$TRAIN_TXT" \
        --data.robot_config_root configs/robot_configs \
        --data.robot_name openarm \
        --data.norm_path "$NORM_STATS"
else
    echo "[+] Đã tồn tại $NORM_STATS, bỏ qua bước tính lại."
fi

echo "=========================================================="
echo "  4. KHỞI CHẠY HUẤN LUYỆN ACTION EXPERT (RTX 4090)"
echo "=========================================================="
python tasks/vla/train_lingbotvla.py --config "$TRAIN_CONFIG"

