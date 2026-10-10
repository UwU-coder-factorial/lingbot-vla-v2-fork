#!/bin/bash
# ==============================================================================
# SCRIPT TỰ ĐỘNG TỪ A-Z CHO MÁY TRAIN (ALL-IN-ONE PIPELINE)
#
# Cách dùng:
# 1. Nếu data ĐÃ CÓ SẴN trên máy:
#    bash scripts/train_remote_openarm.sh --data-dir /duong_dan/toi/data_san_co
#    hoặc:
#    bash scripts/train_remote_openarm.sh /duong_dan/toi/data_san_co
#
# 2. Nếu muốn tự tải data từ HuggingFace qua Mirror tốc độ cao:
#    bash scripts/train_remote_openarm.sh --use-mirror
# ==============================================================================

set -e

REPO_ID="Tuyen062004/data_openarm_8_10"
RAW_DATA_DIR="data/openarm_raw"
PROCESSED_DATA_DIR="data/openarm_processed"
TRAIN_TXT="assets/training_data/openarm_train.txt"
NORM_STATS="assets/norm_stats/openarm_dual.json"
TRAIN_CONFIG="configs/vla/real_robot/openarm_expert_4090.yaml"
MODELS_DIR="models"
USE_MIRROR=false

# Phân tích tham số dòng lệnh
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            RAW_DATA_DIR="$2"
            shift 2
            ;;
        -m|--model-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --use-mirror)
            USE_MIRROR=true
            shift
            ;;
        *)
            # Nếu truyền đường dẫn trực tiếp: bash script.sh /path/to/data
            if [ -d "$1" ]; then
                RAW_DATA_DIR="$1"
            fi
            shift
            ;;
    esac
done

if [ "$USE_MIRROR" = true ]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "[*] Kích hoạt HuggingFace Mirror: https://hf-mirror.com"
fi

echo "=========================================================="
echo "  0. KIỂM TRA & TẢI BASE PRETRAINED MODELS"
echo "=========================================================="
mkdir -p "$MODELS_DIR"
MIRROR_FLAG=""
if [ "$USE_MIRROR" = true ]; then
    MIRROR_FLAG="--use_mirror"
fi

# Tải LingBot-VLA 2.0 (tự động bỏ qua teacher thừa để giảm dung lượng)
if [ ! -d "$MODELS_DIR/lingbot-vla-v2-6b" ]; then
    echo "[*] Đang tải lingbot-vla-v2-6b..."
    python scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir "$MODELS_DIR" $MIRROR_FLAG
else
    echo "[+] Đã tồn tại $MODELS_DIR/lingbot-vla-v2-6b (Bỏ qua bước tải)."
fi

# Qwen3-VL: chỉ tải config/tokenizer (~30MB), KHÔNG tải 9GB weights thừa
if [ ! -d "$MODELS_DIR/Qwen3-VL-4B-Instruct" ]; then
    echo "[*] Đang tải cấu hình Qwen3-VL-4B-Instruct..."
    python scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir "$MODELS_DIR" $MIRROR_FLAG
else
    echo "[+] Đã tồn tại $MODELS_DIR/Qwen3-VL-4B-Instruct (Bỏ qua bước tải)."
fi

echo "=========================================================="
echo "  1. KIỂM TRA DỮ LIỆU ĐẦU VÀO"
echo "=========================================================="
# Kiểm tra xem thư mục RAW_DATA_DIR đã có file hdf5/h5/npz chưa
EXISTING_FILES=$(find "$RAW_DATA_DIR" -type f \( -name "*.hdf5" -o -name "*.h5" -o -name "*.npz" \) 2>/dev/null | wc -l || echo "0")

if [ "$EXISTING_FILES" -gt 0 ]; then
    echo "[+] Phát hiện $EXISTING_FILES files dữ liệu đã có sẵn tại: $RAW_DATA_DIR"
    echo "[+] BỎ QUA BƯỚC TẢI HUGGINGFACE."
else
    echo "[*] Chưa có dữ liệu tại $RAW_DATA_DIR. Bắt đầu tải từ HuggingFace: $REPO_ID..."
    mkdir -p "$RAW_DATA_DIR"
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download --repo-type dataset "$REPO_ID" --local-dir "$RAW_DATA_DIR"
    else
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$REPO_ID', repo_type='dataset', local_dir='$RAW_DATA_DIR')"
    fi
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
