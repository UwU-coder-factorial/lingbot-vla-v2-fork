#!/bin/bash
# Script 1-Click Finetune Action Expert trên 1x RTX 4090 cho robot OpenArm (Linux/Bash).

set -e

echo "=========================================================="
echo "  LINGBOT-VLA 2.0: FINETUNE ACTION EXPERT (1x RTX 4090)   "
echo "=========================================================="

ROBOT_CONFIG="configs/robot_configs/openarm_dual.yaml"
TRAIN_CONFIG="configs/vla/real_robot/openarm_expert_4090.yaml"
TRAIN_TXT="assets/training_data/openarm_train.txt"
NORM_STATS="assets/norm_stats/openarm_dual.json"

if [ ! -f "$TRAIN_TXT" ]; then
    echo "[!] Không tìm thấy file danh sách dữ liệu: $TRAIN_TXT"
    echo "[*] Chạy tiền xử lý dữ liệu trước: python tools/prepare_openarm_data.py --input_dir <path_to_raw_data>"
    exit 1
fi

if [ ! -f "$NORM_STATS" ]; then
    echo "[*] Đang tính toán thống kê chuẩn hóa góc khớp..."
    python scripts/compute_norm_stats.py \
        --config configs/vla/norm_compute/post_data.yaml \
        --data.train_path "$TRAIN_TXT" \
        --data.robot_config_root configs/robot_configs \
        --data.robot_name openarm \
        --data.norm_path "$NORM_STATS"
    echo "[+] Đã tạo file chuẩn hóa: $NORM_STATS"
else
    echo "[+] Đã tồn tại file chuẩn hóa: $NORM_STATS"
fi

echo "[*] Khởi động quá trình huấn luyện Action Expert..."
python tasks/vla/train_lingbotvla.py --config "$TRAIN_CONFIG"

