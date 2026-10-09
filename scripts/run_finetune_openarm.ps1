<#
.SYNOPSIS
    Script 1-Click Finetune Action Expert trên 1x RTX 4090 cho robot OpenArm (Windows PowerShell).
#>

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  LINGBOT-VLA 2.0: FINETUNE ACTION EXPERT (1x RTX 4090)   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Kiểm tra file cấu hình và dữ liệu
$RobotConfig = "configs/robot_configs/openarm_dual.yaml"
$TrainConfig = "configs/vla/real_robot/openarm_expert_4090.yaml"
$TrainTxt    = "assets/training_data/openarm_train.txt"
$NormStats   = "assets/norm_stats/openarm_dual.json"

if (-not (Test-Path $TrainTxt)) {
    Write-Host "[!] Không tìm thấy file danh sách dữ liệu: $TrainTxt" -ForegroundColor Yellow
    Write-Host "[*] Hãy chạy script chuẩn bị dữ liệu trước:" -ForegroundColor Yellow
    Write-Host "    python tools/prepare_openarm_data.py --input_dir <duong_dan_thu_muc_raw_data>" -ForegroundColor Green
    Exit 1
}

# 2. Tính toán Normalization Statistics nếu chưa có
if (-not (Test-Path $NormStats)) {
    Write-Host "[*] Đang tính toán thống kê chuẩn hóa góc khớp (Normalization Stats)..." -ForegroundColor Yellow
    python scripts/compute_norm_stats.py `
        --config configs/vla/norm_compute/post_data.yaml `
        --data.train_path $TrainTxt `
        --data.robot_config_root configs/robot_configs `
        --data.robot_name openarm `
        --data.norm_path $NormStats
    
    if (-not (Test-Path $NormStats)) {
        Write-Host "[X] Tính toán chuẩn hóa thất bại!" -ForegroundColor Red
        Exit 1
    }
    Write-Host "[+] Đã tạo file thống kê chuẩn hóa: $NormStats" -ForegroundColor Green
} else {
    Write-Host "[+] Đã tồn tại file chuẩn hóa: $NormStats (Bỏ qua bước tính lại)." -ForegroundColor Green
}

# 3. Chạy Huấn luyện Action Expert
Write-Host "[*] Khởi động quá trình huấn luyện Action Expert trên RTX 4090..." -ForegroundColor Cyan
Write-Host "    - VLM Backbone: FROZEN (0 gradient, 0 optimizer states)" -ForegroundColor Gray
Write-Host "    - Action Expert: ACTIVE TRAINING (bfloat16 + FP32 heads)" -ForegroundColor Gray
Write-Host "    - Batch size: micro=2, global=8 (gradient accumulation)" -ForegroundColor Gray

python tasks/vla/train_lingbotvla.py --config $TrainConfig

Write-Host "[+] Quá trình hoàn tất!" -ForegroundColor Green

