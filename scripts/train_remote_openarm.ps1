<#
.SYNOPSIS
    Script tự động từ A-Z cho máy train Windows: Tải data HF, tiền xử lý, tính norm stats và finetune Action Expert.
#>

$ErrorActionPreference = "Stop"

$RepoId = "Tuyen062004/data_openarm_8_10"
$RawDataDir = "data/openarm_raw"
$ProcessedDataDir = "data/openarm_processed"
$TrainTxt = "assets/training_data/openarm_train.txt"
$NormStats = "assets/norm_stats/openarm_dual.json"
$TrainConfig = "configs/vla/real_robot/openarm_expert_4090.yaml"
$ModelsDir = "models"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  0. KIỂM TRA & TẢI BASE PRETRAINED MODELS                " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
if (-not (Test-Path $ModelsDir)) { New-Item -ItemType Directory -Path $ModelsDir | Out-Null }

if (-not (Test-Path "$ModelsDir/lingbot-vla-v2-6b")) {
    Write-Host "[*] Đang tải lingbot-vla-v2-6b..." -ForegroundColor Yellow
    python scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir $ModelsDir
} else {
    Write-Host "[+] Đã tồn tại $ModelsDir/lingbot-vla-v2-6b" -ForegroundColor Green
}

if (-not (Test-Path "$ModelsDir/Qwen3-VL-4B-Instruct")) {
    Write-Host "[*] Đang tải Qwen3-VL-4B-Instruct..." -ForegroundColor Yellow
    python scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir $ModelsDir
} else {
    Write-Host "[+] Đã tồn tại $ModelsDir/Qwen3-VL-4B-Instruct" -ForegroundColor Green
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  1. TẢI DATASET TỪ HUGGINGFACE: $RepoId" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan


if (-not (Test-Path $RawDataDir)) { New-Item -ItemType Directory -Path $RawDataDir | Out-Null }

if (Get-Command huggingface-cli -ErrorAction SilentlyContinue) {
    Write-Host "[*] Sử dụng huggingface-cli download..." -ForegroundColor Yellow
    huggingface-cli download --repo-type dataset $RepoId --local-dir $RawDataDir
} else {
    Write-Host "[*] Sử dụng python snapshot_download..." -ForegroundColor Yellow
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$RepoId', repo_type='dataset', local_dir='$RawDataDir')"
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  2. TIỀN XỬ LÝ & ĐÓNG GÓI DỮ LIỆU CHUẨN HÓA" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
python tools/prepare_openarm_data.py `
    --input_dir $RawDataDir `
    --output_dir $ProcessedDataDir `
    --txt_path $TrainTxt

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  3. TÍNH TOÁN NORMALIZATION STATISTICS" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
if (-not (Test-Path $NormStats)) {
    python scripts/compute_norm_stats.py `
        --config configs/vla/norm_compute/post_data.yaml `
        --data.train_path $TrainTxt `
        --data.robot_config_root configs/robot_configs `
        --data.robot_name openarm `
        --data.norm_path $NormStats
} else {
    Write-Host "[+] Đã tồn tại $NormStats, bỏ qua bước tính lại." -ForegroundColor Green
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  4. KHỞI CHẠY HUẤN LUYỆN ACTION EXPERT (RTX 4090)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
python tasks/vla/train_lingbotvla.py --config $TrainConfig

