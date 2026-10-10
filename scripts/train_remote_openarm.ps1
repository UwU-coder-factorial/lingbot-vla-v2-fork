<#
.SYNOPSIS
    Script tự động từ A-Z cho máy train Windows:
    - Cho phép trỏ trực tiếp đến thư mục data có sẵn (-DataDir "C:\path\to\data")
    - Tự động bỏ qua tải nếu data đã có sẵn trên máy
    - Tự động bỏ qua tải 9GB weights thừa của Qwen3-VL (chỉ tải config ~30MB)
    - Hỗ trợ mirror tăng tốc độ mạng (-UseMirror)
#>

param(
    [Parameter(Position=0)]
    [string]$DataDir = "data/openarm_raw",

    [Parameter(Position=1)]
    [string]$ModelDir = "models",

    [switch]$UseMirror
)

$ErrorActionPreference = "Stop"

$RepoId = "Tuyen062004/data_openarm_8_10"
$RawDataDir = $DataDir
$ProcessedDataDir = "data/openarm_processed"
$TrainTxt = "assets/training_data/openarm_train.txt"
$NormStats = "assets/norm_stats/openarm_dual.json"
$TrainConfig = "configs/vla/real_robot/openarm_expert_4090.yaml"
$ModelsDir = $ModelDir

if ($UseMirror) {
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    Write-Host "[*] Kích hoạt HuggingFace Mirror: https://hf-mirror.com" -ForegroundColor Cyan
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  0. KIỂM TRA & TẢI BASE PRETRAINED MODELS                " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
if (-not (Test-Path $ModelsDir)) { New-Item -ItemType Directory -Path $ModelsDir | Out-Null }

$mirrorArg = if ($UseMirror) { "--use_mirror" } else { "" }

# Tải LingBot-VLA 2.0
if (-not (Test-Path "$ModelsDir/lingbot-vla-v2-6b")) {
    Write-Host "[*] Đang tải lingbot-vla-v2-6b..." -ForegroundColor Yellow
    python scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir $ModelsDir $mirrorArg
} else {
    Write-Host "[+] Đã tồn tại $ModelsDir/lingbot-vla-v2-6b (Bỏ qua bước tải)." -ForegroundColor Green
}

# Tải Qwen3-VL Tokenizer/Config (~30MB)
if (-not (Test-Path "$ModelsDir/Qwen3-VL-4B-Instruct")) {
    Write-Host "[*] Đang tải cấu hình Qwen3-VL-4B-Instruct..." -ForegroundColor Yellow
    python scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir $ModelsDir $mirrorArg
} else {
    Write-Host "[+] Đã tồn tại $ModelsDir/Qwen3-VL-4B-Instruct (Bỏ qua bước tải)." -ForegroundColor Green
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  1. KIỂM TRA DỮ LIỆU ĐẦU VÀO" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$existingFiles = 0
if (Test-Path $RawDataDir) {
    $files = Get-ChildItem -Path $RawDataDir -Include *.hdf5, *.h5, *.npz -Recurse -File -ErrorAction SilentlyContinue
    $existingFiles = $files.Count
}

if ($existingFiles -gt 0) {
    Write-Host "[+] Phát hiện $existingFiles files dữ liệu đã có sẵn tại: $RawDataDir" -ForegroundColor Green
    Write-Host "[+] BỎ QUA BƯỚC TẢI HUGGINGFACE." -ForegroundColor Green
} else {
    Write-Host "[*] Chưa có dữ liệu tại $RawDataDir. Bắt đầu tải từ HuggingFace: $RepoId..." -ForegroundColor Yellow
    if (-not (Test-Path $RawDataDir)) { New-Item -ItemType Directory -Path $RawDataDir | Out-Null }
    if (Get-Command huggingface-cli -ErrorAction SilentlyContinue) {
        huggingface-cli download --repo-type dataset $RepoId --local-dir $RawDataDir
    } else {
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$RepoId', repo_type='dataset', local_dir='$RawDataDir')"
    }
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
