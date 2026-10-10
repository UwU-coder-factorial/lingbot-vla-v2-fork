import argparse
import os
import sys

"""
Script tải model tối ưu dung lượng và tốc độ.
- Hỗ trợ tải qua mirror (hf-mirror.com / modelscope) tăng tốc độ mạng châu Á.
- Tự động bỏ qua các weights thừa không dùng đến (như depth/dino video teacher hoặc weights của Qwen3-VL khi chỉ cần tokenizer).
"""

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def download_model(repo_id, local_dir, use_mirror=False, source="huggingface"):
    model_name = repo_id.split("/")[-1]
    target_path = os.path.join(local_dir, model_name)
    os.makedirs(local_dir, exist_ok=True)

    # 1. Tối ưu: Qwen3-VL chỉ cần tokenizer/config, KHÔNG cần 9GB weights
    ignore_patterns = []
    allow_patterns = None
    if "Qwen3-VL" in repo_id or "Qwen2-VL" in repo_id:
        print(f"[*] {repo_id}: Chỉ tải Tokenizer & Config (~30MB), bỏ qua 9GB weights thừa...")
        ignore_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]

    # 2. Tối ưu: LingBot-VLA khi finetune 4090 không dùng depth/video teacher
    if "lingbot-vla" in repo_id.lower():
        print(f"[*] {repo_id}: Bỏ qua depth teacher & video teacher để tiết kiệm ~4GB...")
        ignore_patterns = ["depth/*", "dino_video/*", "assets/*"]

    # 3. Thử tải qua ModelScope nếu được chỉ định hoặc có thư viện
    if source == "modelscope":
        try:
            from modelscope import snapshot_download as ms_download
            print(f"[*] Đang tải {repo_id} qua ModelScope CDN tốc độ cao...")
            ms_id = repo_id
            if repo_id == "robbyant/lingbot-vla-v2-6b":
                ms_id = "Robbyant/lingbot-vla-v2-6b"
            ms_download(
                model_id=ms_id,
                local_dir=target_path,
                ignore_file_pattern=ignore_patterns if ignore_patterns else None,
            )
            print(f"[+] Hoàn tất tải về: {target_path}")
            return
        except Exception as e:
            print(f"[!] ModelScope không khả dụng ({e}), chuyển sang HuggingFace...")

    # 4. Tải qua HuggingFace (hỗ trợ mirror)
    if use_mirror and "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[*] Đã kích hoạt HuggingFace Mirror (https://hf-mirror.com) để tăng tốc độ tải.")

    from huggingface_hub import snapshot_download
    print(f"[*] Đang tải {repo_id} về '{target_path}'...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_path,
        ignore_patterns=ignore_patterns if ignore_patterns else None,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
        max_workers=8,
    )
    print(f"[+] Hoàn tất tải về: {target_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải model tối ưu dung lượng và tốc độ")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repo ID")
    parser.add_argument("--local_dir", type=str, default="models", help="Thư mục lưu")
    parser.add_argument("--use_mirror", action="store_true", help="Dùng hf-mirror.com")
    parser.add_argument("--source", type=str, default="huggingface", choices=["huggingface", "modelscope"])
    args = parser.parse_args()

    download_model(args.repo_id, args.local_dir, use_mirror=args.use_mirror, source=args.source)
