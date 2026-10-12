#!/bin/bash
# ==============================================================================
# SCRIPT ALL-IN-ONE MASTER CHO LINUX/BASH
#
# Cách dùng:
#   bash scripts/train_all_in_one.sh --data-dir /path/to/data
#   bash scripts/train_all_in_one.sh --data-dir /path/to/data --use-mirror
# ==============================================================================

set -e

# Tìm binary python hợp lệ (python3 hoặc python)
PYTHON_BIN="python"
if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
fi

$PYTHON_BIN scripts/train_all_in_one.py "$@"

