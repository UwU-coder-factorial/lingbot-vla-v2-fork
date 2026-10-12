<#
.SYNOPSIS
    Script All-in-One Master cho Windows PowerShell.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/train_all_in_one.ps1 -data-dir "D:\path\to\data"
    powershell -ExecutionPolicy Bypass -File scripts/train_all_in_one.ps1 -data-dir "D:\path\to\data" -use-mirror
#>

$ErrorActionPreference = "Stop"

python scripts/train_all_in_one.py @args

