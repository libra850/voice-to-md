#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

# Hugging Face Hubのテレメトリを無効化（使用統計の送信を停止）
export HF_HUB_DISABLE_TELEMETRY=1

python -m src.main
