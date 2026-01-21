#!/bin/bash
set -e

echo "=== Voice to MD セットアップ ==="

# Python 3.11確認
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11が必要です。以下でインストールしてください:"
    echo "  brew install python@3.11"
    exit 1
fi

# PyObjCはpipでインストールされるため、事前チェック不要

# venv作成
python3.11 -m venv .venv
source .venv/bin/activate

# 依存インストール
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== セットアップ完了 ==="
echo "初回起動時にモデルがダウンロードされます（数分かかります）"
echo "起動方法: ./run.sh"
