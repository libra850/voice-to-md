"""Voice to MD アプリケーションの設定値"""

from pathlib import Path
import torch

# デバイス設定（Apple Silicon GPU対応）
def get_device() -> str:
    """利用可能な最速のデバイスを取得する"""
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    return "cpu"

def get_whisper_device() -> str:
    """Whisper用のデバイスを取得する（faster-whisperはcuda/cpuのみ）"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE: str = get_device()
WHISPER_DEVICE: str = get_whisper_device()

# 録音設定
SAMPLE_RATE: int = 16000  # Whisperの要求仕様
CHANNELS: int = 1  # モノラル

# Whisperモデル設定
WHISPER_MODEL_SIZE: str = "medium" 
LANGUAGE: str = "ja"

# ファイルパス設定
OUTPUT_DIR: Path = Path.home() / "Desktop"
TEMP_AUDIO_PATH: Path = Path("/tmp/voice_recording.wav")

# 話者分離設定
CHUNK_DURATION_SEC: int = 3  # 話者分離用のチャンク長（秒）

# 進捗ウィンドウ設定
WINDOW_WIDTH: int = 400
WINDOW_HEIGHT: int = 150
