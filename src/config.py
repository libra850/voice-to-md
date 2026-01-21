"""Voice to MD アプリケーションの設定値"""

from pathlib import Path

# 録音設定
SAMPLE_RATE: int = 16000  # Whisperの要求仕様
CHANNELS: int = 1  # モノラル

# Whisperモデル設定
WHISPER_MODEL_SIZE: str = "small"  # 処理速度優先
LANGUAGE: str = "ja"

# ファイルパス設定
OUTPUT_DIR: Path = Path.home() / "Desktop"
TEMP_AUDIO_PATH: Path = Path("/tmp/voice_recording.wav")

# 話者分離設定
CHUNK_DURATION_SEC: int = 10  # 話者分離用のチャンク長（秒）

# 進捗ウィンドウ設定
WINDOW_WIDTH: int = 400
WINDOW_HEIGHT: int = 150
