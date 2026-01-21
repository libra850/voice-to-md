"""文字起こし機能を提供するモジュール"""

from typing import Optional
from faster_whisper import WhisperModel

from .config import WHISPER_MODEL_SIZE, LANGUAGE


class TranscriptionSegment:
    """文字起こしセグメント"""

    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class Transcriber:
    """faster-whisperを使用した文字起こしクラス"""

    def __init__(self) -> None:
        """Whisperモデルを初期化する。初回実行時にモデルがダウンロードされる。"""
        self._model: Optional[WhisperModel] = None

    def _ensure_model_loaded(self) -> None:
        """モデルがロードされていることを確認する"""
        if self._model is None:
            self._model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )

    def transcribe(self, audio_path: str) -> list[TranscriptionSegment]:
        """
        音声ファイルを文字起こしする。

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            文字起こしセグメントのリスト
        """
        self._ensure_model_loaded()

        segments, info = self._model.transcribe(
            audio_path,
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=300,
            ),
        )

        result = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                result.append(TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=text,
                ))

        return result

    def get_duration(self, audio_path: str) -> float:
        """音声ファイルの長さを取得する（秒）"""
        self._ensure_model_loaded()

        segments, info = self._model.transcribe(
            audio_path,
            language=LANGUAGE,
            beam_size=1,
        )
        # セグメントを消費して情報を取得
        last_end = 0.0
        for segment in segments:
            last_end = segment.end

        return last_end
