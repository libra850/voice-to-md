"""文字起こし機能を提供するモジュール"""

from typing import Optional, Callable
import librosa
from faster_whisper import WhisperModel

from .config import WHISPER_MODEL_SIZE, LANGUAGE, WHISPER_DEVICE, SAMPLE_RATE


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
            # faster-whisperはCUDAまたはCPUのみ対応（MPSは未対応）
            compute_type = "float16" if WHISPER_DEVICE == "cuda" else "int8"
            self._model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=compute_type,
            )

    def transcribe(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[TranscriptionSegment]:
        """
        音声ファイルを文字起こしする。

        Args:
            audio_path: 音声ファイルのパス
            progress_callback: 進捗コールバック関数（0.0〜1.0）

        Returns:
            文字起こしセグメントのリスト
        """
        self._ensure_model_loaded()

        # 音声の長さを取得
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        total_duration = len(audio) / sr

        segments, info = self._model.transcribe(
            audio_path,
            language=LANGUAGE,
            beam_size=5,
            vad_filter=False,
            # vad_parameters=dict(
            #     min_silence_duration_ms=500,
            #     speech_pad_ms=300,
            # ),
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

            # 進捗を報告（セグメントの終了時間 / 総時間）
            if progress_callback and total_duration > 0:
                progress = min(segment.end / total_duration, 1.0)
                progress_callback(progress)

        # 完了を報告
        if progress_callback:
            progress_callback(1.0)

        return result
