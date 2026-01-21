"""話者分離機能を提供するモジュール"""

from typing import Optional
import numpy as np
import torch
import librosa
from sklearn.cluster import MeanShift
from speechbrain.inference.speaker import EncoderClassifier

from .config import SAMPLE_RATE, CHUNK_DURATION_SEC


class Diarizer:
    """SpeechBrain ECAPA-TDNNを使用した話者分離クラス"""

    def __init__(self) -> None:
        """話者埋め込みモデルを初期化する。初回実行時にモデルがダウンロードされる。"""
        self._classifier: Optional[EncoderClassifier] = None

    def _ensure_model_loaded(self) -> None:
        """モデルがロードされていることを確認する"""
        if self._classifier is None:
            self._classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )

    def diarize(self, audio_path: str) -> list[tuple[float, float, str]]:
        """
        音声ファイルを話者分離する。

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            話者セグメントのリスト [(start_sec, end_sec, speaker_label), ...]
        """
        self._ensure_model_loaded()

        # 音声ファイルをロード
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr

        # チャンクに分割
        chunk_length = SAMPLE_RATE * CHUNK_DURATION_SEC
        chunks = []
        chunk_times = []

        for i in range(0, len(audio), chunk_length):
            chunk = audio[i:i + chunk_length]
            # 短すぎるチャンクはスキップ
            if len(chunk) < SAMPLE_RATE * 2:  # 最低2秒
                continue
            # パディング
            if len(chunk) < chunk_length:
                chunk = np.pad(chunk, (0, chunk_length - len(chunk)))
            chunks.append(chunk)
            start_sec = i / SAMPLE_RATE
            end_sec = min((i + chunk_length) / SAMPLE_RATE, duration)
            chunk_times.append((start_sec, end_sec))

        if not chunks:
            # 音声が短すぎる場合は単一話者として扱う
            return [(0.0, duration, "Speaker 1")]

        # 各チャンクの話者埋め込みを抽出
        embeddings = []
        for chunk in chunks:
            chunk_tensor = torch.tensor(chunk).unsqueeze(0)
            with torch.no_grad():
                embedding = self._classifier.encode_batch(chunk_tensor)
            embeddings.append(embedding.squeeze().numpy())

        embeddings = np.array(embeddings)

        # クラスタリングで話者を分類
        if len(embeddings) < 2:
            # チャンクが1つしかない場合
            labels = [0]
        else:
            try:
                clustering = MeanShift(bandwidth=None)
                labels = clustering.fit_predict(embeddings)
            except Exception:
                # クラスタリングに失敗した場合は単一話者として扱う
                labels = [0] * len(embeddings)

        # 話者ラベルを割り当て
        unique_labels = sorted(set(labels))
        label_map = {label: f"Speaker {i + 1}" for i, label in enumerate(unique_labels)}

        # 連続する同一話者のチャンクをマージ
        segments = []
        current_speaker = None
        current_start = None
        current_end = None

        for (start, end), label in zip(chunk_times, labels):
            speaker = label_map[label]
            if speaker == current_speaker:
                current_end = end
            else:
                if current_speaker is not None:
                    segments.append((current_start, current_end, current_speaker))
                current_speaker = speaker
                current_start = start
                current_end = end

        # 最後のセグメントを追加
        if current_speaker is not None:
            segments.append((current_start, current_end, current_speaker))

        return segments

    def get_speaker_count(self, segments: list[tuple[float, float, str]]) -> int:
        """話者数を取得する"""
        speakers = set(seg[2] for seg in segments)
        return len(speakers)
