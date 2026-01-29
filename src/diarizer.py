"""話者分離機能を提供するモジュール"""

from typing import Optional, Callable
import numpy as np
import torch
import librosa
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

from .config import SAMPLE_RATE, CHUNK_DURATION_SEC, DEVICE


class Diarizer:
    """SpeechBrain ECAPA-TDNNを使用した話者分離クラス"""

    def __init__(self) -> None:
        """話者埋め込みモデルを初期化する。初回実行時にモデルがダウンロードされる。"""
        self._classifier: Optional[EncoderClassifier] = None
        self._device = DEVICE

    def _ensure_model_loaded(self) -> None:
        """モデルがロードされていることを確認する"""
        if self._classifier is None:
            self._classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self._device},
            )

    def diarize(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[tuple[float, float, str]]:
        """
        音声ファイルを話者分離する。

        Args:
            audio_path: 音声ファイルのパス
            progress_callback: 進捗コールバック関数（0.0〜1.0）

        Returns:
            話者セグメントのリスト [(start_sec, end_sec, speaker_label), ...]
        """
        self._ensure_model_loaded()

        # 音声ファイルをロード
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / sr

        # チャンクに分割（オーバーラップあり）
        chunk_length = SAMPLE_RATE * CHUNK_DURATION_SEC
        hop_length = chunk_length // 2  # 50%オーバーラップ
        chunks = []
        chunk_times = []

        for i in range(0, len(audio) - chunk_length + 1, hop_length):
            chunk = audio[i:i + chunk_length]
            chunks.append(chunk)
            start_sec = i / SAMPLE_RATE
            end_sec = (i + chunk_length) / SAMPLE_RATE
            chunk_times.append((start_sec, end_sec))

        # 最後のチャンクを追加（短い場合もパディングして使用）
        if len(audio) > chunk_length:
            last_start = len(audio) - chunk_length
            if last_start > (len(chunks) - 1) * hop_length:
                chunk = audio[last_start:]
                if len(chunk) < chunk_length:
                    chunk = np.pad(chunk, (0, chunk_length - len(chunk)))
                chunks.append(chunk)
                chunk_times.append((last_start / SAMPLE_RATE, duration))

        if not chunks:
            # 音声が短すぎる場合
            if len(audio) >= SAMPLE_RATE:  # 最低1秒
                chunk = np.pad(audio, (0, chunk_length - len(audio)))
                chunks.append(chunk)
                chunk_times.append((0.0, duration))
            else:
                return [(0.0, duration, "Speaker 1")]

        # 各チャンクの話者埋め込みを抽出
        embeddings = []
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            chunk_tensor = torch.tensor(chunk, device=self._device).unsqueeze(0)
            with torch.no_grad():
                embedding = self._classifier.encode_batch(chunk_tensor)
            embeddings.append(embedding.squeeze().cpu().numpy())

            # 進捗を報告
            if progress_callback:
                progress_callback((idx + 1) / total_chunks)

        embeddings = np.array(embeddings)

        # コサイン類似度でクラスタリング
        if len(embeddings) < 2:
            labels = [0]
        else:
            try:
                # AgglomerativeClusteringを使用（話者分離に適している）
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0.5,  # コサイン距離の閾値
                    metric="cosine",
                    linkage="average",
                )
                labels = clustering.fit_predict(embeddings)
            except Exception:
                labels = [0] * len(embeddings)

        # 話者ラベルを割り当て（出現順に番号付け）
        label_order = []
        for label in labels:
            if label not in label_order:
                label_order.append(label)
        label_map = {label: f"Speaker {i + 1}" for i, label in enumerate(label_order)}

        # セグメントを作成（オーバーラップを考慮して中央の時間を使用）
        segments = []
        for i, ((start, end), label) in enumerate(zip(chunk_times, labels)):
            speaker = label_map[label]
            # オーバーラップ領域の処理：チャンクの中央部分を使用
            if i == 0:
                seg_start = start
            else:
                seg_start = start + (end - start) / 4

            if i == len(chunk_times) - 1:
                seg_end = end
            else:
                seg_end = end - (end - start) / 4

            segments.append((seg_start, seg_end, speaker))

        # 連続する同一話者のセグメントをマージ
        merged_segments = []
        current_speaker = None
        current_start = None
        current_end = None

        for start, end, speaker in segments:
            if speaker == current_speaker:
                current_end = end
            else:
                if current_speaker is not None:
                    merged_segments.append((current_start, current_end, current_speaker))
                current_speaker = speaker
                current_start = start
                current_end = end

        if current_speaker is not None:
            merged_segments.append((current_start, current_end, current_speaker))

        # 最終的なセグメントの時間を調整（ギャップを埋める）
        final_segments = []
        for i, (start, end, speaker) in enumerate(merged_segments):
            if i == 0:
                adj_start = 0.0
            else:
                adj_start = merged_segments[i - 1][1]

            if i == len(merged_segments) - 1:
                adj_end = duration
            else:
                adj_end = end

            final_segments.append((adj_start, adj_end, speaker))

        return final_segments

    def get_speaker_count(self, segments: list[tuple[float, float, str]]) -> int:
        """話者数を取得する"""
        speakers = set(seg[2] for seg in segments)
        return len(speakers)
