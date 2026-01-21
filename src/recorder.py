"""録音機能を提供するモジュール"""

from typing import Optional
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from .config import SAMPLE_RATE, CHANNELS, TEMP_AUDIO_PATH


class Recorder:
    """マイクからの録音を管理するクラス"""

    def __init__(self) -> None:
        self.is_recording: bool = False
        self._audio_data: list[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddeviceのコールバック関数。音声データをリストに追加する。"""
        if status:
            print(f"録音ステータス: {status}")
        self._audio_data.append(indata.copy())

    def start(self) -> None:
        """録音を開始する。"""
        if self.is_recording:
            return

        self._audio_data = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=self._callback,
        )
        self._stream.start()
        self.is_recording = True

    def stop(self) -> str:
        """
        録音を停止し、wavファイルとして保存する。

        Returns:
            保存先のファイルパス
        """
        if not self.is_recording or self._stream is None:
            raise RuntimeError("録音が開始されていません")

        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.is_recording = False

        if not self._audio_data:
            raise RuntimeError("録音データがありません")

        audio_array = np.concatenate(self._audio_data, axis=0)
        audio_int16 = (audio_array * 32767).astype(np.int16)

        output_path = str(TEMP_AUDIO_PATH)
        wavfile.write(output_path, SAMPLE_RATE, audio_int16)

        return output_path

    def get_duration(self) -> float:
        """現在の録音時間を取得する（秒）"""
        if not self._audio_data:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self._audio_data)
        return total_samples / SAMPLE_RATE
