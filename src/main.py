"""Voice to MD アプリケーションのエントリーポイント"""

import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import rumps
import librosa

from .config import OUTPUT_DIR, TEMP_AUDIO_PATH, SAMPLE_RATE
from .recorder import Recorder
from .transcriber import Transcriber, TranscriptionSegment
from .diarizer import Diarizer
from .progress_window import ProgressWindow


class VoiceToMdApp(rumps.App):
    """メニューバーに常駐する音声文字起こしアプリケーション"""

    def __init__(self) -> None:
        super().__init__(name="Voice to MD", title="🎤 Voice")
        self._recorder = Recorder()
        self._transcriber = Transcriber()
        self._diarizer = Diarizer()
        self._progress_window = ProgressWindow()
        self._is_recording = False
        self._is_processing = False

        rumps.notification(
            title="Voice to MD",
            subtitle="起動しました",
            message="メニューバーの🎤 Voiceをクリックして録音を開始",
        )

    @rumps.clicked("録音 開始/停止")
    def toggle_recording(self, _: rumps.MenuItem) -> None:
        """録音の開始/停止を切り替える"""
        if self._is_processing:
            return

        if not self._is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        """録音を開始する"""
        try:
            self._recorder.start()
            self._is_recording = True
            self.title = "🔴 REC"
        except Exception as e:
            rumps.alert(
                title="マイクエラー",
                message=(
                    f"録音を開始できませんでした。\n\n"
                    f"エラー: {e}\n\n"
                    f"システム環境設定 > セキュリティとプライバシー > プライバシー > "
                    f"マイク でアプリケーションの許可を確認してください。"
                ),
            )

    def _stop_recording(self) -> None:
        """録音を停止し、処理を開始する"""
        if self._is_processing:
            return

        self._is_processing = True
        self._is_recording = False
        self.title = "⏳ ..."

        try:
            audio_path = self._recorder.stop()
            threading.Thread(
                target=self._process_audio,
                args=(audio_path,),
                daemon=True,
            ).start()
        except Exception as e:
            self._is_processing = False
            self.title = "🎤 Voice"
            rumps.alert(
                title="録音エラー",
                message=f"録音の停止中にエラーが発生しました。\n\n{e}",
            )

    def _process_audio(self, audio_path: str) -> None:
        """音声ファイルを処理する（別スレッド）"""
        try:
            self._progress_window.show()

            # 録音時間を取得
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            duration_sec = len(audio) / sr
            duration_str = self._format_duration(duration_sec)

            # 話者分離
            self._progress_window.set_status("話者を分析中...")
            try:
                speaker_segments = self._diarizer.diarize(audio_path)
                speaker_count = self._diarizer.get_speaker_count(speaker_segments)
            except Exception as e:
                print(f"話者分離エラー: {e}")
                speaker_segments = [(0.0, duration_sec, "Speaker 1")]
                speaker_count = 1

            # 文字起こし
            self._progress_window.set_status("文字起こし中...")
            try:
                transcription_segments = self._transcriber.transcribe(audio_path)
            except Exception as e:
                print(f"文字起こしエラー: {e}")
                transcription_segments = []

            # 結果を統合
            self._progress_window.set_status("結果を統合中...")
            merged_segments = self._assign_speakers(
                transcription_segments, speaker_segments
            )

            # Markdownファイルを生成
            self._progress_window.set_status("ファイルを保存中...")
            now = datetime.now()
            filename = f"voice_{now.strftime('%Y%m%d_%H%M%S')}.md"
            output_path = OUTPUT_DIR / filename

            content = self._create_markdown_content(
                now, duration_str, speaker_count, merged_segments
            )
            self._save_markdown(output_path, content)

            self._progress_window.hide()

            rumps.notification(
                title="Voice to MD",
                subtitle="文字起こし完了",
                message=f"保存先: {filename}",
            )

            self._cleanup_temp_file()

        except Exception as e:
            self._progress_window.hide()
            print(f"処理エラー: {e}")
            rumps.notification(
                title="処理エラー",
                subtitle="処理中にエラーが発生しました",
                message=str(e),
            )

        self.title = "🎤 Voice"
        self._is_processing = False

    def _assign_speakers(
        self,
        transcription_segments: list[TranscriptionSegment],
        speaker_segments: list[tuple[float, float, str]],
    ) -> list[dict]:
        """文字起こしセグメントに話者を割り当てる"""
        result = []
        for trans_seg in transcription_segments:
            trans_start = trans_seg.start
            speaker = "Unknown"
            for spk_start, spk_end, spk_label in speaker_segments:
                if spk_start <= trans_start < spk_end:
                    speaker = spk_label
                    break
            result.append({
                "speaker": speaker,
                "start": trans_seg.start,
                "end": trans_seg.end,
                "text": trans_seg.text,
            })
        return result

    def _format_duration(self, seconds: float) -> str:
        """秒数を「XX分XX秒」形式にフォーマットする"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs:02d}秒"

    def _format_timestamp(self, seconds: float) -> str:
        """秒数を「MM:SS」形式にフォーマットする"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _create_markdown_content(
        self,
        timestamp: datetime,
        duration: str,
        speaker_count: int,
        segments: list[dict],
    ) -> str:
        """Markdownファイルの内容を生成する"""
        lines = [
            f"# MTGメモ - {timestamp.strftime('%Y/%m/%d %H:%M')}",
            "",
            f"録音時間: {duration}",
            f"検出された話者数: {speaker_count}",
            "",
            "---",
            "",
        ]

        for seg in segments:
            start_ts = self._format_timestamp(seg["start"])
            end_ts = self._format_timestamp(seg["end"])
            lines.append(f"**{seg['speaker']}** [{start_ts} - {end_ts}]")
            lines.append(seg["text"])
            lines.append("")

        return "\n".join(lines)

    def _save_markdown(self, path: Path, content: str) -> None:
        """Markdownファイルを保存する"""
        path.write_text(content, encoding="utf-8")

    def _cleanup_temp_file(self) -> None:
        """一時ファイルを削除する"""
        try:
            if TEMP_AUDIO_PATH.exists():
                TEMP_AUDIO_PATH.unlink()
        except OSError:
            pass

    @rumps.clicked("終了")
    def quit_app(self, _: rumps.MenuItem) -> None:
        """アプリケーションを終了する"""
        rumps.quit_application()


if __name__ == "__main__":
    VoiceToMdApp().run()
