"""Voice to MD アプリケーションのエントリーポイント"""

import threading
from datetime import datetime
from pathlib import Path

import rumps
import librosa

from .config import OUTPUT_DIR, TEMP_AUDIO_PATH, SAMPLE_RATE, DEVICE
from .recorder import Recorder
from .transcriber import Transcriber, TranscriptionSegment
from .progress_window import ProgressWindow


class VoiceToMdApp(rumps.App):
    """メニューバーに常駐する音声文字起こしアプリケーション"""

    # 進捗の配分（合計100%）
    PROGRESS_TRANSCRIBE = 95  # 文字起こし: 0-95%
    PROGRESS_FINALIZE = 5  # 最終処理: 95-100%

    def __init__(self) -> None:
        super().__init__(name="Voice to MD", title="🎤 Voice")
        self._recorder = Recorder()
        self._transcriber = Transcriber()
        self._progress_window = ProgressWindow()
        self._is_recording = False
        self._is_processing = False

        # デバイス情報を表示
        device_info = "GPU (Apple Silicon)" if DEVICE == "mps" else "CPU"
        rumps.notification(
            title="Voice to MD",
            subtitle="起動しました",
            message=f"デバイス: {device_info}",
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
            self._progress_window.set_status_with_progress("準備中...", 0)

            # 録音時間を取得
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            duration_sec = len(audio) / sr
            duration_str = self._format_duration(duration_sec)

            # 文字起こし（0-95%）
            def transcribe_progress(p: float) -> None:
                progress = p * self.PROGRESS_TRANSCRIBE
                self._progress_window.set_status_with_progress(
                    f"文字起こし中... ({int(p * 100)}%)",
                    progress
                )

            try:
                transcription_segments = self._transcriber.transcribe(
                    audio_path,
                    progress_callback=transcribe_progress
                )
            except Exception as e:
                print(f"文字起こしエラー: {e}")
                transcription_segments = []

            # 結果を準備（95%）
            self._progress_window.set_status_with_progress("結果を準備中...", 95)
            merged_segments = [
                {
                    "speaker": "Speaker 1",
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in transcription_segments
            ]

            # Markdownファイルを生成（97%）
            self._progress_window.set_status_with_progress("ファイルを保存中...", 97)
            now = datetime.now()
            filename = f"voice_{now.strftime('%Y%m%d_%H%M%S')}.md"
            output_path = OUTPUT_DIR / filename

            content = self._create_markdown_content(
                now, duration_str, 1, merged_segments
            )
            self._save_markdown(output_path, content)

            # 完了（100%）
            self._progress_window.set_status_with_progress("完了!", 100)
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
