"""処理中の進捗を表示するウィンドウ（PyObjC版）"""

from typing import Optional, Callable
import objc
import AppKit
from Foundation import NSMakeRect, NSObject

from .config import WINDOW_WIDTH, WINDOW_HEIGHT


class MainThreadExecutor(NSObject):
    """メインスレッドでコードを実行するためのヘルパー"""

    @objc.python_method
    def execute_on_main_thread(self, block: Callable[[], None]) -> None:
        """ブロックをメインスレッドで実行する"""
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "runBlock:", block, True
        )

    def runBlock_(self, block: Callable[[], None]) -> None:
        """実際にブロックを実行する（Objective-Cから呼び出される）"""
        if block:
            block()


class ProgressWindow:
    """処理中の進捗を表示するウィンドウ"""

    def __init__(self) -> None:
        self._window: Optional[AppKit.NSWindow] = None
        self._status_label: Optional[AppKit.NSTextField] = None
        self._progress_label: Optional[AppKit.NSTextField] = None
        self._progress_indicator: Optional[AppKit.NSProgressIndicator] = None
        self._initialized = False
        self._executor = MainThreadExecutor.alloc().init()

    def _run_on_main_thread(self, block: Callable[[], None]) -> None:
        """メインスレッドでブロックを実行する"""
        self._executor.execute_on_main_thread(block)

    def _setup_window(self) -> None:
        """ウィンドウを初期化する（メインスレッドで呼び出す）"""
        if self._initialized:
            return

        # 画面中央に配置
        screen = AppKit.NSScreen.mainScreen()
        screen_rect = screen.visibleFrame()
        window_height = WINDOW_HEIGHT + 30  # 進捗ラベル用に高さを追加
        x = screen_rect.origin.x + (screen_rect.size.width - WINDOW_WIDTH) / 2
        y = screen_rect.origin.y + (screen_rect.size.height - window_height) / 2

        # ウィンドウを作成
        window_rect = NSMakeRect(x, y, WINDOW_WIDTH, window_height)
        style_mask = AppKit.NSWindowStyleMaskTitled
        self._window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            window_rect,
            style_mask,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self._window.setTitle_("処理中...")
        self._window.setLevel_(AppKit.NSFloatingWindowLevel)
        self._window.setReleasedWhenClosed_(False)

        content_view = self._window.contentView()

        # ステータスラベル（処理内容）
        label_rect = NSMakeRect(20, 110, WINDOW_WIDTH - 40, 30)
        self._status_label = AppKit.NSTextField.alloc().initWithFrame_(label_rect)
        self._status_label.setEditable_(False)
        self._status_label.setBezeled_(False)
        self._status_label.setDrawsBackground_(False)
        self._status_label.setAlignment_(AppKit.NSTextAlignmentCenter)
        self._status_label.setStringValue_("準備中...")
        self._status_label.setFont_(AppKit.NSFont.systemFontOfSize_(14))
        content_view.addSubview_(self._status_label)

        # 進捗ラベル（パーセント表示）
        progress_label_rect = NSMakeRect(20, 80, WINDOW_WIDTH - 40, 25)
        self._progress_label = AppKit.NSTextField.alloc().initWithFrame_(progress_label_rect)
        self._progress_label.setEditable_(False)
        self._progress_label.setBezeled_(False)
        self._progress_label.setDrawsBackground_(False)
        self._progress_label.setAlignment_(AppKit.NSTextAlignmentCenter)
        self._progress_label.setStringValue_("0%")
        self._progress_label.setFont_(AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(16, AppKit.NSFontWeightMedium))
        content_view.addSubview_(self._progress_label)

        # プログレスバー（確定モード）
        progress_rect = NSMakeRect(40, 50, WINDOW_WIDTH - 80, 20)
        self._progress_indicator = AppKit.NSProgressIndicator.alloc().initWithFrame_(progress_rect)
        self._progress_indicator.setStyle_(AppKit.NSProgressIndicatorStyleBar)
        self._progress_indicator.setIndeterminate_(False)
        self._progress_indicator.setMinValue_(0.0)
        self._progress_indicator.setMaxValue_(100.0)
        self._progress_indicator.setDoubleValue_(0.0)
        content_view.addSubview_(self._progress_indicator)

        self._initialized = True

    def show(self) -> None:
        """ウィンドウを表示する"""
        def _show():
            self._setup_window()
            if self._window:
                self._progress_indicator.setDoubleValue_(0.0)
                self._progress_label.setStringValue_("0%")
                self._window.makeKeyAndOrderFront_(None)
        self._run_on_main_thread(_show)

    def hide(self) -> None:
        """ウィンドウを非表示にする"""
        def _hide():
            if self._window:
                self._window.orderOut_(None)
        self._run_on_main_thread(_hide)

    def set_status(self, status: str) -> None:
        """ステータスを設定する"""
        def _set_status():
            if self._status_label:
                self._status_label.setStringValue_(status)
        self._run_on_main_thread(_set_status)

    def set_progress(self, progress: float) -> None:
        """進捗を設定する（0.0〜100.0）"""
        def _set_progress():
            if self._progress_indicator:
                self._progress_indicator.setDoubleValue_(progress)
            if self._progress_label:
                self._progress_label.setStringValue_(f"{int(progress)}%")
        self._run_on_main_thread(_set_progress)

    def set_status_with_progress(self, status: str, progress: float) -> None:
        """ステータスと進捗を同時に設定する"""
        def _update():
            if self._status_label:
                self._status_label.setStringValue_(status)
            if self._progress_indicator:
                self._progress_indicator.setDoubleValue_(progress)
            if self._progress_label:
                self._progress_label.setStringValue_(f"{int(progress)}%")
        self._run_on_main_thread(_update)
