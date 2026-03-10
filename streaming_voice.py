import sys
import threading
import time
import os

# 🌟 核心防闪退机制 1：禁止底层 Tokenizer 的多进程 fork 行为。
# 当 Mac 的 GUI 现成 (PyQt) 正在渲染时，任何背后的多进程 Fork 都会导致底层信号量 (Semaphore) 泄露和段错误崩坏。
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper

from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut

# --------- 配置区域 ---------
MODEL = "mlx-community/whisper-small-mlx"
SAMPLE_RATE = 16000
CHANNELS = 1
HOTKEY = '<ctrl>+<alt>+r'
# --------------------------

class Signals(QObject):
    update_text = pyqtSignal(str, bool)
    status_update = pyqtSignal(str, str)
    btn_update = pyqtSignal(str, bool)

class VoiceInputApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音捕捉")
        self.resize(320, 200)
        # 固定悬浮最前
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        
        # 居中显示
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

        self.signals = Signals()
        self.signals.update_text.connect(self.on_update_text)
        self.signals.status_update.connect(self.on_status_update)
        self.signals.btn_update.connect(self.on_btn_update)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 提示域
        self.status_label = QLabel("初始化中...")
        self.status_label.setStyleSheet("font-size: 14px; color: gray;")
        layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 文本框
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setStyleSheet("background-color: #282c34; color: #abb2bf; font-size: 14px; padding: 5px;")
        layout.addWidget(self.text_area)
        
        # 按钮区域
        self.record_btn = QPushButton("🔴 开始录音 (Ctrl+Alt+R)")
        self.record_btn.setStyleSheet("font-size: 14px; padding: 8px; background-color: #e0e0e0; color: black; border-radius: 5px;")
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)
        
        self.is_recording = False
        self.audio_data = []
        self.recording_thread = None
        self.transcribe_thread = None
        self.temp_file = "stream_temp.wav"
        self.mlx_lock = threading.Lock() # 🌟 增加 GPU 硬件锁，防止 Metal 引擎并发崩溃
        
        # 🌟 核心防闪退机制 2：彻底移除冲突的 pynput，改用 macOS 绝对安全的 Qt 内部快捷键
        # 注意：快捷键需要在点击、激活该窗口时生效。
        self.shortcut = QShortcut(QKeySequence("Ctrl+Alt+R"), self)
        self.shortcut.activated.connect(self.toggle_recording)
        
        # 开启预热线程
        threading.Thread(target=self.warmup, daemon=True).start()

    def warmup(self):
        try:
            self.signals.status_update.emit("⏳ 模型预热中，请稍候...", "gray")
            with self.mlx_lock:
                mlx_whisper.transcribe("test.m4a", path_or_hf_repo=MODEL)
            self.signals.status_update.emit("✅ 预热完毕！请点击下方按钮或按快捷键开始", "green")
        except Exception as e:
            self.signals.status_update.emit("✅ 就绪 (无测试文件)", "green")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
            os.system("afplay /System/Library/Sounds/Ping.aiff")
        else:
            self.stop_recording()
            
    def start_recording(self):
        self.is_recording = True
        self.signals.btn_update.emit("结束说话并复制 (⏹️)", True)
        self.signals.status_update.emit("🎙️ 正在倾听，请说话... (字幕正在生成)", "red")
        
        self.audio_data = []
        self.text_area.clear()
        
        self.recording_thread = threading.Thread(target=self.record_loop, daemon=True)
        self.recording_thread.start()
        
        self.transcribe_thread = threading.Thread(target=self.transcribe_loop, daemon=True)
        self.transcribe_thread.start()
        
    def stop_recording(self):
        self.is_recording = False
        self.signals.btn_update.emit("处理最后一段文字...", False)
        self.signals.status_update.emit("✅ 完成", "green")
        
        if self.recording_thread:
            self.recording_thread.join()
            
        # 录音结束后进行最后一次完整包含标点的转写
        self.do_transcribe(final=True)
        self.signals.btn_update.emit("🔴 开始录音 (Ctrl+Alt+R)", True)
        
    def record_loop(self):
        def callback(indata, frames, time, status):
            if status: pass
            self.audio_data.append(indata.copy())
            
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
            while self.is_recording:
                time.sleep(0.1)
                
    def transcribe_loop(self):
        while self.is_recording:
            time.sleep(1.5) # 每 1.5 秒丢给模型刷新一次文字
            if len(self.audio_data) > 0 and self.is_recording:
                self.do_transcribe(final=False)
                
    def do_transcribe(self, final=False):
        if not self.audio_data: return
        
        # 拼接波形
        audio_np = np.concatenate(self.audio_data, axis=0)
        sf.write(self.temp_file, audio_np, SAMPLE_RATE)
        
        try:
            # 强化中文标点提示词，并加上硬件锁，确保 GPU 安全
            with self.mlx_lock:
                result = mlx_whisper.transcribe(
                    self.temp_file, 
                    path_or_hf_repo=MODEL,
                    initial_prompt="这是一段口语化的中文对话，请务必准确识别，并为我的断句补充上逗号、句号和问号等标点符号。"
                )
            text = result.get('text', '').strip()
            
            if text:
                self.signals.update_text.emit(text, final)
        except Exception as e:
            print(f"转写出错 (被拦截): {e}")

    def on_update_text(self, text, final):
        self.text_area.clear()
        self.text_area.setText(text)
        
        # 滚动到底部
        scrollbar = self.text_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        if final:
            # 操作剪贴板
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.text_area.append("\n\n✅ 已复制到剪贴板！(含标点)")
            scrollbar.setValue(scrollbar.maximum())
            os.system("afplay /System/Library/Sounds/Glass.aiff")
            
    def on_status_update(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-size: 12px; color: {color}; font-weight: bold;")
        
    def on_btn_update(self, text, enabled):
        self.record_btn.setText(text)
        self.record_btn.setEnabled(enabled)

    def closeEvent(self, event):
        self.is_recording = False
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceInputApp()
    window.show()
    sys.exit(app.exec())
