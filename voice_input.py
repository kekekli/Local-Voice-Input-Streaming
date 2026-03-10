import os
import time
import queue
import threading
import subprocess
import sounddevice as sd
import soundfile as sf
import mlx_whisper
from pynput import keyboard

# --------- 配置区域 ---------
MODEL = "mlx-community/whisper-small-mlx" 
HOTKEY = '<ctrl>+<alt>+r' # 控制开始/停止录音的快捷键
SAMPLE_RATE = 16000 
CHANNELS = 1 
TEMP_FILE = "temp_recording.wav"
# --------------------------

is_recording = False
audio_queue = queue.Queue()
recording_thread = None
kb_controller = keyboard.Controller()

def show_notification(title, text):
    """在 Mac 右上角弹出原生通知，确认识别结果"""
    # 处理字符串里的引号，防止命令报错
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    os.system(f'osascript -e \'display notification "{safe_text}" with title "{title}"\'')

def auto_type_text(text):
    """直接模拟键盘，把文字打进你当前光标所在的输入框"""
    # 模拟输入前稍微等一下，确保焦点在原来的窗口
    time.sleep(0.1)
    kb_controller.type(text)

def copy_to_clipboard(text):
    process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
    process.communicate(text.encode('utf-8'))

def audio_callback(indata, frames, time, status):
    if status:
        pass
    audio_queue.put(indata.copy())

def record_audio():
    global is_recording
    frames = []
    print("\n🟢 [开始录音] 正在倾听...")
    os.system("afplay /System/Library/Sounds/Ping.aiff")
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        while is_recording:
            try:
                data = audio_queue.get(timeout=0.1)
                frames.append(data)
            except queue.Empty:
                continue

    if frames:
        import numpy as np
        audio_data = np.concatenate(frames, axis=0)
        sf.write(TEMP_FILE, audio_data, SAMPLE_RATE)
        print("🔴 [录音结束] 正在转写...")
        transcribe_and_type()

def transcribe_and_type():
    try:
        result = mlx_whisper.transcribe(TEMP_FILE, path_or_hf_repo=MODEL)
        text = result.get('text', '').strip()
        
        if text:
            # 1. 复制文本，以备你需要时手动粘贴
            copy_to_clipboard(text)
            
            # 2. 屏幕右上角弹出 Mac 提示框，让你看到你说的话
            show_notification("✅ 语音识别成功", text)
            print(f"✨ 文本：{text}")
            
            # 3. 自动帮你“打字”到光标所在的输入框！
            auto_type_text(text)
            
            os.system("afplay /System/Library/Sounds/Glass.aiff")
        else:
            show_notification("⚠️ 警告", "没有听到任何声音或未能识别出文字。")
    except Exception as e:
        print(f"❌ 转写失败：{e}")
        show_notification("❌ 错误", str(e))

def toggle_recording():
    global is_recording, recording_thread
    if not is_recording:
        is_recording = True
        while not audio_queue.empty():
            audio_queue.get()
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
    else:
        is_recording = False
        if recording_thread:
            recording_thread.join()

def on_activate():
    toggle_recording()

def main():
    print("🚀 初始化完成！")
    try:
        mlx_whisper.transcribe("test.m4a", path_or_hf_repo=MODEL)
        print("✅ 模型预热完毕")
        os.system("afplay /System/Library/Sounds/Tink.aiff")
    except:
        pass

    print("🎙️ 后台运行中...")
    print("按 Control + Option + R 开始/停止录音")
    with keyboard.GlobalHotKeys({HOTKEY: on_activate}) as h:
        h.join()

if __name__ == "__main__":
    main()
