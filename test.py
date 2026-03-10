import mlx_whisper
import os
import sys

def transcribe_audio(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        return

    print(f"🚀 正在唤醒本地 AI 模型 (mlx-community/whisper-small-mlx)...")
    print("💡 第一次运行会自动下载模型权重，请保持网络连接。")
    
    try:
        # 使用 path_or_hf_repo 指定模型
        result = mlx_whisper.transcribe(
            file_path, 
            path_or_hf_repo="mlx-community/whisper-small-mlx"
        )
        
        print("\n✨ 识别成功！")
        print("-" * 30)
        print(f"📝 识别结果：\n{result['text']}")
        print("-" * 30)
    except Exception as e:
        print(f"❌ 运行出错：{str(e)}")

if __name__ == "__main__":
    test_file = "test.m4a"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    transcribe_audio(test_file)
