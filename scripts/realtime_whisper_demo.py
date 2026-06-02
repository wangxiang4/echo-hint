#!/usr/bin/env python3
"""
超低延迟实时语音识别 - Whisper 版本
使用 Distil-Large-v3.5 模型，支持多语言
"""

import sounddevice as sd
import sherpa_onnx
import numpy as np


def create_recognizer():
    """创建 Whisper 识别器"""
    model_dir = "../models/sherpa-onnx-whisper-distil-large-v3.5"

    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=f"{model_dir}/distil-large-v3.5-encoder.int8.onnx",
        decoder=f"{model_dir}/distil-large-v3.5-decoder.int8.onnx",
        tokens=f"{model_dir}/distil-large-v3.5-tokens.txt",
        num_threads=4,
        provider="cpu",
        language="zh",
        task="transcribe",
    )

    return recognizer


class RealtimeWhisperASR:
    def __init__(self):
        self.recognizer = create_recognizer()
        self.sample_rate = 16000
        self.buffer = []
        self.buffer_duration = 3.0
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        self.last_text = ""

    def audio_callback(self, indata, frames, time_info, status):
        """音频回调"""
        audio_data = indata[:, 0].copy()
        self.buffer.extend(audio_data)

        if len(self.buffer) >= self.buffer_size:
            audio_chunk = np.array(self.buffer, dtype=np.float32)
            self.buffer = []

            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, audio_chunk)

            self.recognizer.decode_stream(stream)
            result = stream.result.text.strip()

            if result and result != self.last_text:
                print(f"\n> {result}")
                self.last_text = result

    def run(self):
        """启动实时识别"""
        print("=" * 60)
        print("Whisper 超低延迟语音识别")
        print("=" * 60)
        print("模型: Distil-Large-v3.5 (INT8 量化)")
        print("支持: 多语言自动检测")
        print("优化设置:")
        print("  • 缓冲区: 3秒")
        print("  • INT8 量化加速")
        print("  • 线程数: 4 (M1 优化)")
        print("  • CPU 推理优化")
        print("-" * 60)
        print("开始监听... (Ctrl+C 停止)\n")

        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.float32,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1),
                latency='low',
            ):
                while True:
                    sd.sleep(100)

        except KeyboardInterrupt:
            print("\n\n识别已停止")
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    asr = RealtimeWhisperASR()
    asr.run()
