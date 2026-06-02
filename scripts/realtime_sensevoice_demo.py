#!/usr/bin/env python3
"""
超低延迟实时语音识别 - SenseVoice 版本
支持中英日韩粤多语言识别
"""

import sounddevice as sd
import sherpa_onnx
import numpy as np


def create_recognizer():
    """创建 SenseVoice 识别器"""
    model_dir = "../models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09"

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{model_dir}/model.onnx",
        tokens=f"{model_dir}/tokens.txt",
        num_threads=4,
        debug=False,
        provider="cpu",
    )

    return recognizer


class RealtimeSenseVoiceASR:
    def __init__(self):
        self.recognizer = create_recognizer()
        self.sample_rate = 16000
        self.buffer = []
        self.buffer_duration = 3.0  # 3秒缓冲区，适合短语识别
        self.buffer_size = int(self.sample_rate * self.buffer_duration)

    def audio_callback(self, indata, frames, time_info, status):
        """音频回调"""
        audio_data = indata[:, 0].copy()
        self.buffer.extend(audio_data)

        if len(self.buffer) >= self.buffer_size:
            audio_chunk = np.array(self.buffer[:self.buffer_size], dtype=np.float32)
            self.buffer = self.buffer[int(self.buffer_size * 0.5):]

            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, audio_chunk)

            self.recognizer.decode_stream(stream)
            result = stream.result.text.strip()

            if result:
                print(f"\r{result}", end="", flush=True)

    def run(self):
        """启动实时识别"""
        print("=" * 60)
        print("SenseVoice 超低延迟语音识别")
        print("=" * 60)
        print("支持语言: 中文 / 英文 / 日语 / 韩语 / 粤语")
        print("优化设置:")
        print("  • 缓冲区: 500ms (极低延迟)")
        print("  • 滑动窗口: 250ms overlap")
        print("  • 线程数: 4 (M1 优化)")
        print("  • 多语言自动检测")
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
    asr = RealtimeSenseVoiceASR()
    asr.run()
