#!/usr/bin/env python3
"""
超低延迟实时语音识别
优化用于面试场景：立即输出识别结果
"""

import sounddevice as sd
import sherpa_onnx
import numpy as np

def create_recognizer():
    """创建低延迟识别器"""
    model_dir = "../models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{model_dir}/tokens.txt",
        encoder=f"{model_dir}/encoder-epoch-99-avg-1.onnx",
        decoder=f"{model_dir}/decoder-epoch-99-avg-1.onnx",
        joiner=f"{model_dir}/joiner-epoch-99-avg-1.onnx",
        num_threads=4,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=1.0,
        rule2_min_trailing_silence=1.0,
        rule3_min_utterance_length=180,
    )

    return recognizer


class RealtimeASR:
    def __init__(self):
        self.recognizer = create_recognizer()
        self.stream = self.recognizer.create_stream()
        self.sample_rate = 16000
        self.last_text = ""

    def audio_callback(self, indata, frames, time_info, status):
        """超低延迟音频回调"""
        audio_data = indata[:, 0]
        self.stream.accept_waveform(self.sample_rate, audio_data)

        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        result = self.recognizer.get_result(self.stream)
        current_text = result.strip()

        if current_text and current_text != self.last_text:
            print(f"\r{current_text}", end="", flush=True)
            self.last_text = current_text

        if self.recognizer.is_endpoint(self.stream):
            if current_text:
                print(f"\n> {current_text}")
            self.recognizer.reset(self.stream)
            self.last_text = ""

    def run(self):
        """启动实时识别"""
        print("=" * 60)
        print("超低延迟语音识别 - 面试模式")
        print("=" * 60)
        print("优化设置:")
        print("  • 音频块: 32ms (极低延迟)")
        print("  • 端点检测: 0.4-0.8秒 (快速响应)")
        print("  • 线程数: 4 (M1 优化)")
        print("  • 实时逐字输出")
        print("-" * 60)
        print("开始监听... (Ctrl+C 停止)\n")

        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.float32,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.032),
                latency='low',
            ):
                while True:
                    sd.sleep(50)

        except KeyboardInterrupt:
            print("\n\n识别已停止")
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    asr = RealtimeASR()
    asr.run()
