#!/usr/bin/env python3
"""
下载 Zipformer 语音识别模型
支持多个中文和英文模型选择
"""

import sys
import urllib.request
import tarfile
from pathlib import Path


MODELS = {
    "sense-voice": {
        "name": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09.tar.bz2",
        "description": "sense-voice模型",
    },
    "whisper": {
        "name": "sherpa-onnx-whisper-distil-large-v3.5",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-distil-large-v3.5.tar.bz2",
        "description": "whisper模型",
    }
}


def download_with_progress(url, output_path):
    """下载文件并显示进度"""

    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r下载进度: {percent}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook)
    print()


def extract_archive(archive_path, extract_to):
    """解压 tar.bz2 文件"""
    print(f"正在解压到 {extract_to}...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(path=extract_to)
    print("解压完成！")



def download_model(model_key):
    """下载并解压指定模型"""
    if model_key not in MODELS:
        print(f"错误: 未知的模型 '{model_key}'")
        print(f"可用模型: {', '.join(MODELS.keys())}")
        return False

    model = MODELS[model_key]
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / model["name"]
    if model_path.exists():
        print(f"模型已存在: {model_path}")
        return True

    archive_path = models_dir / f"{model['name']}.tar.bz2"

    print(f"下载模型: {model['description']}")
    print(f"URL: {model['url']}")
    print()

    try:
        download_with_progress(model["url"], archive_path)
        extract_archive(archive_path, models_dir)

        archive_path.unlink()
        print(f"模型已保存到: {model_path}")
        return True

    except Exception as e:
        print(f"下载失败: {e}")
        if archive_path.exists():
            archive_path.unlink()
        return False


def main():
    """主函数"""
    print("=== Sherpa-ONNX Zipformer 模型下载器 ===\n")

    print("可用模型:")
    for key, model in MODELS.items():
        print(f"  [{key}] {model['description']}")

    print()

    if len(sys.argv) > 1:
        model_key = sys.argv[1]
    else:
        model_key = input("请选择模型 (默认: zh-14M): ").strip() or "zh-14M"

    success = download_model(model_key)

    if success:
        print("\n✓ 模型下载成功！")
        print(f"\n现在可以运行: python scripts/realtime_asr_demo.py")
    else:
        print("\n✗ 模型下载失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
