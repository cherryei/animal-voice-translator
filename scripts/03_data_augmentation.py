"""
数据增强：用 audiomentations 扩充训练数据集
MacBook CPU 可运行，几分钟即可完成
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import shutil

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUGMENTED_DIR = DATA_DIR / "augmented"
AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

# 简单的手工数据增强（不需要额外安装 audiomentations）
# 直接在 numpy 层面操作


def add_noise(audio, noise_level=0.005):
    """添加背景噪声"""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise


def shift_time(audio, shift_max=0.1):
    """时间偏移（模拟不同步的录音起始点）"""
    shift = int(np.random.randn() * len(audio) * shift_max)
    shift = np.clip(shift, -len(audio) // 4, len(audio) // 4)
    if shift > 0:
        return np.pad(audio[shift:], (0, shift))
    else:
        return np.pad(audio[:shift], (-shift, 0))


def pitch_shift(audio, sr=16000, semitones=2):
    """简单音高偏移（通过重采样模拟）"""
    rate = 2 ** (semitones / 12)
    indices = np.arange(0, len(audio), rate).astype(int)
    indices = indices[indices < len(audio)]
    return audio[indices]


def augment_audio(audio, sr=16000):
    """对一条音频应用一组随机增强"""
    augments = []
    augments.append(add_noise(audio, noise_level=np.random.uniform(0.001, 0.01)))
    augments.append(add_noise(audio, noise_level=np.random.uniform(0.005, 0.02)))
    augments.append(shift_time(audio, shift_max=np.random.uniform(0.05, 0.2)))
    augments.append(pitch_shift(audio, sr, semitones=np.random.randint(-3, 4)))
    # 组合增强
    augments.append(add_noise(shift_time(audio), noise_level=0.005))
    augments.append(pitch_shift(add_noise(audio), sr, semitones=np.random.randint(-2, 3)))
    return augments


def main():
    print("=" * 60)
    print("数据增强")
    print("=" * 60)

    # 读取训练集
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    print(f"原始训练集: {len(train_df)} 条")

    aug_records = []
    for _, row in tqdm(train_df.iterrows(), desc="增强", total=len(train_df)):
        try:
            audio, sr = sf.read(row["path"])
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            aug_audios = augment_audio(audio, sr)

            for i, aug_audio in enumerate(aug_audios):
                aug_name = f"{Path(row['path']).stem}_aug{i}.wav"
                aug_path = AUGMENTED_DIR / aug_name
                # 确保长度一致
                target_len = 5 * sr
                if len(aug_audio) > target_len:
                    aug_audio = aug_audio[:target_len]
                else:
                    aug_audio = np.pad(aug_audio, (0, max(0, target_len - len(aug_audio))))
                sf.write(str(aug_path), aug_audio, sr)
                aug_records.append({
                    "path": str(aug_path),
                    "species": row["species"],
                    "label": row["label"],
                    "source": f"{row['source']}_aug",
                })
        except Exception as e:
            tqdm.write(f"  跳过 {row['path']}: {e}")

    aug_df = pd.DataFrame(aug_records)
    print(f"增强后新增: {len(aug_df)} 条")

    # 合并到新训练集
    combined = pd.concat([train_df, aug_df]).reset_index(drop=True)
    combined.to_csv(PROCESSED_DIR / "train_augmented.csv", index=False)
    print(f"增强后训练集: {len(combined)} 条")

    # 标签分布
    print(f"\n标签分布:")
    for label, count in combined["label"].value_counts().items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
