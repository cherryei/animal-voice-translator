"""
数据准备脚本：从 CatMeows 和 ESC-50 中提取并整理训练数据
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import shutil
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def prepare_catmeows():
    """解析 CatMeows 数据集，文件名编码了标签"""
    catmeows_dir = RAW_DIR / "CatMeows" / "dataset"
    if not catmeows_dir.exists():
        print(f"CatMeows 目录不存在: {catmeows_dir}")
        return []

    records = []
    for wav_path in catmeows_dir.glob("*.wav"):
        filename = wav_path.stem  # e.g. B_MAT01_EU_FN_RIT01_101
        parts = filename.split("_")
        context_code = parts[0]  # B, F, or I

        context_map = {
            "B": "brushing",     # 被梳毛
            "F": "food",         # 等食物
            "I": "isolation",    # 陌生环境隔离
        }
        context = context_map.get(context_code, "unknown")
        if context == "unknown":
            continue

        records.append({
            "path": str(wav_path),
            "species": "cat",
            "label": context,
            "source": "CatMeows",
        })
    print(f"CatMeows: {len(records)} 条记录 (brushing={sum(1 for r in records if r['label']=='brushing')}, "
          f"food={sum(1 for r in records if r['label']=='food')}, "
          f"isolation={sum(1 for r in records if r['label']=='isolation')})")
    return records


def prepare_esc50_animals():
    """从 ESC-50 提取动物声音数据"""
    esc50_dir = RAW_DIR / "ESC-50-master"
    meta_path = esc50_dir / "meta" / "esc50.csv"
    audio_dir = esc50_dir / "audio"

    if not meta_path.exists() or not audio_dir.exists():
        print(f"ESC-50 目录不完整: {esc50_dir}")
        return []

    meta = pd.read_csv(meta_path)
    # 只取动物类（按类别名称过滤）
    animal_categories = ["dog", "cat", "rooster", "pig", "cow", "frog", "hen", "sheep", "crow"]
    animal_meta = meta[meta["category"].isin(animal_categories)]
    print(f"ESC-50 动物类: {animal_meta['category'].value_counts().to_dict()}")

    species_map = {
        "dog": "dog", "cat": "cat", "rooster": "rooster",
        "pig": "pig", "cow": "cow", "frog": "frog",
        "hen": "hen", "sheep": "sheep", "crow": "crow",
    }

    # label 映射: 将动物+情境组合映射到语义标签
    # ESC-50 没有情境标签，我们用叫声类型作为标签
    labels_map = {
        "dog": "barking",
        "cat": "meowing",
        "rooster": "crowing",
        "pig": "grunting",
        "cow": "mooing",
        "frog": "croaking",
        "hen": "clucking",
        "sheep": "bleating",
        "crow": "cawing",
    }

    records = []
    for _, row in animal_meta.iterrows():
        category = row["category"]
        if category not in species_map:
            continue
        wav_path = audio_dir / row["filename"]
        if not wav_path.exists():
            continue
        records.append({
            "path": str(wav_path),
            "species": species_map[category],
            "label": labels_map.get(category, category),
            "source": "ESC-50",
        })

    print(f"ESC-50 animals: {len(records)} 条记录")
    return records


def split_and_save(records, output_dir):
    """按 70/15/15 拆分训练/验证/测试集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按物种-标签分层次拆分，保证各集合分布一致
    df = pd.DataFrame(records)

    train_dfs = []
    val_dfs = []
    test_dfs = []

    for (species, label), group in df.groupby(["species", "label"]):
        if len(group) < 3:
            # 太少就全放训练
            train_dfs.append(group)
            continue
        train, temp = train_test_split(group, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    # 保存 CSV
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    print(f"  总计: {len(train_df) + len(val_df) + len(test_df)} 条")

    # 保存标签映射
    all_labels = sorted(df["label"].unique())
    label_to_id = {l: i for i, l in enumerate(all_labels)}
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_to_id, f, indent=2)
    print(f"  标签: {label_to_id}")

    return train_df, val_df, test_df


def main():
    print("=" * 60)
    print("开始数据准备")
    print("=" * 60)

    # 收集所有数据
    all_records = []
    all_records.extend(prepare_catmeows())
    all_records.extend(prepare_esc50_animals())

    print(f"\n共收集 {len(all_records)} 条音频记录")

    if len(all_records) == 0:
        print("错误：没有找到任何数据！")
        sys.exit(1)

    # 按物种统计
    df = pd.DataFrame(all_records)
    print(f"\n物种分布:")
    for species, count in df["species"].value_counts().items():
        print(f"  {species}: {count}")
    print(f"\n标签分布:")
    for label, count in df["label"].value_counts().items():
        print(f"  {label}: {count}")

    # 拆分并保存
    split_and_save(all_records, PROCESSED_DIR)

    print(f"\n数据准备完成！保存到: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
