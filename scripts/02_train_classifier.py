"""
BEATs 特征提取 + MLP 分类器训练
- 使用微软 BEATs 编码器提取音频特征（无需 GPU，CPU 可运行）
- 训练轻量 MLP 分类器
- 保存模型权重和归一化参数
"""

import os, sys, json, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "audio_model_name": "facebook/wav2vec2-base",
    "target_sr": 16000,
    "max_length_sec": 5,
    "mlp_hidden_dim": 256,       # 数据量增加，增大模型容量
    "mlp_n_layers": 2,           # 两层 MLP
    "mlp_dropout": 0.4,          # 增强正则化
    "batch_size": 32,
    "epochs": 150,
    "learning_rate": 5e-4,
    "weight_decay": 1e-3,
    "early_stop_patience": 20,
}


class AudioDataset(Dataset):
    def __init__(self, df, target_sr=16000, max_len=5):
        self.df = df.reset_index(drop=True) if isinstance(df, pd.DataFrame) else pd.read_csv(df)
        self.target_sr = target_sr
        self.max_samples = target_sr * max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            audio, sr = sf.read(row["path"])
            # 转为单声道
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            # 重采样到 target_sr
            if sr != self.target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            # 统一长度
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            else:
                audio = np.pad(audio, (0, max(0, self.max_samples - len(audio))))
            return torch.FloatTensor(audio)
        except Exception as e:
            print(f"读取失败 {row['path']}: {e}")
            return torch.zeros(self.max_samples)


class MLPClassifier(nn.Module):
    """MLP 分类器（支持多层）"""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3, n_layers=2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def extract_audio_features(model, processor, audio_batch, device="cpu"):
    """使用 wav2vec2 等音频模型提取特征"""
    inputs = processor(
        audio_batch.numpy(), sampling_rate=CONFIG["target_sr"],
        return_tensors="pt", padding=True, truncation=True,
        max_length=CONFIG["target_sr"] * CONFIG["max_length_sec"],
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 取最后一层 hidden states
        hidden_states = outputs.hidden_states[-1]  # [B, T, D]
        # 统计池化: mean + std + max
        feat_mean = hidden_states.mean(dim=1)
        feat_std = hidden_states.std(dim=1)
        feat_max = hidden_states.max(dim=1).values
        features = torch.cat([feat_mean, feat_std, feat_max], dim=1)

    return features.cpu()


def train_epoch(model, loader, criterion, optimizer, device="cpu"):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(loader, desc="训练", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, all_preds, all_labels


def eval_model(model, loader, criterion, device="cpu"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="评估", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, all_preds, all_labels


def main():
    print("=" * 60)
    print("BEATs 特征提取 + MLP 分类器训练")
    print("=" * 60)
    print(f"配置: {json.dumps(CONFIG, indent=2)}")
    print()

    device = "cpu"
    print(f"使用设备: {device}")
    print()

    # 1. 加载数据（优先使用增强后的训练集）
    print("=" * 40)
    print("加载数据...")
    aug_train_path = PROCESSED_DIR / "train_augmented.csv"
    if aug_train_path.exists():
        train_df = pd.read_csv(aug_train_path)
        print(f"使用增强训练集: {len(train_df)} 条")
    else:
        train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
        print(f"使用原始训练集: {len(train_df)} 条")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    with open(PROCESSED_DIR / "label_map.json") as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    print(f"训练: {len(train_df)}, 验证: {len(val_df)}, 测试: {len(test_df)}, 类别: {num_classes}")

    # 2. 加载音频模型并提取特征
    print("\n" + "=" * 40)
    print(f"加载 {CONFIG['audio_model_name']} 并提取特征...")
    print("（首次运行会自动下载模型权重）")

    model_name = CONFIG["audio_model_name"]
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    audio_model = AutoModel.from_pretrained(model_name)
    audio_model.eval()

    # wav2vec2-base 输出维度: hidden_size=768, 统计池化后 = 768*3 = 2304
    input_dim = 768 * 3
    print(f"模型特征维度: {input_dim}")

    # 批量提取特征
    def extract_features_batched(df, desc):
        dataset = AudioDataset(
            df,
            target_sr=CONFIG["target_sr"],
            max_len=CONFIG["max_length_sec"],
        )
        loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)
        features_list = []
        for batch in tqdm(loader, desc=f"特征提取 ({desc})"):
            feats = extract_audio_features(audio_model, processor, batch, device)
            features_list.append(feats)
        return torch.cat(features_list)

    t0 = time.time()

    # 检查缓存
    cache_path = PROCESSED_DIR / "features_cache.pt"
    if cache_path.exists():
        print("发现特征缓存，加载中...")
        cache = torch.load(cache_path)
        train_features = cache["train_features"]
        val_features = cache["val_features"]
        test_features = cache["test_features"]
        print(f"从缓存加载特征 (训练 {train_features.shape[0]}条)")
    else:
        train_features = extract_features_batched(train_df, "train")
        val_features = extract_features_batched(val_df, "val")
        test_features = extract_features_batched(test_df, "test")
        torch.save({
            "train_features": train_features,
            "val_features": val_features,
            "test_features": test_features,
        }, cache_path)
        print(f"特征已缓存到 {cache_path}")

    print(f"特征提取完成，耗时 {time.time()-t0:.1f} 秒")
    print(f"  训练集特征: {train_features.shape}")
    print(f"  验证集特征: {val_features.shape}")
    print(f"  测试集特征: {test_features.shape}")

    # 3. 特征标准化
    print("\n" + "=" * 40)
    print("特征标准化...")
    scaler = StandardScaler()
    train_features_np = scaler.fit_transform(train_features.numpy())
    val_features_np = scaler.transform(val_features.numpy())
    test_features_np = scaler.transform(test_features.numpy())

    # 保存 scaler
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"标准化参数保存到 {scaler_path}")

    # 4. 准备 DataLoader
    train_labels = torch.LongTensor([label_map[l] for l in train_df["label"]])
    val_labels = torch.LongTensor([label_map[l] for l in val_df["label"]])
    test_labels = torch.LongTensor([label_map[l] for l in test_df["label"]])

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_features_np), train_labels
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_features_np), val_labels
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_features_np), test_labels
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])

    # 5. 训练 MLP 分类器
    print("\n" + "=" * 40)
    print("训练 MLP 分类器...")

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=CONFIG["mlp_hidden_dim"],
        num_classes=num_classes,
        dropout=CONFIG["mlp_dropout"],
        n_layers=CONFIG["mlp_n_layers"],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc, _, _ = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion)

        print(f"  Epoch {epoch:2d}/{CONFIG['epochs']} | "
              f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
              f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
            torch.save(best_model_state, MODELS_DIR / "best_mlp_classifier.pt")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"  早停: {epoch} 轮后停止")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(MODELS_DIR / "best_mlp_classifier.pt"))

    # 6. 测试集评估
    print("\n" + "=" * 40)
    print("测试集评估...")
    test_loss, test_acc, test_preds, test_true = eval_model(model, test_loader, criterion)
    print(f"\n测试集准确率: {test_acc:.4f}")
    print(f"\n分类报告:")
    print(classification_report(test_true, test_preds,
                                target_names=[id_to_label[i] for i in range(num_classes)]))

    # 混淆矩阵
    cm = confusion_matrix(test_true, test_preds)
    print("混淆矩阵:")
    print(f"{'':>20}", end="")
    for l in range(num_classes):
        print(f"{id_to_label[l][:8]:>8}", end="")
    print()
    for i in range(num_classes):
        print(f"{id_to_label[i]:>20}", end="")
        for j in range(num_classes):
            print(f"{cm[i][j]:>8}", end="")
        print()

    # 7. 保存最终模型
    final_path = MODELS_DIR / "mlp_classifier_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": CONFIG["mlp_hidden_dim"],
        "n_layers": CONFIG["mlp_n_layers"],
        "num_classes": num_classes,
        "label_map": label_map,
        "id_to_label": id_to_label,
        "config": CONFIG,
        "test_accuracy": test_acc,
    }, final_path)
    print(f"\n最终模型保存到 {final_path}")

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
