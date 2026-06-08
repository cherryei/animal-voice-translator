"""
推理脚本：用训练好的分类器对任意音频文件进行预测
"""

import os, sys, warnings, json, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"


class MLPClassifier(nn.Module):
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


class AnimalVoiceClassifier:
    """动物叫声分类器封装"""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODELS_DIR / "mlp_classifier_final.pt"

        # 加载模型元数据
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.label_map = checkpoint["label_map"]
        self.id_to_label = checkpoint["id_to_label"]
        self.num_classes = checkpoint["num_classes"]
        self.config = checkpoint["config"]

        # 加载音频编码器
        self.device = "cpu"
        model_name = self.config.get("audio_model_name", "facebook/wav2vec2-base")
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.audio_model = AutoModel.from_pretrained(model_name)
        self.audio_model.eval()

        # 加载分类器
        self.classifier = MLPClassifier(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=self.num_classes,
            n_layers=checkpoint.get("n_layers", 2),
        )
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.eval()

        # 加载标准化器
        scaler_path = MODELS_DIR / "feature_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None

        print(f"模型加载完成: {model_path}")
        print(f"  类别: {self.label_map}")

    def extract_features(self, audio):
        """提取音频特征"""
        inputs = self.processor(
            [audio], sampling_rate=16000,
            return_tensors="pt", padding=True, truncation=True,
            max_length=16000 * 5,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.audio_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            feat_mean = hidden_states.mean(dim=1)
            feat_std = hidden_states.std(dim=1)
            feat_max = hidden_states.max(dim=1).values
            features = torch.cat([feat_mean, feat_std, feat_max], dim=1)

        return features.cpu()

    def predict(self, audio_path_or_array, sr=None):
        """
        预测音频的情绪/情境类别

        参数:
            audio_path_or_array: 音频文件路径 或 numpy 数组
            sr: 如果传入数组，需要提供采样率

        返回:
            dict: 分类结果
        """
        # 加载音频
        if isinstance(audio_path_or_array, (str, Path)):
            audio, sr = sf.read(str(audio_path_or_array))
        else:
            audio = audio_path_or_array

        # 转为单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # 重采样到 16kHz
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # 统一长度
        target_len = 16000 * 5
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, max(0, target_len - len(audio))))

        # 提取特征
        features = self.extract_features(audio)

        # 标准化
        if self.scaler:
            features = torch.FloatTensor(self.scaler.transform(features.numpy()))

        # 预测
        with torch.no_grad():
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        # 解析结果
        top_idx = probs.argmax().item()
        top_label = self.id_to_label[top_idx]
        top_prob = probs[top_idx].item()

        # 所有类别的概率
        all_probs = {}
        for i in range(self.num_classes):
            label = self.id_to_label[i]
            all_probs[label] = round(probs[i].item(), 4)

        return {
            "prediction": top_label,
            "confidence": round(top_prob, 4),
            "all_probabilities": dict(sorted(all_probs.items(), key=lambda x: -x[1])),
            "num_classes": self.num_classes,
        }

    def predict_with_audio_array(self, audio, sr):
        """传入 numpy 数组进行预测"""
        return self.predict(audio, sr=sr)


def main():
    """命令行使用示例"""
    import argparse
    parser = argparse.ArgumentParser(description="动物叫声分类推理")
    parser.add_argument("audio_path", help="音频文件路径")
    parser.add_argument("--model", default=None, help="模型文件路径")
    args = parser.parse_args()

    classifier = AnimalVoiceClassifier(args.model)

    if not os.path.exists(args.audio_path):
        print(f"文件不存在: {args.audio_path}")
        sys.exit(1)

    result = classifier.predict(args.audio_path)
    print(f"\n预测结果:")
    print(f"  类别: {result['prediction']}")
    print(f"  置信度: {result['confidence']:.2%}")
    print(f"\n所有类别概率:")
    for label, prob in result["all_probabilities"].items():
        bar = "█" * int(prob * 50)
        print(f"  {label:>12}: {prob:>6.2%} {bar}")


if __name__ == "__main__":
    main()
