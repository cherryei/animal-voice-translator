"""
训练手机端离线分类器
两端使用完全一致的简化 MFCC 算法（Python 训练 = Dart 推理）
"""

import os, sys, json, warnings, struct
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from pathlib import Path
from math import cos, pi, sqrt, log, exp, sin

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ====== 功率谱 ======
def power_spectrum(frame):
    """与 Dart FFT 算法一致的功率谱（使用 numpy 快速计算）"""
    n = len(frame)
    # np.fft.rfft 计算与手动 FFT 相同（标准算法）
    fft = np.fft.rfft(frame, n=n)
    return (fft.real ** 2 + fft.imag ** 2) / n


def hamming_window(size):
    return np.array([0.54 - 0.46 * cos(2 * pi * n / (size - 1)) for n in range(size)])


def hz_to_mel(hz):
    return 2595 * log(1 + hz / 700) / log(10)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filterbank(n_fft, sr, n_mels=26):
    """与 Dart 端完全一致的 Mel 滤波器组"""
    half_n = n_fft // 2 + 1
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.array([mel_to_hz(low_mel + i * (high_mel - low_mel) / (n_mels + 1)) for i in range(n_mels + 2)])
    bin_idx = (mel_points / sr * n_fft).astype(int)
    filterbank = np.zeros((n_mels, half_n))
    for m in range(n_mels):
        for k in range(bin_idx[m], min(bin_idx[m + 1], half_n)):
            filterbank[m, k] = (k - bin_idx[m]) / (bin_idx[m + 1] - bin_idx[m])
        for k in range(bin_idx[m + 1], min(bin_idx[m + 2], half_n)):
            filterbank[m, k] = (bin_idx[m + 2] - k) / (bin_idx[m + 2] - bin_idx[m + 1])
    return filterbank


# ====== 与 Dart 端完全一致的特征提取 ======
SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 26
N_MFCC = 13
_HAMMING = hamming_window(N_FFT)
_MEL_BASIS = mel_filterbank(N_FFT, SR, N_MELS)


def extract_features(audio, sr=SR):
    """
    简化 MFCC + 谱特征（与 Dart 端完全一致的算法）
    共 32 维
    """
    if len(audio) < 256:
        audio = np.pad(audio, (0, max(0, 256 - len(audio))))
    audio = audio[:5 * sr]
    n = len(audio)

    frames = []
    for start in range(0, n - N_FFT + 1, HOP):
        frames.append(audio[start:start + N_FFT])
    if not frames:
        frames.append(np.pad(audio, (0, max(0, N_FFT - n)))[:N_FFT])

    # 预先计算 DCT 权重
    dct_w = np.array([[cos(pi * j * (k + 0.5) / N_MELS) * sqrt(2.0 / N_MELS) for k in range(N_MELS)] for j in range(N_MFCC)])

    mfcc_frames = []
    frame_energies = []
    zcr_list = []
    rms_list = []

    for frame in frames:
        windowed = frame * _HAMMING
        spectrum = power_spectrum(windowed)
        # Mel 能量（向量化）
        mel_energy = np.log(np.maximum(spectrum @ _MEL_BASIS.T, 1e-10))
        # DCT-II（向量化）
        mfcc = dct_w @ mel_energy
        mfcc_frames.append(mfcc)

        # 谱特征
        freqs = np.arange(len(spectrum)) * SR / N_FFT
        total_mag = max(np.sum(spectrum), 1e-10)
        centroid = np.sum(freqs * spectrum) / total_mag
        bw = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total_mag)
        frame_energies.append([centroid, bw])

        # ZCR
        sign_changes = np.sum(np.diff(np.signbit(frame).astype(int)) != 0)
        zcr_list.append(sign_changes / len(frame))

        # RMS
        rms_list.append(np.sqrt(np.mean(frame ** 2)))

    mfcc = np.array(mfcc_frames)
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)

    fe = np.array(frame_energies)
    sc_mean = np.mean(fe[:, 0])
    sc_std = np.std(fe[:, 0])
    bw_mean = np.mean(fe[:, 1])
    zcr_mean = np.mean(zcr_list)
    rms_mean = np.mean(rms_list)
    rms_std = np.std(rms_list)

    return np.concatenate([mfcc_mean, mfcc_std, [sc_mean, sc_std, bw_mean, zcr_mean, rms_mean, rms_std]])


def extract_batch(df, desc):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), desc=desc, total=len(df)):
        try:
            # 用 soundfile 读取保持与 librosa 一致
            import soundfile as sf
            audio, sr = sf.read(row["path"])
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
            X.append(extract_features(audio, SR))
            y.append(row["label"])
        except Exception as e:
            tqdm.write(f"  跳过: {e}")
    return np.array(X), y


def export_weights_json(model, scaler, label_map, output_path):
    weights = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        weights[f"layer{i}_weight"] = coef.tolist()
        weights[f"layer{i}_bias"] = intercept.tolist()
    weights["label_map"] = label_map
    weights["feature_dim"] = scaler.mean_.shape[0]

    with open(output_path, "w") as f:
        json.dump(weights, f, ensure_ascii=False)

    # 二进制
    bin_path = output_path.with_suffix(".bin")
    with open(bin_path, "wb") as f:
        f.write(b"PETM003")
        label_json = json.dumps(label_map).encode()
        f.write(struct.pack(">I", len(label_json)))
        f.write(label_json)
        f.write(struct.pack(">I", len(scaler.mean_)))
        scaler.mean_.astype(np.float32).tofile(f)
        scaler.scale_.astype(np.float32).tofile(f)
        f.write(struct.pack(">I", len(model.coefs_)))
        for coef, intercept in zip(model.coefs_, model.intercepts_):
            shape = coef.shape
            f.write(struct.pack(">II", shape[0], shape[1]))
            coef.astype(np.float32).tofile(f)
            intercept.astype(np.float32).tofile(f)

    return weights


def main():
    print("=" * 60)
    print("训练离线分类器（简化 MFCC，Python = Dart 一致）")
    print("=" * 60)

    train_df = pd.read_csv(PROCESSED_DIR / "train_augmented.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    with open(PROCESSED_DIR / "label_map.json") as f:
        label_map = json.load(f)
    id_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    print(f"训练: {len(train_df)}, 验证: {len(val_df)}, 测试: {len(test_df)}, 类别: {num_classes}")

    print("\n提取特征...")
    X_train, y_train = extract_batch(train_df, "训练")
    X_val, y_val = extract_batch(val_df, "验证")
    X_test, y_test = extract_batch(test_df, "测试")

    print(f"特征维度: {X_train.shape[1]}")

    y_train_n = np.array([label_map[l] for l in y_train])
    y_val_n = np.array([label_map[l] for l in y_val])
    y_test_n = np.array([label_map[l] for l in y_test])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("\n训练 MLP...")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500, early_stopping=True,
        validation_fraction=0.15, random_state=42,
    )
    model.fit(X_train_s, y_train_n)

    train_acc = accuracy_score(y_train_n, model.predict(X_train_s))
    val_acc = accuracy_score(y_val_n, model.predict(X_val_s))
    test_acc = accuracy_score(y_test_n, model.predict(X_test_s))

    print(f"训练: {train_acc:.4f}, 验证: {val_acc:.4f}, 测试: {test_acc:.4f}")

    print(f"\n测试集分类报告:")
    print(classification_report(y_test_n, model.predict(X_test_s),
                                target_names=[id_to_label[i] for i in range(num_classes)]))

    print("\n导出模型...")
    json_path = MODELS_DIR / "ondevice_model.json"
    export_weights_json(model, scaler, label_map, json_path)

    # 验证
    print("\n验证导出...")
    with open(json_path) as f:
        loaded = json.load(f)
    W0 = np.array(loaded["layer0_weight"])
    b0 = np.array(loaded["layer0_bias"])
    W1 = np.array(loaded["layer1_weight"])
    b1 = np.array(loaded["layer1_bias"])
    W2 = np.array(loaded["layer2_weight"])
    b2 = np.array(loaded["layer2_bias"])
    sm = np.array(loaded["scaler_mean"])
    ss = np.array(loaded["scaler_scale"])

    def relu(x): return np.maximum(0, x)
    def softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def dart_infer(features):
        x = (features - sm) / ss
        # Dart 端加载时转置了权重，模拟 Dart 的 matMul
        x = relu(x @ W0 + b0)   # (32,) @ (32,64) = (64,)
        x = relu(x @ W1 + b1)   # (64,) @ (64,32) = (32,)
        x = softmax(x @ W2 + b2)  # (32,) @ (32,12) = (12,)
        return np.argmax(x)

    errors = 0
    for i in range(min(100, len(X_test))):
        if dart_infer(X_test[i]) != y_test_n[i]:
            errors += 1
    print(f"  导出验证: {100-errors}/100 匹配")

    js = json_path.stat().st_size / 1024
    bs = json_path.with_suffix(".bin").stat().st_size / 1024
    print(f"  JSON: {js:.1f} KB, 二进制: {bs:.1f} KB")

    print(f"\n{'=' * 60}")
    print("完成!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
