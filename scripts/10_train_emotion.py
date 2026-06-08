"""
分物种猫狗情绪模型 (Valence-Arousal)
- 猫和狗分别训练独立的 valence + arousal 模型
- App 端按宠物物种选择对应模型
- 特征: 32维简化MFCC(与Dart端一致)
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
from math import cos, pi, sqrt, log

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SR, N_FFT, HOP, N_MELS, N_MFCC = 16000, 512, 256, 26, 13

def _hamming(s): return np.array([0.54 - 0.46 * cos(2*pi*n/(s-1)) for n in range(s)])
def _hz_to_mel(hz): return 2595 * log(1 + hz/700) / log(10)
def _mel_to_hz(m): return 700 * (10**(m/2595) - 1)
def _mel_fb():
    half = N_FFT//2+1
    lo, hi = _hz_to_mel(0), _hz_to_mel(SR/2)
    pts = np.array([_mel_to_hz(lo+i*(hi-lo)/(N_MELS+1)) for i in range(N_MELS+2)])
    b = (pts/SR*N_FFT).astype(int)
    fb = np.zeros((N_MELS, half))
    for m in range(N_MELS):
        for k in range(b[m], min(b[m+1], half)): fb[m,k] = (k-b[m])/(b[m+1]-b[m])
        for k in range(b[m+1], min(b[m+2], half)): fb[m,k] = (b[m+2]-k)/(b[m+2]-b[m+1])
    return fb

_HAMMING, _MEL = _hamming(N_FFT), _mel_fb()
_DCT = np.array([[cos(pi*j*(k+0.5)/N_MELS)*sqrt(2.0/N_MELS) for k in range(N_MELS)] for j in range(N_MFCC)])

def _ps(frame):
    fft = np.fft.rfft(frame, n=len(frame))
    return (fft.real**2 + fft.imag**2) / len(frame)

def extract_features(audio, sr=SR):
    """32 维: MFCC mean(13) + std(13) + 谱特征(6)"""
    if len(audio) < 256: audio = np.pad(audio, (0, 256-len(audio)))
    audio = audio[:5*sr]
    n = len(audio)
    frames = [audio[s:s+N_FFT] for s in range(0, n-N_FFT+1, HOP)]
    if not frames: frames = [np.pad(audio, (0, max(0, N_FFT-n)))[:N_FFT]]

    mfcc_f, cs, bs, zs, rs = [], [], [], [], []
    for fr in frames:
        spec = _ps(fr * _HAMMING)
        mfcc_f.append(_DCT @ np.log(np.maximum(spec @ _MEL.T, 1e-10)))
        freqs = np.arange(len(spec)) * SR / N_FFT
        tot = max(np.sum(spec), 1e-10)
        c = np.sum(freqs*spec)/tot
        cs.append(c); bs.append(sqrt(np.sum(((freqs-c)**2)*spec)/tot))
        zs.append(np.sum(np.diff(np.signbit(fr).astype(int)) != 0) / len(fr))
        rs.append(sqrt(np.mean(fr**2)))
    mfcc = np.array(mfcc_f)
    return np.concatenate([np.mean(mfcc,0), np.std(mfcc,0),
                           [np.mean(cs), np.std(cs), np.mean(bs), np.mean(zs), np.mean(rs), np.std(rs)]])

def augment(audio):
    return [audio,
            audio + np.random.randn(len(audio))*0.005,
            audio + np.random.randn(len(audio))*0.012,
            np.pad(audio, (int(len(audio)*0.1), 0))[:len(audio)],
            np.pad(audio[int(len(audio)*0.1):], (0, int(len(audio)*0.1))),
            audio*0.7, np.clip(audio*1.4, -1, 1)]

VAL = {"Negative":0, "Neutral":1, "Positive":2}
ARO = {"Low":0, "Medium":1, "High":2}

def load_cat():
    d = DATA_DIR/"raw"/"CatMeows"/"dataset"
    # brushing→(Pos,Low) food→(Neu,High) isolation→(Neg,Med)
    m = {"B":(2,0), "F":(1,2), "I":(0,1)}
    return [{"path":str(w), "valence":m[w.stem.split("_")[0]][0], "arousal":m[w.stem.split("_")[0]][1]}
            for w in d.glob("*.wav") if w.stem.split("_")[0] in m]

def load_dog():
    base = DATA_DIR/"raw"/"BarkopediaDogEmotion"
    recs = []
    for breed in ["husky","shiba"]:
        csv = base/f"{breed}_train_labels.csv"
        if not csv.exists(): continue
        for _, r in pd.read_csv(csv).iterrows():
            w = base/"train"/breed/f"{r['audio_id']}.wav"
            if w.exists():
                recs.append({"path":str(w), "valence":VAL[r["valence"]], "arousal":ARO[r["arousal"]]})
    return recs

def read_audio(path):
    a, sr = sf.read(path)
    if len(a.shape) > 1: a = a.mean(axis=1)
    if sr != SR:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a

def train_species(records, name):
    np.random.seed(42)
    np.random.shuffle(records)
    n_test = int(len(records)*0.15)
    test_r, train_r = records[:n_test], records[n_test:]

    Xtr, vtr, atr = [], [], []
    for r in tqdm(train_r, desc=f"{name}训练"):
        try:
            a = read_audio(r["path"])
            for aug in augment(a):
                Xtr.append(extract_features(aug)); vtr.append(r["valence"]); atr.append(r["arousal"])
        except Exception as e: tqdm.write(f"skip {e}")

    Xte, vte, ate = [], [], []
    for r in tqdm(test_r, desc=f"{name}测试"):
        try:
            a = read_audio(r["path"])
            Xte.append(extract_features(a)); vte.append(r["valence"]); ate.append(r["arousal"])
        except Exception as e: tqdm.write(f"skip {e}")

    Xtr, Xte = np.array(Xtr), np.array(Xte)
    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)

    out = {"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}
    accs = {}
    for task, ytr, yte in [("valence", vtr, vte), ("arousal", atr, ate)]:
        model = MLPClassifier(hidden_layer_sizes=(96,48), activation="relu", max_iter=1000,
                              early_stopping=True, validation_fraction=0.12, random_state=42, alpha=0.001)
        model.fit(Xtr_s, ytr)
        acc = accuracy_score(yte, model.predict(Xte_s))
        accs[task] = acc
        out[task] = {"weights": [w.tolist() for w in model.coefs_],
                     "biases": [b.tolist() for b in model.intercepts_]}
        print(f"  {name} {task}: {acc:.4f}")

    vp = np.argmax([_forward(Xte_s[i], out["valence"], scaler, raw=True) for i in range(len(Xte_s))], axis=1) if False else None
    return out, accs

def _forward(x, head, scaler, raw=False):
    for i in range(len(head["weights"])-1):
        x = np.maximum(0, x @ np.array(head["weights"][i]) + np.array(head["biases"][i]))
    return x @ np.array(head["weights"][-1]) + np.array(head["biases"][-1])

def main():
    print("="*60 + "\n分物种猫狗情绪模型\n" + "="*60)
    cat, dog = load_cat(), load_dog()
    print(f"猫: {len(cat)} 条, 狗: {len(dog)} 条\n")

    cat_model, cat_acc = train_species(cat, "猫")
    print()
    dog_model, dog_acc = train_species(dog, "狗")

    model_data = {"feature_dim": 32, "cat": cat_model, "dog": dog_model}
    out_path = MODELS_DIR / "emotion_model.json"
    with open(out_path, "w") as f:
        json.dump(model_data, f, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"猫: valence={cat_acc['valence']:.1%} arousal={cat_acc['arousal']:.1%}")
    print(f"狗: valence={dog_acc['valence']:.1%} arousal={dog_acc['arousal']:.1%}")
    print(f"模型: {out_path} ({out_path.stat().st_size/1024:.0f} KB)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
