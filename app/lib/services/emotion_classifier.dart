import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'audio_features.dart';
import '../models/emotion_result.dart';

/// 手机端离线情绪分类器（分物种 Valence-Arousal 双模型）
class EmotionClassifier {
  Map<String, dynamic>? _model;
  bool _loaded = false;
  String? _error;

  bool get isLoaded => _loaded;
  String? get error => _error;

  Future<void> load() async {
    try {
      final jsonStr = await rootBundle.loadString('assets/emotion_model.json');
      _model = jsonDecode(jsonStr) as Map<String, dynamic>;
      _loaded = true;
    } catch (e) {
      _error = '模型加载失败: $e';
    }
  }

  /// 按物种分类（species: 'cat' / 'dog'）
  EmotionResult? predict(List<double> audio, String species) {
    if (!_loaded || _model == null) return null;

    final speciesModel = _model![species] as Map<String, dynamic>?;
    if (speciesModel == null) return null;

    // 峰值归一化：消除录音/下载音频的音量差异
    final peak = audio.map((s) => s.abs()).reduce(max);
    if (peak > 0.01) {
      final scale = 0.95 / peak;
      for (int i = 0; i < audio.length; i++) {
        audio[i] *= scale;
      }
    }

    // 特征提取 + 标准化
    final features = AudioFeatureExtractor.extract(audio);
    final scalerMean = (speciesModel['scaler_mean'] as List).cast<num>().map((e) => e.toDouble()).toList();
    final scalerScale = (speciesModel['scaler_scale'] as List).cast<num>().map((e) => e.toDouble()).toList();
    final x = List<double>.generate(features.length, (i) =>
        (features[i] - scalerMean[i]) / scalerScale[i]);

    final valProbs = _forwardSoftmax(x, speciesModel['valence'] as Map<String, dynamic>);
    final aroProbs = _forwardSoftmax(x, speciesModel['arousal'] as Map<String, dynamic>);

    return EmotionResult(
      valence: _argmax(valProbs),
      arousal: _argmax(aroProbs),
      valenceConfidence: valProbs.reduce(max),
      arousalConfidence: aroProbs.reduce(max),
      valenceProbs: valProbs,
      arousalProbs: aroProbs,
    );
  }

  List<double> _forwardSoftmax(List<double> input, Map<String, dynamic> head) {
    final weights = (head['weights'] as List).map((w) =>
        (w as List).map((row) => (row as List).cast<num>().map((e) => e.toDouble()).toList()).toList()).toList();
    final biases = (head['biases'] as List).map((b) =>
        (b as List).cast<num>().map((e) => e.toDouble()).toList()).toList();

    var x = input;
    for (int layer = 0; layer < weights.length; layer++) {
      final w = weights[layer]; // [in][out]
      final b = biases[layer];
      final outDim = b.length;
      final out = List<double>.filled(outDim, 0.0);
      for (int j = 0; j < outDim; j++) {
        double sum = b[j];
        for (int i = 0; i < x.length; i++) {
          sum += x[i] * w[i][j];
        }
        out[j] = sum;
      }
      // ReLU 除最后一层
      if (layer < weights.length - 1) {
        for (int j = 0; j < outDim; j++) {
          if (out[j] < 0) out[j] = 0;
        }
      }
      x = out;
    }
    return _softmax(x);
  }

  List<double> _softmax(List<double> x) {
    final mx = x.reduce(max);
    final exps = x.map((v) => exp(v - mx)).toList();
    final sum = exps.reduce((a, b) => a + b);
    return exps.map((v) => v / sum).toList();
  }

  int _argmax(List<double> x) {
    int idx = 0;
    for (int i = 1; i < x.length; i++) {
      if (x[i] > x[idx]) idx = i;
    }
    return idx;
  }
}
