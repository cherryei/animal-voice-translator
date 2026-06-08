import 'dart:math';

/// 从原始音频中提取丰富的声学特征描述，直接发给 LLM 做情绪分析。
///
/// 之前的方案是用 440 个样本训练的小分类器做情绪判断，只能区分 3 种场景（刷毛/食物/隔离），
/// 母猫叫小猫、领地叫、发情叫等完全不在训练分布内，导致所有叫声都输出同一个结果。
/// 现在改为提取详细声学特征，让 LLM 利用其对动物声音的大量知识来判断。
class AcousticAnalyzer {
  static const int sr = 16000;

  /// 提取声学特征并格式化为中文描述文本
  static String analyze(List<double> audio) {
    if (audio.length < sr ~/ 10) return '音频过短，无法分析';

    final dur = audio.length / sr;
    final rms = _rms(audio);
    final peak = audio.map((s) => s.abs()).reduce(max);

    // === 频谱特征 ===
    final frames = _frame(audio, 512, 256);
    final centroids = <double>[];
    final flatnesses = <double>[];
    final zcrs = <double>[];
    final rmsFrames = <double>[];

    for (final fr in frames) {
      final spec = _powerSpectrum(fr);
      centroids.add(_spectralCentroid(spec));
      flatnesses.add(_spectralFlatness(spec));
      zcrs.add(_zcr(fr));
      rmsFrames.add(_rms(fr));
    }

    // === 基频（音高）===
    final pitches = _detectPitch(audio);
    final validPitches = pitches.where((p) => p > 50 && p < 2000).toList();

    // === 能量包络 — 检测突发/渐变 ===
    final onsetCount = _countOnsets(rmsFrames);

    // === 格式化 ===
    return _formatDescription(
      dur: dur, rms: rms, peak: peak,
      centroids: centroids, flatnesses: flatnesses, zcrs: zcrs,
      rmsFrames: rmsFrames, pitches: validPitches, onsetCount: onsetCount,
    );
  }

  // ============ 格式化为中文描述 ============
  static String _formatDescription({
    required double dur, required double rms, required double peak,
    required List<double> centroids, required List<double> flatnesses,
    required List<double> zcrs, required List<double> rmsFrames,
    required List<double> pitches, required int onsetCount,
  }) {
    final b = StringBuffer();

    // 时长
    b.writeln('【时长】${dur.toStringAsFixed(1)}秒');

    // 音量
    final volLabel = rms > 0.08 ? '大声' : (rms > 0.03 ? '中等音量' : '轻声');
    final peakLabel = peak > 0.5 ? '（有明显的强音段）' : '';
    b.writeln('【音量】$volLabel，均方根能量 ${rms.toStringAsFixed(3)}$peakLabel');

    // 音高
    if (pitches.isNotEmpty) {
      final meanP = pitches.reduce((a, b) => a + b) / pitches.length;
      final minP = pitches.reduce(min);
      final maxP = pitches.reduce(max);
      final stdP = _std(pitches);
      final range = maxP - minP;

      String pitchDesc;
      if (meanP > 800) {
        pitchDesc = '高音调';
      } else if (meanP > 400) {
        pitchDesc = '中等音调';
      } else {
        pitchDesc = '低音调';
      }

      String variationDesc;
      if (stdP > 150 || range > 500) {
        variationDesc = '音高变化大（忽高忽低）';
      } else if (stdP > 60) {
        variationDesc = '音高有明显起伏';
      } else {
        variationDesc = '音高较平稳';
      }

      // 趋势
      final trend = _pitchTrend(pitches);
      b.writeln('【音高】平均 ${meanP.round()}Hz（$pitchDesc），范围 ${minP.round()}-${maxP.round()}Hz，$variationDesc$trend');
    } else {
      b.writeln('【音高】未检测到明显基频（可能是咕噜声或噪音）');
    }

    // 音色
    final meanCentroid = centroids.isEmpty ? 0.0 : centroids.reduce((a, b) => a + b) / centroids.length;
    final meanFlatness = flatnesses.isEmpty ? 0.0 : flatnesses.reduce((a, b) => a + b) / flatnesses.length;
    String timbreDesc;
    if (meanFlatness > 0.3) {
      timbreDesc = '噪音性强（嘶嘶声、气流声）';
    } else if (meanCentroid > 3000) {
      timbreDesc = '音色明亮尖锐';
    } else if (meanCentroid > 1500) {
      timbreDesc = '音色中等亮度';
    } else {
      timbreDesc = '音色低沉浑厚';
    }
    b.writeln('【音色】$timbreDesc（频谱质心 ${meanCentroid.round()}Hz）');

    // 能量动态
    if (rmsFrames.isNotEmpty) {
      final rmsStd = _std(rmsFrames);
      final rmsMax = rmsFrames.reduce(max);
      final attackRatio = rmsFrames.isEmpty ? 0.0 : rmsMax / (rmsFrames.reduce((a, b) => a + b) / rmsFrames.length + 0.001);

      String dynDesc;
      if (attackRatio > 4) {
        dynDesc = '突发性强（突然大声然后减弱）';
      } else if (rmsStd > 0.04) {
        dynDesc = '音量变化明显';
      } else if (rmsStd > 0.015) {
        dynDesc = '音量略有起伏';
      } else {
        dynDesc = '音量较平稳';
      }
      b.writeln('【动态】$dynDesc');
    }

    // 节奏
    if (onsetCount > 0) {
      final rate = onsetCount / dur;
      String rhythmDesc;
      if (rate > 3) {
        rhythmDesc = '频繁重复（每秒${rate.toStringAsFixed(1)}次）';
      } else if (rate > 1) {
        rhythmDesc = '有节奏地重复';
      } else {
        rhythmDesc = '单次发声';
      }
      b.writeln('【节奏】$rhythmDesc，检测到 ${onsetCount} 个声音片段');
    }

    return b.toString();
  }

  static String _pitchTrend(List<double> pitches) {
    if (pitches.length < 5) return '';
    final half = pitches.length ~/ 2;
    final first = pitches.sublist(0, half).reduce((a, b) => a + b) / half;
    final second = pitches.sublist(half).reduce((a, b) => a + b) / (pitches.length - half);
    final diff = second - first;
    if (diff > 50) return '，整体音高上升';
    if (diff < -50) return '，整体音高下降';
    return '';
  }

  // ============ 信号处理 ============
  static List<List<double>> _frame(List<double> sig, int win, int hop) {
    final frames = <List<double>>[];
    for (int i = 0; i + win <= sig.length; i += hop) {
      frames.add(sig.sublist(i, i + win));
    }
    if (frames.isEmpty) {
      final padded = [...sig, ...List.filled(512 - sig.length, 0.0)];
      frames.add(padded.sublist(0, 512));
    }
    return frames;
  }

  static List<double> _powerSpectrum(List<double> frame) {
    final n = frame.length;
    final re = List<double>.from(frame);
    final im = List<double>.filled(n, 0.0);
    _fft(re, im);
    final half = n ~/ 2 + 1;
    return List<double>.generate(half, (i) => (re[i] * re[i] + im[i] * im[i]) / n);
  }

  static double _spectralCentroid(List<double> spec) {
    double num = 0, den = 0;
    for (int i = 0; i < spec.length; i++) {
      num += i * sr / (2 * (spec.length - 1)) * spec[i];
      den += spec[i];
    }
    return den > 0 ? num / den : 0;
  }

  static double _spectralFlatness(List<double> spec) {
    double logSum = 0, linSum = 0;
    int count = 0;
    for (final s in spec) {
      if (s > 1e-12) { logSum += log(s); linSum += s; count++; }
    }
    if (count == 0) return 0;
    return exp(logSum / count) / (linSum / count + 1e-12);
  }

  static double _zcr(List<double> frame) {
    int count = 0;
    for (int i = 1; i < frame.length; i++) {
      if ((frame[i] >= 0) != (frame[i - 1] >= 0)) count++;
    }
    return count / frame.length;
  }

  static double _rms(List<double> s) {
    if (s.isEmpty) return 0;
    return sqrt(s.map((x) => x * x).reduce((a, b) => a + b) / s.length);
  }

  static double _std(List<double> v) {
    if (v.length < 2) return 0;
    final m = v.reduce((a, b) => a + b) / v.length;
    return sqrt(v.map((x) => (x - m) * (x - m)).reduce((a, b) => a + b) / v.length);
  }

  // === 基频检测（自相关法）===
  static List<double> _detectPitch(List<double> audio) {
    const win = 1024, hop = 512;
    const minLag = 40, maxLag = 500; // 16000/40=400Hz, 16000/500=32Hz
    final pitches = <double>[];
    for (int i = 0; i + win <= audio.length; i += hop) {
      final frame = audio.sublist(i, i + win);
      final energy = frame.map((x) => x * x).reduce((a, b) => a + b) / frame.length;
      if (energy < 1e-6) continue; // 静音帧
      double bestCorr = 0; int bestLag = 0;
      for (int lag = minLag; lag <= maxLag && lag < win; lag++) {
        double corr = 0, e1 = 0, e2 = 0;
        for (int j = 0; j < win - lag; j++) {
          corr += frame[j] * frame[j + lag];
          e1 += frame[j] * frame[j];
          e2 += frame[j + lag] * frame[j + lag];
        }
        final norm = sqrt(e1 * e2);
        if (norm > 0) corr /= norm;
        if (corr > bestCorr) { bestCorr = corr; bestLag = lag; }
      }
      if (bestCorr > 0.3 && bestLag > 0) {
        pitches.add(sr / bestLag);
      }
    }
    return pitches;
  }

  // === 起音检测 ===
  static int _countOnsets(List<double> rmsFrames) {
    if (rmsFrames.length < 3) return rmsFrames.isEmpty ? 0 : 1;
    final mean = rmsFrames.reduce((a, b) => a + b) / rmsFrames.length;
    final threshold = mean * 1.8;
    int onsets = 0;
    bool wasBelow = rmsFrames[0] < threshold;
    for (int i = 1; i < rmsFrames.length; i++) {
      final isAbove = rmsFrames[i] >= threshold;
      if (wasBelow && isAbove) onsets++;
      wasBelow = !isAbove;
    }
    return onsets;
  }

  // === FFT (Cooley-Tukey) ===
  static void _fft(List<double> re, List<double> im) {
    final n = re.length;
    if (n <= 1) return;
    // bit-reversal
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
      if (i < j) {
        var t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
      int k = n >> 1;
      while (k <= j) { j -= k; k >>= 1; }
      j += k;
    }
    for (int len = 2; len <= n; len <<= 1) {
      final ang = -2 * pi / len;
      final wr = cos(ang), wi = sin(ang);
      for (int i = 0; i < n; i += len) {
        double cr = 1, ci = 0;
        for (int jj = 0; jj < len ~/ 2; jj++) {
          final ur = re[i + jj], ui = im[i + jj];
          final vr = re[i + jj + len ~/ 2] * cr - im[i + jj + len ~/ 2] * ci;
          final vi = re[i + jj + len ~/ 2] * ci + im[i + jj + len ~/ 2] * cr;
          re[i + jj] = ur + vr; im[i + jj] = ui + vi;
          re[i + jj + len ~/ 2] = ur - vr; im[i + jj + len ~/ 2] = ui - vi;
          final nr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr; cr = nr;
        }
      }
    }
  }
}
