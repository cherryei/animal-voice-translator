import 'dart:math';

/// 简化 MFCC + 谱特征（与 Python 训练端的算法完全一致）
/// 共 32 维：13 MFCC mean + 13 MFCC std + 6 谱特征
class AudioFeatureExtractor {
  static const int sampleRate = 16000;
  static const int nFft = 512;
  static const int hopLength = 256;
  static const int nMels = 26;
  static const int nMfcc = 13;

  /// 提取 32 维特征向量
  static List<double> extract(List<double> audio) {
    if (audio.length < 256) {
      audio = [...audio, ...List.filled(256 - audio.length, 0.0)];
    }
    final maxLen = 5 * sampleRate;
    if (audio.length > maxLen) {
      audio = audio.sublist(0, maxLen);
    }

    // 分帧
    final frames = <List<double>>[];
    for (int start = 0; start + nFft <= audio.length; start += hopLength) {
      frames.add(audio.sublist(start, start + nFft));
    }
    if (frames.isEmpty) {
      final padded = List<double>.from(audio);
      while (padded.length < nFft) padded.add(0);
      frames.add(padded.sublist(0, nFft));
    }

    // 汉明窗
    final hamming = List<double>.generate(nFft, (n) =>
        0.54 - 0.46 * cos(2 * pi * n / (nFft - 1)));

    // Mel 滤波器组
    final melBasis = _melFilterbank();

    // DCT-II 权重 (13 x 26)
    final dctW = List.generate(nMfcc, (j) => List.generate(nMels, (k) =>
        cos(pi * j * (k + 0.5) / nMels) * sqrt(2.0 / nMels)));

    final mfccFrames = <List<double>>[];
    final frameEnergies = <List<double>>[];
    final zcrList = <double>[];
    final rmsList = <double>[];

    for (final frame in frames) {
      // 加窗
      final windowed = List<double>.generate(nFft, (i) => frame[i] * hamming[i]);
      // 功率谱
      final spectrum = _powerSpectrum(windowed);

      // Mel 能量 (向量化)
      final melEnergy = List<double>.generate(nMels, (m) {
        double e = 0;
        for (int k = 0; k < spectrum.length; k++) {
          e += spectrum[k] * melBasis[m][k];
        }
        return log(max(e, 1e-10));
      });

      // DCT-II (向量化)
      final mfcc = List<double>.generate(nMfcc, (j) {
        double s = 0;
        for (int k = 0; k < nMels; k++) s += dctW[j][k] * melEnergy[k];
        return s;
      });
      mfccFrames.add(mfcc);

      // 谱特征: centroid, bandwidth
      double totalMag = spectrum.reduce((a, b) => a + b);
      if (totalMag < 1e-10) totalMag = 1e-10;
      double weightedSum = 0;
      for (int i = 0; i < spectrum.length; i++) {
        weightedSum += i * sampleRate / nFft * spectrum[i];
      }
      final centroid = weightedSum / totalMag;

      double bwSum = 0;
      for (int i = 0; i < spectrum.length; i++) {
        final freq = i * sampleRate / nFft;
        bwSum += (freq - centroid) * (freq - centroid) * spectrum[i];
      }
      final bw = sqrt(bwSum / totalMag);
      frameEnergies.add([centroid, bw]);

      // ZCR
      int signChanges = 0;
      for (int i = 1; i < frame.length; i++) {
        if ((frame[i] >= 0 && frame[i - 1] < 0) ||
            (frame[i] < 0 && frame[i - 1] >= 0)) {
          signChanges++;
        }
      }
      zcrList.add(signChanges / frame.length);

      // RMS
      double rmsSum = 0;
      for (final s in frame) rmsSum += s * s;
      rmsList.add(sqrt(rmsSum / frame.length));
    }

    // 聚合 MFCC
    final mfccMean = _meanEachCol(mfccFrames);
    final mfccStd = _stdEachCol(mfccFrames);

    // 聚合谱特征
    final centroids = frameEnergies.map((e) => e[0]).toList();
    final bws = frameEnergies.map((e) => e[1]).toList();
    final scMean = _mean(centroids);
    final scStd = _std(centroids);
    final bwMean = _mean(bws);
    final zcrMean = _mean(zcrList);
    final rmsMean = _mean(rmsList);
    final rmsStd = _std(rmsList);

    return [
      ...mfccMean,
      ...mfccStd,
      scMean, scStd, bwMean, zcrMean, rmsMean, rmsStd,
    ];
  }

  // ====== FFT (Cooley-Tukey) ======
  static void _fft(List<Complex> data) {
    final n = data.length;
    if (n <= 1) return;
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
      if (i < j) {
        final tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
      }
      int k = n >> 1;
      while (k <= j) { j -= k; k >>= 1; }
      j += k;
    }
    for (int len = 2; len <= n; len <<= 1) {
      final ang = -2 * pi / len;
      final wlen = Complex(cos(ang), sin(ang));
      for (int i = 0; i < n; i += len) {
        var w = Complex(1, 0);
        for (int j = 0; j < len ~/ 2; j++) {
          final u = data[i + j];
          final v = data[i + j + len ~/ 2] * w;
          data[i + j] = u + v;
          data[i + j + len ~/ 2] = u - v;
          w = w * wlen;
        }
      }
    }
  }

  // ====== 功率谱 ======
  static List<double> _powerSpectrum(List<double> frame) {
    final n = frame.length;
    final complex = List<Complex>.generate(n, (i) => Complex(frame[i], 0));
    _fft(complex);
    final halfN = n ~/ 2 + 1;
    return List<double>.generate(halfN, (i) {
      return (complex[i].real * complex[i].real +
              complex[i].imag * complex[i].imag) / n;
    });
  }

  // ====== Mel 滤波器组 ======
  static double _hzToMel(double hz) => 2595 * log(1 + hz / 700) / log(10);
  static double _melToHz(double mel) => 700 * (pow(10, mel / 2595) - 1);

  static List<List<double>> _melFilterbank() {
    final halfN = nFft ~/ 2 + 1;
    final lowMel = _hzToMel(0);
    final highMel = _hzToMel(sampleRate / 2.0);
    final melPoints = List<double>.generate(nMels + 2, (i) {
      return _melToHz(lowMel + i * (highMel - lowMel) / (nMels + 1));
    });
    final bin = melPoints.map((m) => (m / sampleRate * nFft).round()).toList();
    return List.generate(nMels, (m) {
      final row = List.filled(halfN, 0.0);
      for (int k = bin[m]; k < bin[m + 1] && k < halfN; k++) {
        row[k] = (k - bin[m]) / (bin[m + 1] - bin[m]);
      }
      for (int k = bin[m + 1]; k < bin[m + 2] && k < halfN; k++) {
        row[k] = (bin[m + 2] - k) / (bin[m + 2] - bin[m + 1]);
      }
      return row;
    });
  }

  // ====== 统计辅助 ======
  static double _mean(List<double> v) {
    if (v.isEmpty) return 0;
    return v.reduce((a, b) => a + b) / v.length;
  }

  static double _std(List<double> v) {
    if (v.isEmpty) return 0;
    final m = _mean(v);
    return sqrt(v.map((x) => (x - m) * (x - m)).reduce((a, b) => a + b) / v.length);
  }

  static List<double> _meanEachCol(List<List<double>> matrix) {
    if (matrix.isEmpty || matrix[0].isEmpty) return [];
    final cols = matrix[0].length;
    return List<double>.generate(cols, (j) {
      double s = 0;
      for (int i = 0; i < matrix.length; i++) s += matrix[i][j];
      return s / matrix.length;
    });
  }

  static List<double> _stdEachCol(List<List<double>> matrix) {
    if (matrix.isEmpty || matrix[0].isEmpty) return [];
    final means = _meanEachCol(matrix);
    final cols = matrix[0].length;
    return List<double>.generate(cols, (j) {
      double s = 0;
      for (int i = 0; i < matrix.length; i++) {
        s += (matrix[i][j] - means[j]) * (matrix[i][j] - means[j]);
      }
      return sqrt(s / matrix.length);
    });
  }
}

class Complex {
  final double real;
  final double imag;
  Complex(this.real, this.imag);
  Complex operator +(Complex other) => Complex(real + other.real, imag + other.imag);
  Complex operator -(Complex other) => Complex(real - other.real, imag - other.imag);
  Complex operator *(Complex other) => Complex(
    real * other.real - imag * other.imag,
    real * other.imag + imag * other.real,
  );
}
