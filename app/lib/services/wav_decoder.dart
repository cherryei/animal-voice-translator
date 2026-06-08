import 'dart:io';
import 'dart:typed_data';

/// 解码结果：单声道 16kHz 浮点采样
class DecodedAudio {
  final List<double> samples; // [-1, 1] 单声道
  final int sampleRate;       // 固定 16000
  DecodedAudio(this.samples, this.sampleRate);
}

/// 健壮的 WAV 解码器：正确解析头部（任意采样率/声道/位深），统一重采样到 16kHz 单声道。
///
/// 之前的实现假设文件一定是 16kHz、数据从第 44 字节开始——这对手机录音勉强成立，
/// 但网上下载的音频通常是 44.1k/48kHz，被当成 16kHz 读取后所有频率都会错位，
/// 特征变成噪声，模型只能输出最常见的类别（闷闷不乐）。
class WavDecoder {
  static const int targetRate = 16000;

  /// 返回 16kHz 单声道采样；无法解析（如 MP3）时返回 null
  static DecodedAudio? decodeFile(String path) {
    final bytes = File(path).readAsBytesSync();
    return decodeBytes(bytes);
  }

  static DecodedAudio? decodeBytes(Uint8List bytes) {
    if (bytes.length < 44) return null;
    // RIFF....WAVE
    if (_str(bytes, 0, 4) != 'RIFF' || _str(bytes, 8, 4) != 'WAVE') {
      return null; // 不是 WAV（可能是 MP3/M4A 等压缩格式）
    }

    final bd = ByteData.sublistView(bytes);
    int fmtFormat = 1, channels = 1, sampleRate = targetRate, bits = 16;
    int dataOffset = -1, dataLen = 0;

    // 从第 12 字节开始遍历 chunk
    int pos = 12;
    while (pos + 8 <= bytes.length) {
      final id = _str(bytes, pos, 4);
      final size = bd.getUint32(pos + 4, Endian.little);
      final body = pos + 8;
      if (id == 'fmt ') {
        fmtFormat = bd.getUint16(body, Endian.little);
        channels = bd.getUint16(body + 2, Endian.little);
        sampleRate = bd.getUint32(body + 4, Endian.little);
        bits = bd.getUint16(body + 14, Endian.little);
      } else if (id == 'data') {
        dataOffset = body;
        dataLen = size;
        // data 通常是最后一个有效块，但继续遍历以防万一
      }
      // chunk 大小为奇数时有 1 字节填充
      pos = body + size + (size.isOdd ? 1 : 0);
    }

    if (dataOffset < 0 || channels < 1) return null;
    if (dataOffset + dataLen > bytes.length) {
      dataLen = bytes.length - dataOffset; // 容错：头部声明的长度超出实际
    }

    // 解码为单声道浮点（合并多声道）
    final mono = <double>[];
    final bytesPerSample = bits ~/ 8;
    final frameSize = bytesPerSample * channels;
    if (frameSize == 0) return null;

    for (int i = dataOffset; i + frameSize <= dataOffset + dataLen; i += frameSize) {
      double sum = 0;
      for (int c = 0; c < channels; c++) {
        final o = i + c * bytesPerSample;
        sum += _sampleAt(bd, o, bits, fmtFormat);
      }
      mono.add(sum / channels);
    }

    if (mono.isEmpty) return null;

    // 重采样到 16kHz（线性插值）
    final resampled = (sampleRate == targetRate)
        ? mono
        : _resampleLinear(mono, sampleRate, targetRate);

    return DecodedAudio(resampled, targetRate);
  }

  static double _sampleAt(ByteData bd, int o, int bits, int format) {
    if (format == 3) {
      // IEEE float
      if (bits == 32) return bd.getFloat32(o, Endian.little);
      if (bits == 64) return bd.getFloat64(o, Endian.little);
    }
    switch (bits) {
      case 8:
        // 8-bit PCM 是无符号 (0..255)，中心 128
        return (bd.getUint8(o) - 128) / 128.0;
      case 16:
        return bd.getInt16(o, Endian.little) / 32768.0;
      case 24:
        final b0 = bd.getUint8(o), b1 = bd.getUint8(o + 1), b2 = bd.getUint8(o + 2);
        int v = b0 | (b1 << 8) | (b2 << 16);
        if (v & 0x800000 != 0) v -= 0x1000000; // 符号扩展
        return v / 8388608.0;
      case 32:
        return bd.getInt32(o, Endian.little) / 2147483648.0;
      default:
        return 0;
    }
  }

  static List<double> _resampleLinear(List<double> input, int srcRate, int dstRate) {
    final ratio = dstRate / srcRate;
    final outLen = (input.length * ratio).floor();
    final out = List<double>.filled(outLen, 0.0);
    for (int i = 0; i < outLen; i++) {
      final srcPos = i / ratio;
      final i0 = srcPos.floor();
      final i1 = (i0 + 1 < input.length) ? i0 + 1 : i0;
      final frac = srcPos - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  static String _str(Uint8List b, int off, int len) {
    return String.fromCharCodes(b.sublist(off, off + len));
  }
}
