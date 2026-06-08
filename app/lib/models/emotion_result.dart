/// 情绪结果：基于 Valence(效价) × Arousal(唤醒度) 的二维情感模型
class EmotionResult {
  final int valence; // 0=消极 1=中性 2=积极
  final int arousal; // 0=平静 1=中等 2=激动
  final double valenceConfidence;
  final double arousalConfidence;
  final List<double> valenceProbs;
  final List<double> arousalProbs;

  EmotionResult({
    required this.valence,
    required this.arousal,
    required this.valenceConfidence,
    required this.arousalConfidence,
    required this.valenceProbs,
    required this.arousalProbs,
  });

  /// 综合置信度
  double get confidence => (valenceConfidence + arousalConfidence) / 2;

  /// 情绪名称（9 宫格组合）
  String get emotionLabel {
    const labels = [
      // arousal:  Low      Medium    High
      ['低落难过', '闷闷不乐', '焦虑烦躁'], // valence Negative
      ['慵懒放松', '平静好奇', '警觉关注'], // valence Neutral
      ['满足惬意', '友好开心', '兴奋激动'], // valence Positive
    ];
    return labels[valence][arousal];
  }

  /// 情绪 emoji
  String get emoji {
    const emojis = [
      ['😔', '😟', '😾'],
      ['😴', '🙂', '👀'],
      ['😌', '😺', '🤩'],
    ];
    return emojis[valence][arousal];
  }

  /// 情绪主色调
  int get colorValue {
    const colors = [
      [0xFF7986CB, 0xFF5C6BC0, 0xFFEF5350], // 消极
      [0xFF9E9E9E, 0xFF66BB6A, 0xFFFFB74D], // 中性
      [0xFF26A69A, 0xFF66BB6A, 0xFFFFCA28], // 积极
    ];
    return colors[valence][arousal];
  }

  String get valenceLabel => ['消极', '中性', '积极'][valence];
  String get arousalLabel => ['平静', '中等', '激动'][arousal];
}
