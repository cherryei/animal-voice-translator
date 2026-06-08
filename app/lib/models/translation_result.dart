class ClassifierResult {
  final String prediction;
  final double confidence;
  final Map<String, double> allProbabilities;

  ClassifierResult({
    required this.prediction,
    required this.confidence,
    required this.allProbabilities,
  });

  factory ClassifierResult.fromJson(Map<String, dynamic> json) {
    return ClassifierResult(
      prediction: json['prediction'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      allProbabilities: (json['all_probabilities'] as Map<String, dynamic>)
          .map((k, v) => MapEntry(k, (v as num).toDouble())),
    );
  }

  List<MapEntry<String, double>> get top3 {
    final sorted = allProbabilities.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return sorted.take(3).toList();
  }
}

class TranslationResult {
  final String translation;
  final String emotion;
  final String confidenceLevel;
  final String reasoning;
  final String suggestion;

  TranslationResult({
    required this.translation,
    required this.emotion,
    required this.confidenceLevel,
    required this.reasoning,
    required this.suggestion,
  });

  factory TranslationResult.fromJson(Map<String, dynamic> json) {
    return TranslationResult(
      translation: json['translation'] as String? ?? '',
      emotion: json['emotion'] as String? ?? '',
      confidenceLevel: json['confidence_level'] as String? ?? '',
      reasoning: json['reasoning'] as String? ?? '',
      suggestion: json['suggestion'] as String? ?? '',
    );
  }

  bool get isEmpty => translation.isEmpty;
}

class AnalysisResult {
  final ClassifierResult classifier;
  final TranslationResult? translation;
  final String audioPath;
  final DateTime timestamp;
  final String? petName;
  final String? context;

  AnalysisResult({
    required this.classifier,
    this.translation,
    required this.audioPath,
    required this.timestamp,
    this.petName,
    this.context,
  });
}
