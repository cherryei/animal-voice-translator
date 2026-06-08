/// LLM API 配置
class LlmConfig {
  String baseUrl;
  String apiKey;
  String model;

  LlmConfig({
    this.baseUrl = '',
    this.apiKey = '',
    this.model = 'gpt-4o-mini',
  });

  Map<String, dynamic> toJson() => {
    'base_url': baseUrl,
    'api_key': apiKey,
    'model': model,
  };

  factory LlmConfig.fromJson(Map<String, dynamic> json) => LlmConfig(
    baseUrl: json['base_url'] as String? ?? '',
    apiKey: json['api_key'] as String? ?? '',
    model: json['model'] as String? ?? 'gpt-4o-mini',
  );

  bool get isValid => baseUrl.isNotEmpty && apiKey.isNotEmpty;
}
