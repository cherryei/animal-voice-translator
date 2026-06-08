import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/llm_config.dart';
import '../models/pet_profile.dart';
import '../models/translation_result.dart';
import '../models/emotion_result.dart';

/// 直接调用 LLM API（OpenAI 兼容）进行情绪翻译
class LlmService {
  /// 查询可用模型列表（直接传入 url 和 key，不依赖已保存配置）
  static Future<List<String>> listModels(String baseUrl, String apiKey) async {
    String url = baseUrl.replaceAll(RegExp(r'/+$'), '');
    if (url.endsWith('/chat/completions')) {
      url = url.replaceAll('/chat/completions', '');
    }
    url = '$url/models';

    final response = await http.get(
      Uri.parse(url),
      headers: {'Authorization': 'Bearer $apiKey'},
    ).timeout(const Duration(seconds: 15));

    if (response.statusCode != 200) {
      throw Exception('HTTP ${response.statusCode}: ${response.body}');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    final allModels = (data['data'] as List?) ?? [];
    return allModels
        .map((m) => m is Map ? m['id'] as String? : null)
        .whereType<String>()
        .where((id) => !id.contains('tts') && !id.contains('embed'))
        .toList();
  }

  /// 将情绪结果翻译成自然语言
  /// [acousticAnalysis] 是从 AcousticAnalyzer 生成的详细声学特征文本
  static Future<TranslationResult> translate({
    required EmotionResult emotion,
    required LlmConfig config,
    PetProfile? pet,
    String context = '',
    String acousticAnalysis = '',
  }) async {
    final species = pet?.species == 'dog' ? '狗' : '猫';
    final name = pet?.name ?? '宝贝';
    final breed = pet?.breed ?? '';
    final age = pet?.age;

    final systemPrompt = '你是一位资深的宠物行为学家和动物声音翻译专家，拥有20年$species行为研究经验。\n'
        '你的任务是根据$species叫声的声学特征分析它的情绪，并用第一人称翻译它的"心声"。\n\n'
        '分析要点：\n'
        '- 高音调+快速重复 → 通常是兴奋、求食、呼唤同伴\n'
        '- 低音调+长音 → 通常是满足、领地宣示、警告\n'
        '- 音高突然上升 → 警觉、惊讶、疼痛\n'
        '- 音高逐渐下降 → 放松、满足、无聊\n'
        '- 高频噪音/嘶嘶声 → 恐惧、防御、攻击\n'
        '- 低频振动/咕噜 → 满足、舒适、撒娇\n'
        '- 短促高频重复 → 焦急、催促、不耐烦\n'
        '- 长而低沉的嚎叫 → 孤独、呼唤同伴、发情\n\n'
        '输出要求：\n'
        '1. translation：用第一人称、口语化、有情感，像${species}在说话，要生动有趣\n'
        '2. emotion：2-4个字的情绪词（如：兴奋期待、慵懒满足、焦虑不安、孤独思念、好奇警觉、愤怒警告、恐惧防御、撒娇卖萌）\n'
        '3. confidence_level：根据声学特征的明确程度判断高/中/低\n'
        '4. reasoning：结合声学特征给出专业解读，说明为什么判断为这种情绪\n'
        '5. suggestion：给主人的建议\n\n'
        '严格输出JSON格式：{"emotion":"情绪词","translation":"第一人称心声","confidence_level":"高/中/低","reasoning":"专业解读","suggestion":"给主人的建议"}';

    final buf = StringBuffer();
    buf.writeln('请分析以下${name}（$species）的叫声：\n');

    if (acousticAnalysis.isNotEmpty) {
      buf.writeln('【声学特征分析】');
      buf.writeln(acousticAnalysis);
      buf.writeln();
    }

    buf.writeln('【宠物信息】');
    buf.writeln('- 名字: $name');
    buf.writeln('- 物种: $species');
    if (breed.isNotEmpty) buf.writeln('- 品种: $breed');
    if (age != null) buf.writeln('- 年龄: $age岁');
    if (context.isNotEmpty) buf.writeln('- 场景: $context');
    buf.writeln();
    buf.writeln('请结合声学特征分析情绪，输出JSON，全部中文。');

    String url = config.baseUrl.replaceAll(RegExp(r'/+$'), '');
    if (!url.endsWith('/chat/completions')) {
      url += '/chat/completions';
    }

    final payload = {
      'model': config.model,
      'messages': [
        {'role': 'system', 'content': systemPrompt},
        {'role': 'user', 'content': buf.toString()},
      ],
      'max_tokens': 4096,
      'temperature': 0.7,
    };

    final response = await http.post(
      Uri.parse(url),
      headers: {
        'Authorization': 'Bearer ${config.apiKey}',
        'Content-Type': 'application/json',
      },
      body: jsonEncode(payload),
    ).timeout(const Duration(seconds: 120));

    if (response.statusCode != 200) {
      throw Exception('API ${response.statusCode}: ${response.body}');
    }

    final result = jsonDecode(response.body) as Map<String, dynamic>;
    final content = result['choices']?[0]?['message']?['content'] as String? ?? '';

    try {
      return TranslationResult.fromJson(jsonDecode(content) as Map<String, dynamic>);
    } catch (_) {
      final match = RegExp(r'\{.*\}', dotAll: true).firstMatch(content);
      if (match != null) {
        try {
          return TranslationResult.fromJson(jsonDecode(match.group(0)!) as Map<String, dynamic>);
        } catch (_) {}
      }
      return TranslationResult(
        translation: content.isEmpty ? '（翻译失败）' : content,
        emotion: '', confidenceLevel: '', reasoning: '', suggestion: '',
      );
    }
  }
}
