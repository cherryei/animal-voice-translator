import 'package:flutter/material.dart';
import '../theme.dart';
import '../models/emotion_result.dart';
import '../models/translation_result.dart';
import '../models/pet_profile.dart';

class ResultScreen extends StatelessWidget {
  final EmotionResult emotion;
  final TranslationResult? translation;
  final String? llmError;
  final PetProfile? pet;
  final String? acousticAnalysis;

  const ResultScreen({
    super.key,
    required this.emotion,
    this.translation,
    this.llmError,
    this.pet,
    this.acousticAnalysis,
  });

  @override
  Widget build(BuildContext context) {
    final color = Color(emotion.colorValue);
    final petName = pet?.name ?? '宝贝';
    final hasTranslation = translation != null && translation!.translation.isNotEmpty;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(gradient: AppTheme.bgGradient),
        child: SafeArea(
          child: Column(
            children: [
              // 顶栏
              Padding(
                padding: const EdgeInsets.fromLTRB(8, 8, 20, 0),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_rounded, color: AppTheme.textDark),
                      onPressed: () => Navigator.pop(context),
                    ),
                    const Text('翻译结果',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppTheme.textDark)),
                  ],
                ),
              ),
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
                  children: [
                    // ========== 心声翻译（主角）==========
                    if (hasTranslation)
                      _heroCard(color, petName)
                    else
                      _fallbackCard(color, petName),

                    // LLM 未配置 / 出错提示
                    if (!hasTranslation && llmError == null)
                      Padding(
                        padding: const EdgeInsets.only(top: 12),
                        child: _infoCard('💡 配置 AI 大模型后，可获得更生动的"心声翻译"和专业建议', Colors.blue),
                      ),
                    if (llmError != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 12),
                        child: _infoCard('AI 翻译失败：$llmError', Colors.orange),
                      ),

                    const SizedBox(height: 16),

                    // ========== 专业解读 & 建议 ==========
                    if (hasTranslation) ...[
                      if (translation!.reasoning.isNotEmpty)
                        _detailCard('🔍 专业解读', translation!.reasoning, Icons.psychology_rounded),
                      if (translation!.suggestion.isNotEmpty) ...[
                        const SizedBox(height: 12),
                        _detailCard('💝 给你的建议', translation!.suggestion, Icons.favorite_rounded),
                      ],
                      const SizedBox(height: 16),
                    ],

                    // ========== 声学特征（折叠）==========
                    if (acousticAnalysis != null && acousticAnalysis!.isNotEmpty)
                      _acousticCard(),
                    if (acousticAnalysis != null && acousticAnalysis!.isNotEmpty)
                      const SizedBox(height: 16),

                    const SizedBox(height: 24),
                    // 再来一次
                    SizedBox(
                      width: double.infinity,
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: AppTheme.primaryGradient,
                          borderRadius: BorderRadius.circular(16),
                          boxShadow: AppTheme.softShadow,
                        ),
                        child: TextButton.icon(
                          style: TextButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            foregroundColor: Colors.white,
                          ),
                          onPressed: () => Navigator.pop(context),
                          icon: const Icon(Icons.refresh_rounded),
                          label: const Text('再录一次', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ========== 主角卡片：心声翻译推测 ==========
  Widget _heroCard(Color color, String petName) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft, end: Alignment.bottomRight,
              colors: [color.withValues(alpha: 0.85), color.withValues(alpha: 0.6)],
            ),
            borderRadius: BorderRadius.circular(24),
            boxShadow: [BoxShadow(color: color.withValues(alpha: 0.3), blurRadius: 20, offset: const Offset(0, 10))],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // 标题：心声翻译推测
              Row(
                children: [
                  const Icon(Icons.format_quote_rounded, color: Colors.white70, size: 18),
                  const SizedBox(width: 6),
                  const Text('心声翻译推测',
                    style: TextStyle(fontSize: 13, color: Colors.white70, fontWeight: FontWeight.w500)),
                ],
              ),
              const SizedBox(height: 16),
              // 宠物名
              Row(
                children: [
                  Text(pet?.species == 'dog' ? '🐶' : '🐱', style: const TextStyle(fontSize: 20)),
                  const SizedBox(width: 8),
                  Text(petName, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.white)),
                ],
              ),
              const SizedBox(height: 16),
              // 心声翻译（大字）
              Text('"${translation!.translation}"',
                style: const TextStyle(fontSize: 22, height: 1.5, fontWeight: FontWeight.w600, color: Colors.white)),
            ],
          ),
        ),
        // 心情标签
        const SizedBox(height: 10),
        Row(
          children: [
            Text(emotion.emoji, style: const TextStyle(fontSize: 20)),
            const SizedBox(width: 6),
            Text('心情：${emotion.emotionLabel}',
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: color)),
          ],
        ),
      ],
    );
  }

  // ========== 降级卡片（无 LLM 时）==========
  Widget _fallbackCard(Color color, String petName) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: AppTheme.cardShadow,
      ),
      child: Column(
        children: [
          Text(emotion.emoji, style: const TextStyle(fontSize: 56)),
          const SizedBox(height: 12),
          Text(emotion.emotionLabel,
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: color)),
          const SizedBox(height: 8),
          Text('配置 AI 大模型后可获得更准确的翻译',
            style: const TextStyle(fontSize: 13, color: AppTheme.textGray)),
        ],
      ),
    );
  }

  // ========== 专业解读/建议卡片 ==========
  Widget _detailCard(String title, String body, IconData icon) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: AppTheme.cardShadow,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Icon(icon, size: 18, color: AppTheme.primary),
            const SizedBox(width: 6),
            Text(title, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: AppTheme.textDark)),
          ]),
          const SizedBox(height: 10),
          Text(body, style: const TextStyle(fontSize: 14, height: 1.6, color: AppTheme.textGray)),
        ],
      ),
    );
  }

  // ========== 声学特征（可展开）==========
  Widget _acousticCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: AppTheme.cardShadow,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            const Icon(Icons.analytics_rounded, size: 18, color: AppTheme.textGray),
            const SizedBox(width: 6),
            const Text('声学特征分析', style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: AppTheme.textGray)),
          ]),
          const SizedBox(height: 10),
          Text(acousticAnalysis!, style: const TextStyle(fontSize: 12, height: 1.5, color: AppTheme.textGray)),
        ],
      ),
    );
  }

  // ========== 提示卡片 ==========
  Widget _infoCard(String text, Color tint) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: tint.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: tint.withValues(alpha: 0.25)),
      ),
      child: Text(text, style: TextStyle(fontSize: 13, height: 1.5, color: tint.withValues(alpha: 0.9))),
    );
  }
}
