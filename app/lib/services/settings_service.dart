import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/llm_config.dart';
import '../models/pet_profile.dart';

/// 本地持久化设置
class SettingsService {
  static const String _llmKey = 'llm_config';
  static const String _petsKey = 'pet_profiles';
  static const String _activePetKey = 'active_pet_id';
  static const String _serverUrlKey = 'server_url';
  static late SharedPreferences _prefs;

  static Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
  }

  // ---- LLM 配置 ----
  static LlmConfig getLlmConfig() {
    final json = _prefs.getString(_llmKey);
    if (json == null) return LlmConfig();
    return LlmConfig.fromJson(jsonDecode(json) as Map<String, dynamic>);
  }

  static Future<void> saveLlmConfig(LlmConfig config) async {
    await _prefs.setString(_llmKey, jsonEncode(config.toJson()));
  }

  // ---- 宠物信息 ----
  static List<PetProfile> getPets() {
    final json = _prefs.getString(_petsKey);
    if (json == null) return [];
    final list = jsonDecode(json) as List;
    return list.map((e) => PetProfile.fromJson(e as Map<String, dynamic>)).toList();
  }

  static Future<void> savePets(List<PetProfile> pets) async {
    await _prefs.setString(_petsKey, jsonEncode(pets.map((p) => p.toJson()).toList()));
  }

  static String? getActivePetId() => _prefs.getString(_activePetKey);

  static Future<void> setActivePetId(String? id) async {
    if (id == null) {
      await _prefs.remove(_activePetKey);
    } else {
      await _prefs.setString(_activePetKey, id);
    }
  }

  // ---- 服务器地址 ----
  // 真机请用电脑的局域网 IP，如 http://192.168.1.100:8000
  // 模拟器用 http://10.0.2.2:8000
  // macOS 桌面用 http://localhost:8000
  static String getServerUrl() => _prefs.getString(_serverUrlKey) ?? 'http://localhost:8000';

  static Future<void> setServerUrl(String url) async {
    await _prefs.setString(_serverUrlKey, url);
  }
}
