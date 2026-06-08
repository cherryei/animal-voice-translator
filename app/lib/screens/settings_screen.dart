import 'package:flutter/material.dart';
import 'package:uuid/uuid.dart';
import '../theme.dart';
import '../models/llm_config.dart';
import '../models/pet_profile.dart';
import '../services/settings_service.dart';
import '../services/llm_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _urlCtrl = TextEditingController();
  final _keyCtrl = TextEditingController();
  String? _selectedModel;
  List<String> _models = [];
  bool _loadingModels = false;
  String? _modelError;
  bool _obscureKey = true;

  List<PetProfile> _pets = [];
  String? _activePetId;

  @override
  void initState() {
    super.initState();
    final cfg = SettingsService.getLlmConfig();
    _urlCtrl.text = cfg.baseUrl;
    _keyCtrl.text = cfg.apiKey;
    _selectedModel = cfg.model.isEmpty ? null : cfg.model;
    if (_selectedModel != null) _models = [_selectedModel!];
    _pets = SettingsService.getPets();
    _activePetId = SettingsService.getActivePetId();
  }

  @override
  void dispose() {
    _urlCtrl.dispose();
    _keyCtrl.dispose();
    super.dispose();
  }

  // ====== 立即拉取模型列表（用当前输入框的值，无需先保存）======
  Future<void> _fetchModels() async {
    final url = _urlCtrl.text.trim();
    final key = _keyCtrl.text.trim();
    if (url.isEmpty || key.isEmpty) {
      setState(() => _modelError = '请先填写 API 地址和密钥');
      return;
    }
    setState(() { _loadingModels = true; _modelError = null; });
    try {
      final models = await LlmService.listModels(url, key);
      setState(() {
        _models = models;
        _loadingModels = false;
        if (models.isEmpty) {
          _modelError = '未返回可用模型';
        } else if (_selectedModel == null || !models.contains(_selectedModel)) {
          _selectedModel = models.first;
        }
      });
    } catch (e) {
      setState(() {
        _loadingModels = false;
        _modelError = '获取失败：$e';
      });
    }
  }

  Future<void> _save() async {
    await SettingsService.saveLlmConfig(LlmConfig(
      baseUrl: _urlCtrl.text.trim(),
      apiKey: _keyCtrl.text.trim(),
      model: _selectedModel ?? '',
    ));
    await SettingsService.savePets(_pets);
    await SettingsService.setActivePetId(_activePetId);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
      content: Text('已保存'), behavior: SnackBarBehavior.floating, backgroundColor: AppTheme.primary,
    ));
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(gradient: AppTheme.bgGradient),
        child: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(8, 8, 20, 0),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_rounded, color: AppTheme.textDark),
                      onPressed: () => Navigator.pop(context),
                    ),
                    const Text('设置', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppTheme.textDark)),
                    const Spacer(),
                    TextButton(onPressed: _save, child: const Text('保存', style: TextStyle(fontWeight: FontWeight.bold))),
                  ],
                ),
              ),
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
                  children: [
                    _sectionTitle('🐾 我的宠物'),
                    const SizedBox(height: 10),
                    ..._pets.map(_petTile),
                    _addPetButton(),
                    const SizedBox(height: 28),

                    _sectionTitle('🤖 AI 大模型配置'),
                    const SizedBox(height: 4),
                    const Padding(
                      padding: EdgeInsets.only(bottom: 12),
                      child: Text('配置 OpenAI 兼容接口，用于生成生动的心声翻译（可选）',
                        style: TextStyle(fontSize: 12, color: AppTheme.textGray)),
                    ),
                    _llmCard(),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionTitle(String t) => Text(t,
      style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: AppTheme.textDark));

  Widget _llmCard() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: AppTheme.cardShadow,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _field('API 地址', _urlCtrl, hint: 'https://xxx.com/v1'),
          const SizedBox(height: 14),
          _field('API 密钥', _keyCtrl, hint: 'sk-...', obscure: _obscureKey, toggleObscure: () {
            setState(() => _obscureKey = !_obscureKey);
          }),
          const SizedBox(height: 18),

          // 模型选择
          Row(
            children: [
              const Text('模型', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppTheme.textDark)),
              const Spacer(),
              TextButton.icon(
                onPressed: _loadingModels ? null : _fetchModels,
                icon: _loadingModels
                    ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.refresh_rounded, size: 16),
                label: Text(_loadingModels ? '获取中…' : '获取模型列表'),
                style: TextButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 8), visualDensity: VisualDensity.compact),
              ),
            ],
          ),
          const SizedBox(height: 6),
          if (_models.isEmpty)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 12),
              decoration: BoxDecoration(
                color: const Color(0xFFF7F6FB),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Text('点击「获取模型列表」加载可选模型',
                style: TextStyle(fontSize: 13, color: AppTheme.textGray)),
            )
          else
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                color: const Color(0xFFF7F6FB),
                borderRadius: BorderRadius.circular(12),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: _models.contains(_selectedModel) ? _selectedModel : null,
                  isExpanded: true,
                  hint: const Text('选择模型'),
                  items: _models.map((m) => DropdownMenuItem(value: m, child: Text(m, overflow: TextOverflow.ellipsis))).toList(),
                  onChanged: (v) => setState(() => _selectedModel = v),
                ),
              ),
            ),
          if (_modelError != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(_modelError!, style: const TextStyle(fontSize: 12, color: Colors.redAccent)),
            ),
        ],
      ),
    );
  }

  Widget _field(String label, TextEditingController ctrl,
      {String? hint, bool obscure = false, VoidCallback? toggleObscure}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppTheme.textDark)),
        const SizedBox(height: 6),
        TextField(
          controller: ctrl,
          obscureText: obscure,
          style: const TextStyle(fontSize: 14),
          decoration: InputDecoration(
            hintText: hint,
            isDense: true,
            filled: true,
            fillColor: const Color(0xFFF7F6FB),
            contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
            suffixIcon: toggleObscure == null ? null : IconButton(
              icon: Icon(obscure ? Icons.visibility_off_rounded : Icons.visibility_rounded, size: 18),
              onPressed: toggleObscure,
            ),
          ),
        ),
      ],
    );
  }

  Widget _petTile(PetProfile pet) {
    final active = pet.id == _activePetId;
    return GestureDetector(
      onTap: () => setState(() => _activePetId = pet.id),
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: AppTheme.cardShadow,
          border: active ? Border.all(color: AppTheme.primary, width: 2) : null,
        ),
        child: Row(
          children: [
            Text(pet.species == 'dog' ? '🐶' : '🐱', style: const TextStyle(fontSize: 28)),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(pet.name, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: AppTheme.textDark)),
                  Text('${pet.species == 'dog' ? '狗狗' : '猫咪'} · ${pet.age}岁${pet.breed.isNotEmpty ? ' · ${pet.breed}' : ''}',
                    style: const TextStyle(fontSize: 12, color: AppTheme.textGray)),
                ],
              ),
            ),
            if (active)
              const Padding(
                padding: EdgeInsets.only(right: 4),
                child: Icon(Icons.check_circle_rounded, color: AppTheme.primary, size: 20),
              ),
            IconButton(
              icon: const Icon(Icons.edit_rounded, size: 18, color: AppTheme.textGray),
              onPressed: () => _editPet(pet),
            ),
            IconButton(
              icon: const Icon(Icons.delete_outline_rounded, size: 18, color: Colors.redAccent),
              onPressed: () => setState(() {
                _pets.removeWhere((p) => p.id == pet.id);
                if (_activePetId == pet.id) _activePetId = _pets.isNotEmpty ? _pets.first.id : null;
              }),
            ),
          ],
        ),
      ),
    );
  }

  Widget _addPetButton() {
    return GestureDetector(
      onTap: () => _editPet(null),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppTheme.primary.withValues(alpha: 0.4), width: 1.5),
        ),
        child: const Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.add_rounded, color: AppTheme.primary, size: 20),
            SizedBox(width: 6),
            Text('添加宠物', style: TextStyle(color: AppTheme.primary, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }

  void _editPet(PetProfile? existing) {
    final nameCtrl = TextEditingController(text: existing?.name ?? '');
    final breedCtrl = TextEditingController(text: existing?.breed ?? '');
    final ageCtrl = TextEditingController(text: existing?.age.toString() ?? '1');
    String species = existing?.species ?? 'cat';

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setSheet) => Padding(
          padding: EdgeInsets.only(bottom: MediaQuery.of(ctx).viewInsets.bottom),
          child: Container(
            padding: const EdgeInsets.all(24),
            decoration: const BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(existing == null ? '添加宠物' : '编辑宠物',
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppTheme.textDark)),
                const SizedBox(height: 20),
                // 物种切换
                Row(
                  children: [
                    _speciesChip('🐱 猫咪', species == 'cat', () => setSheet(() => species = 'cat')),
                    const SizedBox(width: 12),
                    _speciesChip('🐶 狗狗', species == 'dog', () => setSheet(() => species = 'dog')),
                  ],
                ),
                const SizedBox(height: 16),
                _field('名字', nameCtrl, hint: '给宝贝起个名字'),
                const SizedBox(height: 14),
                Row(
                  children: [
                    Expanded(child: _field('年龄', ageCtrl, hint: '岁')),
                    const SizedBox(width: 14),
                    Expanded(flex: 2, child: _field('品种（可选）', breedCtrl, hint: '如 英短/柯基')),
                  ],
                ),
                const SizedBox(height: 24),
                SizedBox(
                  width: double.infinity,
                  child: Container(
                    decoration: BoxDecoration(
                      gradient: AppTheme.primaryGradient,
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: TextButton(
                      style: TextButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 14), foregroundColor: Colors.white),
                      onPressed: () {
                        final name = nameCtrl.text.trim();
                        if (name.isEmpty) return;
                        final age = int.tryParse(ageCtrl.text.trim()) ?? 1;
                        setState(() {
                          if (existing == null) {
                            final pet = PetProfile(
                              id: const Uuid().v4(), name: name, species: species,
                              age: age, breed: breedCtrl.text.trim(),
                            );
                            _pets.add(pet);
                            _activePetId ??= pet.id;
                          } else {
                            existing.name = name;
                            existing.species = species;
                            existing.age = age;
                            existing.breed = breedCtrl.text.trim();
                          }
                        });
                        Navigator.pop(ctx);
                      },
                      child: const Text('保存', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                    ),
                  ),
                ),
                const SizedBox(height: 8),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _speciesChip(String label, bool active, VoidCallback onTap) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: active ? AppTheme.primary.withValues(alpha: 0.1) : const Color(0xFFF7F6FB),
            borderRadius: BorderRadius.circular(12),
            border: active ? Border.all(color: AppTheme.primary, width: 1.5) : null,
          ),
          child: Center(child: Text(label, style: TextStyle(
            fontWeight: active ? FontWeight.bold : FontWeight.normal,
            color: active ? AppTheme.primary : AppTheme.textGray,
          ))),
        ),
      ),
    );
  }
}
