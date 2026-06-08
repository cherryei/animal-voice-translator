import 'dart:async';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart' as pp;
import 'package:record/record.dart';
import '../theme.dart';
import '../services/emotion_classifier.dart';
import '../services/llm_service.dart';
import '../services/settings_service.dart';
import '../services/wav_decoder.dart';
import '../services/acoustic_analyzer.dart';
import '../models/translation_result.dart';
import '../models/pet_profile.dart';
import 'settings_screen.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  final _classifier = EmotionClassifier();
  final _recorder = AudioRecorder();
  bool _modelLoaded = false;
  bool _isRecording = false;
  bool _isProcessing = false;
  String _status = '点击麦克风，记录宝贝的声音';
  String? _audioPath;

  // 录音相关
  late AnimationController _pulseCtrl;
  StreamSubscription<Amplitude>? _ampSub;
  Timer? _maxTimer;
  Timer? _silenceTimer;
  double _currentAmp = 0;
  final List<double> _waveform = List.filled(40, 0.06);
  bool _hasSpoken = false;
  bool _stopping = false;   // 防止"静音自动停"与"手动点停"同时触发导致状态错乱
  bool _analyzing = false;  // 防止重复分析

  static const _maxRecordSec = 60;        // 最长 60 秒（实际场景足够）
  static const _silenceThreshold = -40.0; // dB
  static const _silenceHoldMs = 3000;     // 说话后静音满 3 秒才自动结束

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 1200))..repeat();
    _loadModel();
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    _ampSub?.cancel();
    _maxTimer?.cancel();
    _silenceTimer?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    await _classifier.load();
    if (mounted) setState(() => _modelLoaded = _classifier.isLoaded);
  }

  PetProfile? get _activePet {
    final id = SettingsService.getActivePetId();
    if (id == null) return null;
    final pets = SettingsService.getPets();
    try { return pets.firstWhere((p) => p.id == id); } catch (_) { return null; }
  }

  // ============ 录音 ============
  // 单按钮：点一次开始；再点一次结束并分析。说话后静音 3 秒 / 满 60 秒也会自动结束。
  Future<void> _toggleRecording() async {
    if (_isProcessing) return;
    if (_isRecording) {
      await _stopRecording(autoTriggered: false);
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    if (_isRecording || _isProcessing) return;
    if (!await _recorder.hasPermission()) {
      _toast('需要麦克风权限');
      return;
    }
    final dir = await pp.getTemporaryDirectory();
    final path = '${dir.path}/rec_${DateTime.now().millisecondsSinceEpoch}.wav';
    _audioPath = path;
    _hasSpoken = false;
    _stopping = false;

    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 16000, numChannels: 1),
      path: path,
    );

    setState(() {
      _isRecording = true;
      _status = '正在录音… 再次点击结束';
    });

    // 振幅监听 → 波形 + 静音检测
    _ampSub = _recorder.onAmplitudeChanged(const Duration(milliseconds: 100)).listen((amp) {
      if (!_isRecording) return;
      final db = amp.current;
      final norm = ((db + 50) / 50).clamp(0.05, 1.0);
      setState(() {
        _currentAmp = norm;
        _waveform.removeAt(0);
        _waveform.add(norm);
      });

      if (db > _silenceThreshold) {
        _hasSpoken = true;
        _silenceTimer?.cancel();
        _silenceTimer = null;
      } else if (_hasSpoken && _silenceTimer == null) {
        // 说过话后陷入静音 → 倒计时自动停
        _silenceTimer = Timer(const Duration(milliseconds: _silenceHoldMs), () {
          _stopRecording(autoTriggered: true);
        });
      }
    });

    // 最大时长保护
    _maxTimer = Timer(const Duration(seconds: _maxRecordSec), () {
      _stopRecording(autoTriggered: true);
    });
  }

  Future<void> _stopRecording({required bool autoTriggered}) async {
    // 幂等：手动点停和自动停可能同时发生，只执行一次
    if (_stopping || !_isRecording) return;
    _stopping = true;

    _ampSub?.cancel();
    _ampSub = null;
    _maxTimer?.cancel();
    _maxTimer = null;
    _silenceTimer?.cancel();
    _silenceTimer = null;

    // 关键：立刻离开"录音中"状态，按钮绝不会卡在"正在录音"
    setState(() {
      _isRecording = false;
      _isProcessing = true;
      _status = '正在分析声音…';
      for (int i = 0; i < _waveform.length; i++) {
        _waveform[i] = 0.06;
      }
    });

    String? path;
    try {
      path = await _recorder.stop();
    } catch (_) {
      path = _audioPath; // 停止异常时回退到已写入的文件
    }
    _stopping = false;

    if (path != null) {
      _audioPath = path;
      await _analyze();
    } else {
      if (mounted) {
        setState(() {
          _isProcessing = false;
          _status = '点击麦克风，记录宝贝的声音';
        });
      }
    }
  }

  Future<void> _pickFile() async {
    final result = await FilePicker.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav'],
    );
    final path = result?.files.single.path;
    if (path == null) return;
    if (!path.toLowerCase().endsWith('.wav')) {
      _toast('暂仅支持 WAV 文件，建议直接用 App 录音');
      return;
    }
    _audioPath = path;
    await _analyze();
  }

  // ============ 分析 ============
  Future<void> _analyze() async {
    if (_analyzing) return;
    _analyzing = true;
    final pet = _activePet;
    final species = pet?.species ?? 'cat';

    if (_audioPath == null || !_modelLoaded) {
      _toast('模型尚未就绪');
      _resetIdle();
      _analyzing = false;
      return;
    }

    setState(() {
      _isProcessing = true;
      _status = '正在分析声音…';
    });

    try {
      final decoded = WavDecoder.decodeFile(_audioPath!);
      if (decoded == null) {
        _toast('无法解析该音频，请使用 WAV 文件或直接录音');
        _resetIdle();
        return;
      }
      final audio = decoded.samples;
      if (audio.length < 1600) {
        _toast('录音太短了，再试一次');
        _resetIdle();
        return;
      }

      final emotion = _classifier.predict(audio, species);
      if (emotion == null) {
        _toast('分析失败');
        _resetIdle();
        return;
      }

      // 声学特征分析（发给 LLM 做更准确的情绪判断）
      final acousticDesc = AcousticAnalyzer.analyze(audio);

      // LLM 翻译（可选）
      TranslationResult? translation;
      String? llmError;
      final config = SettingsService.getLlmConfig();
      if (config.isValid) {
        setState(() => _status = 'AI 正在翻译心声…');
        try {
          translation = await LlmService.translate(
            emotion: emotion, config: config, pet: pet,
            context: '', acousticAnalysis: acousticDesc,
          );
        } catch (e) {
          llmError = e.toString();
        }
      }

      if (!mounted) return;
      _resetIdle();

      Navigator.push(context, MaterialPageRoute(
        builder: (_) => ResultScreen(
          emotion: emotion,
          translation: translation,
          llmError: llmError,
          pet: pet,
          acousticAnalysis: acousticDesc,
        ),
      ));
    } catch (e) {
      _toast('出错了: $e');
      _resetIdle();
    } finally {
      _analyzing = false;
    }
  }

  void _resetIdle() {
    if (!mounted) return;
    setState(() {
      _isProcessing = false;
      _isRecording = false;
      _status = '点击麦克风，记录宝贝的声音';
    });
  }

  void _toast(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg),
      behavior: SnackBarBehavior.floating,
      backgroundColor: AppTheme.textDark,
    ));
  }

  @override
  Widget build(BuildContext context) {
    final pet = _activePet;
    final llmReady = SettingsService.getLlmConfig().isValid;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(gradient: AppTheme.bgGradient),
        child: SafeArea(
          child: Column(
            children: [
              // 顶栏
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 12, 12, 0),
                child: Row(
                  children: [
                    ShaderMask(
                      shaderCallback: (b) => AppTheme.primaryGradient.createShader(b),
                      child: const Text('🐾 宠物心声',
                        style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white)),
                    ),
                    const Spacer(),
                    IconButton(
                      icon: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(12),
                          boxShadow: AppTheme.cardShadow,
                        ),
                        child: const Icon(Icons.settings_rounded, color: AppTheme.primary),
                      ),
                      onPressed: () => Navigator.push(context,
                        MaterialPageRoute(builder: (_) => const SettingsScreen())).then((_) => setState(() {})),
                    ),
                  ],
                ),
              ),

              // 宠物卡片
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
                child: _petCard(pet),
              ),

              const Spacer(),

              // 状态文字
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Text(_status,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 16, color: AppTheme.textGray, fontWeight: FontWeight.w500)),
              ),
              const SizedBox(height: 32),

              // 波形（录音时）
              SizedBox(
                height: 60,
                child: _isRecording ? _waveformView() : const SizedBox(),
              ),
              const SizedBox(height: 24),

              // 录音大按钮
              _recordButton(),

              const SizedBox(height: 40),

              // 选择文件
              if (!_isRecording && !_isProcessing)
                TextButton.icon(
                  onPressed: _pickFile,
                  icon: const Icon(Icons.folder_open_rounded, size: 18, color: AppTheme.textGray),
                  label: const Text('选择音频文件', style: TextStyle(color: AppTheme.textGray)),
                ),

              const Spacer(),

              // 底部状态条
              Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _statusChip(Icons.memory, _modelLoaded ? '模型就绪' : '加载中', _modelLoaded),
                    const SizedBox(width: 12),
                    _statusChip(Icons.translate, llmReady ? 'AI翻译已配置' : '未配置AI', llmReady),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _petCard(PetProfile? pet) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: AppTheme.cardShadow,
      ),
      child: Row(
        children: [
          Container(
            width: 52, height: 52,
            decoration: BoxDecoration(
              gradient: AppTheme.primaryGradient,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Center(child: Text(
              pet == null ? '🐾' : (pet.species == 'dog' ? '🐶' : '🐱'),
              style: const TextStyle(fontSize: 26),
            )),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: pet == null
                ? const Text('点击右侧添加你的宠物', style: TextStyle(color: AppTheme.textGray))
                : Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(pet.name, style: const TextStyle(
                        fontSize: 17, fontWeight: FontWeight.bold, color: AppTheme.textDark)),
                      const SizedBox(height: 2),
                      Text('${pet.species == 'dog' ? '狗狗' : '猫咪'} · ${pet.age}岁${pet.breed.isNotEmpty ? ' · ${pet.breed}' : ''}',
                        style: const TextStyle(fontSize: 13, color: AppTheme.textGray)),
                    ],
                  ),
          ),
          IconButton(
            icon: const Icon(Icons.edit_rounded, size: 20, color: AppTheme.primary),
            onPressed: () => Navigator.push(context,
              MaterialPageRoute(builder: (_) => const SettingsScreen())).then((_) => setState(() {})),
          ),
        ],
      ),
    );
  }

  Widget _waveformView() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: _waveform.map((h) => AnimatedContainer(
        duration: const Duration(milliseconds: 80),
        margin: const EdgeInsets.symmetric(horizontal: 2),
        width: 4,
        height: 8 + h * 50,
        decoration: BoxDecoration(
          gradient: AppTheme.primaryGradient,
          borderRadius: BorderRadius.circular(2),
        ),
      )).toList(),
    );
  }

  Widget _recordButton() {
    return GestureDetector(
      onTap: _isProcessing ? null : _toggleRecording,
      child: AnimatedBuilder(
        animation: _pulseCtrl,
        builder: (context, child) {
          final scale = _isRecording ? 1 + _currentAmp * 0.15 : 1.0;
          return Stack(
            alignment: Alignment.center,
            children: [
              // 外圈脉冲
              if (_isRecording)
                Container(
                  width: 140 + _pulseCtrl.value * 40,
                  height: 140 + _pulseCtrl.value * 40,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppTheme.accent.withValues(alpha: 0.2 * (1 - _pulseCtrl.value)),
                  ),
                ),
              // 主按钮
              Transform.scale(
                scale: scale.toDouble(),
                child: Container(
                  width: 120, height: 120,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: _isRecording ? AppTheme.warmGradient : AppTheme.primaryGradient,
                    boxShadow: [BoxShadow(
                      color: (_isRecording ? AppTheme.accent : AppTheme.primary).withValues(alpha: 0.4),
                      blurRadius: 30, offset: const Offset(0, 10),
                    )],
                  ),
                  child: _isProcessing
                      ? const Padding(padding: EdgeInsets.all(36),
                          child: CircularProgressIndicator(color: Colors.white, strokeWidth: 3))
                      : Icon(_isRecording ? Icons.stop_rounded : Icons.mic_rounded,
                          color: Colors.white, size: 52),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _statusChip(IconData icon, String label, bool ok) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: AppTheme.cardShadow,
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: ok ? Colors.green : Colors.orange),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(fontSize: 12, color: AppTheme.textGray)),
        ],
      ),
    );
  }
}
