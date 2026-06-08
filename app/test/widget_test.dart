import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:pet_voice_translator/services/settings_service.dart';
import 'package:pet_voice_translator/screens/home_screen.dart';

void main() {
  testWidgets('首页能正常加载并显示标题', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues({});
    await SettingsService.init();

    await tester.pumpWidget(const MaterialApp(home: HomeScreen()));
    await tester.pump();

    expect(find.text('🐾 宠物心声'), findsOneWidget);
    expect(find.byIcon(Icons.mic_rounded), findsOneWidget);
  });
}
