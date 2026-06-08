"""
LLM 桥接脚本：将分类器结果 + LLM → 自然语言翻译

不需要 GPU。只需要：
1. 训练好的分类器（04_inference.py 的 AnimalVoiceClassifier）
2. 一个 LLM API key（Claude / GPT / 通义千问）

使用示例:
  python scripts/05_llm_bridge.py /path/to/cat_meow.wav
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
# 导入推理模块（文件名以数字开头，用 importlib 处理）
import importlib.util
spec = importlib.util.spec_from_file_location("inference_mod", SCRIPTS_DIR / "04_inference.py")
inference_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_mod)
AnimalVoiceClassifier = inference_mod.AnimalVoiceClassifier

# ============================================================
# LLM API 配置
# ============================================================
# 方案 A: Anthropic Claude API（推荐）
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# 方案 B: OpenAI API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# 方案 C: 通义千问 API（国内推荐）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# ============================================================
# Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是一个动物叫声分析助手。你接收的是经过专业声学模型处理后的结构化数据（而非原始音频）。

你的任务是：
1. 基于分类结果，用自然语言描述宠物可能的情绪和意图
2. 保持诚实——置信度低于 60% 时必须标注"仅供参考"
3. 不要拟人化过度——不要编造宠物"想说"的话
4. 给出对主人有帮助的行动建议

输出格式为 JSON：
{
  "emotion": "主要情绪标签（中文）",
  "translation": "拟人化翻译（一句话，第一人称），如果置信度低请加上'可能是'",
  "confidence_level": "高/中/低",
  "reasoning": "判断依据（1-2句话）",
  "suggestion": "给主人的行动建议"
}"""


def build_user_prompt(result, species="cat", pet_name="宠物", context=""):
    """构造 LLM 输入"""
    # 取 top-3 概率
    top3 = list(result["all_probabilities"].items())[:3]

    # 置信度水平
    conf = result["confidence"]
    if conf >= 0.8:
        conf_level = "高"
    elif conf >= 0.6:
        conf_level = "中"
    else:
        conf_level = "低"

    prompt = f"""请分析以下{species}的叫声：

【分类结果】
- 最可能类别: {result['prediction']}（置信度: {conf:.1%}）
- Top 3 候选:
"""
    for label, prob in top3:
        prompt += f"  - {label}: {prob:.1%}\n"
    prompt += f"""
【宠物档案】
- 种类: {species}
- 名字: {pet_name}
- 当前场景: {context if context else '未知'}

【置信度水平】{conf_level}
"""
    return prompt


def call_llm_anthropic(prompt, api_key):
    """调用 Claude API"""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",  # 性价比最高的模型
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.content[0].text)


def call_llm_openai(prompt, api_key):
    """调用 OpenAI API"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def call_llm_dashscope(prompt, api_key):
    """调用通义千问 API"""
    import openai
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = client.chat.completions.create(
        model="qwen-turbo",
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


def analyze_audio(audio_path, classifier=None, species="cat", pet_name="宠物",
                  context="", llm_provider="auto"):
    """
    端到端分析：分类器 → LLM → 自然语言翻译

    参数:
        audio_path: 音频文件路径
        classifier: AnimalVoiceClassifier 实例（不传则自动加载）
        species: 动物种类
        pet_name: 宠物名字
        context: 录音时的场景描述
        llm_provider: "anthropic" / "openai" / "dashscope" / "auto"
    """
    # 1. 分类器预测
    if classifier is None:
        classifier = AnimalVoiceClassifier()
    result = classifier.predict(audio_path)

    print(f"分类器结果: {result['prediction']} ({result['confidence']:.1%})")

    # 2. 如果有 API key，调用 LLM
    prompt = build_user_prompt(result, species, pet_name, context)

    llm_result = None
    if llm_provider == "anthropic" or (llm_provider == "auto" and ANTHROPIC_API_KEY):
        llm_result = call_llm_anthropic(prompt, ANTHROPIC_API_KEY)
        print(f"LLM 来源: Anthropic Claude")
    elif llm_provider == "openai" or (llm_provider == "auto" and OPENAI_API_KEY):
        llm_result = call_llm_openai(prompt, OPENAI_API_KEY)
        print(f"LLM 来源: OpenAI")
    elif llm_provider == "dashscope" or (llm_provider == "auto" and DASHSCOPE_API_KEY):
        llm_result = call_llm_dashscope(prompt, DASHSCOPE_API_KEY)
        print(f"LLM 来源: 通义千问")

    return {
        "classifier_result": result,
        "llm_analysis": llm_result,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="动物叫声翻译")
    parser.add_argument("audio_path", help="音频文件路径")
    parser.add_argument("--species", default="cat", choices=["cat", "dog", "auto"],
                        help="动物种类")
    parser.add_argument("--name", default="宝贝", help="宠物名字")
    parser.add_argument("--context", default="", help="当前场景描述")
    parser.add_argument("--llm", default=None,
                        choices=["anthropic", "openai", "dashscope"],
                        help="LLM 供应商（默认自动检测环境变量）")
    parser.add_argument("--no-llm", action="store_true",
                        help="仅使用分类器，不调用 LLM")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"文件不存在: {args.audio_path}")
        sys.exit(1)

    print(f"正在分析: {args.audio_path}")
    print(f"物种: {args.species}, 名字: {args.name}, 场景: {args.context or '未知'}")
    print("=" * 50)

    # 检测 auto 物种
    species = args.species
    if species == "auto":
        # 简单启发式：从文件名判断
        fname = args.audio_path.lower()
        if "dog" in fname or "bark" in fname:
            species = "dog"
        elif "cat" in fname or "meow" in fname:
            species = "cat"
        else:
            species = "cat"

    # 加载分类器
    classifier = AnimalVoiceClassifier()

    if args.no_llm:
        # 仅分类器模式
        result = classifier.predict(args.audio_path)
        print(f"\n分类结果:")
        print(f"  类别: {result['prediction']}")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"\n概率分布:")
        for label, prob in result["all_probabilities"].items():
            bar = "█" * int(prob * 50)
            print(f"  {label:>12}: {prob:>6.2%} {bar}")
    else:
        # 完整翻译模式
        output = analyze_audio(
            args.audio_path,
            classifier=classifier,
            species=species,
            pet_name=args.name,
            context=args.context,
            llm_provider=args.llm or "auto",
        )

        print(f"\n{'=' * 50}")
        print("翻译结果:")
        print(f"{'=' * 50}")

        llm = output.get("llm_analysis")
        if llm:
            if "error" in llm:
                print(f"\nLLM 调用失败（可能未配置 API key）: {llm['error']}")
                print("\n分类器结果（无 LLM 翻译）:")
                print(f"  类别: {output['classifier_result']['prediction']}")
                print(f"  置信度: {output['classifier_result']['confidence']:.2%}")
            else:
                print(f"\n{llm.get('translation', '')}")
                print(f"\n情绪: {llm.get('emotion', '未知')}")
                print(f"置信度: {llm.get('confidence_level', '未知')}")
                print(f"推理依据: {llm.get('reasoning', '无')}")
                print(f"建议: {llm.get('suggestion', '无')}")
        else:
            print("\n分类器结果（未配置 LLM API key）:")
            result = output["classifier_result"]
            print(f"  类别: {result['prediction']}")
            print(f"  置信度: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
