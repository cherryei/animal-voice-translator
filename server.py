"""
Animal Voice Translator - Backend API Server
提供音频分类和 LLM 翻译的 REST API
"""

import os, sys, json, warnings, logging
warnings.filterwarnings("ignore")

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 添加 scripts 到路径
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# 导入推理模块
import importlib.util
spec = importlib.util.spec_from_file_location("inference_mod", SCRIPTS_DIR / "04_inference.py")
inference_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_mod)
AnimalVoiceClassifier = inference_mod.AnimalVoiceClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pet-voice-server")

app = FastAPI(title="Animal Voice Translator API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局分类器实例
classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        logger.info("加载分类器模型...")
        models_dir = Path(__file__).parent / "models"
        model_path = models_dir / "mlp_classifier_final.pt"
        if model_path.exists():
            classifier = AnimalVoiceClassifier(str(model_path))
        else:
            raise RuntimeError(f"模型文件不存在: {model_path}")
    return classifier


@app.get("/")
def root():
    return {
        "name": "Animal Voice Translator API",
        "version": "1.0.0",
        "status": "running",
    }


@app.post("/api/classify")
async def classify_audio(
    audio: UploadFile = File(...),
):
    """上传音频文件，返回分类结果"""
    try:
        # 保存音频到临时文件
        temp_dir = Path(__file__).parent / "data" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / audio.filename

        content = await audio.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 分类
        clf = get_classifier()
        result = clf.predict(str(temp_path))

        # 清理临时文件
        try:
            temp_path.unlink()
        except:
            pass

        return {
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
        }

    except Exception as e:
        logger.error(f"分类失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translate")
async def translate_classification(data: dict):
    """
    接收分类结果 + LLM 配置，返回翻译结果
    兼容 OpenAI API 格式
    """
    try:
        classification = data.get("classification", {})
        llm_config = data.get("llm_config", {})
        pet_info = data.get("pet_info", {})

        base_url = llm_config.get("base_url", "")
        api_key = llm_config.get("api_key", "")
        model = llm_config.get("model", "gpt-4o-mini")

        if not base_url or not api_key:
            return {"success": False, "error": "请先配置 LLM API"}

        # 构建 prompt
        prediction = classification.get("prediction", "unknown")
        confidence = classification.get("confidence", 0)
        probs = classification.get("all_probabilities", {})

        # Top-3
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:3]
        probs_text = "\n".join([f"  - {k}: {v:.1%}" for k, v in sorted_probs])

        species = pet_info.get("species", "宠物")
        name = pet_info.get("name", "宝贝")
        context = pet_info.get("context", "")

        system_prompt = """你是一个宠物叫声翻译助手。根据声学模型分析结果，用中文描述宠物可能的情绪和意图。

规则：
1. 保持诚实——置信度低于60%时在翻译中注明"仅供参考"
2. 不要过度拟人化，每个字段值都必须填完整不能为空
3. 输出格式为严格JSON：
{
  "emotion": "情绪标签（中文，如：兴奋、焦虑、饥饿、舒适）",
  "translation": "拟人化翻译（一句话，中文）",
  "confidence_level": "高/中/低",
  "reasoning": "判断依据（中文，说明为什么这么判断）",
  "suggestion": "给主人的行动建议（中文）"
}
必须输出完整JSON，不要额外文字。"""

        user_prompt = f"""分析以下{species}叫声：

【分类结果】
最可能: {prediction}（置信度: {confidence:.1%}）
候选:
{probs_text}

【信息】
- 物种: {species}
- 名字: {name}
- 场景: {context if context else '未知'}

输出JSON格式分析结果，全部使用中文。"""

        # 调用 OpenAI 兼容 API
        import httpx
        client = httpx.Client(timeout=60.0)

        # 确保 URL 以 /v1/chat/completions 结尾
        url = base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # 尝试解析 JSON
        try:
            parsed = json.loads(content)
        except:
            # 如果返回的不是纯 JSON，尝试提取
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = {"translation": content, "emotion": "未知"}

        return {
            "success": True,
            "translation": parsed.get("translation", ""),
            "emotion": parsed.get("emotion", ""),
            "confidence_level": parsed.get("confidence_level", ""),
            "reasoning": parsed.get("reasoning", ""),
            "suggestion": parsed.get("suggestion", ""),
            "raw": parsed,
            "model_used": model,
        }

    except Exception as e:
        logger.error(f"翻译失败: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/llm/models")
async def list_llm_models(base_url: str = "", api_key: str = ""):
    """查询 LLM API 支持的模型列表"""
    if not base_url or not api_key:
        return {"models": [], "error": "请配置 API 地址和 Key"}

    try:
        import httpx
        url = base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            url = url.replace("/chat/completions", "/models")
        else:
            url += "/models"

        headers = {"Authorization": f"Bearer {api_key}"}
        response = httpx.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        models = [m["id"] for m in data.get("data", []) if "tts" not in m["id"]]
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}


@app.get("/api/models")
async def list_species():
    """返回支持的物种列表"""
    return {
        "species": ["cat", "dog"],
        "labels": {
            "cat": ["brushing", "food", "isolation", "meowing"],
            "dog": ["barking"],
        }
    }


if __name__ == "__main__":
    print("启动 Animal Voice Translator 服务器...")
    print("API 文档: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
