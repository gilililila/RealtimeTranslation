from fastapi import FastAPI, Form, UploadFile, File
import shutil
import librosa
import os
import uvicorn
import gc

import time
import logging
from pathlib import Path

current_file_path = Path(__file__).resolve()

project_root = current_file_path.parent.parent.parent

model_root = project_root / "model"
log_root = project_root / "log"

log_root.mkdir(parents=True, exist_ok=True) # 确保日志目录存在，不存在则自动创建

logging.basicConfig(
    filename=str(log_root / "performance.log"), 
    level=logging.INFO, 
    format='%(asctime)s - %(message)s'
    )

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor
import torch
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_PATH = {
    "asr": str(model_root / "asr" / "whisper-large"), 
    "mt": str(model_root / "mt" / "nllb-200-distilled-600M")
}

MODEL_PATH = {
    "asr": {
        "whisper-large": str(model_root / "asr" / "whisper-large"),
        "whisper-large-finetune": str(model_root / "asr" / "whisper-large-finetune")
        },
    "mt": {
        "nllb-200-distilled-600M": str(model_root / "mt" / "nllb-200-distilled-600M"),
        "nllb-200-distilled-600M-finetune": str(model_root / "mt" / "nllb-200-distilled-600M-finetune")
    }
}

global_models = {
    "asr": {
        "model": None,
        # "configuration": None,
        "processor": None,
        "current_type": None
        },
    "mt": {
        "model": None,
        "tokenizer": None,
        "current_type": None
    }
}

# 加载asr模型
def load_asr(model_type: str):
    """切换asr模型

    Args:
        model_type (str): 需要的asr模型

    Returns:
        (bool): 模型是否成功加载
        
        (str): 加载状态/失败原因
    """
    if global_models["asr"]["current_type"] == model_type:
        return True, "asr模型已加载完成"
    print(f"[ASR] 正在加载 {model_type} 模型")
    
    if global_models["asr"]["model"] is not None:
        del global_models["asr"]["model"]
        # del global_models["asr"]["configuration"]
        del global_models["asr"]["processor"]
        global_models["asr"]["model"] = None
        # global_models["asr"]["configuration"] = None
        global_models["asr"]["processor"] = None
        
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        base_model_path = BASE_MODEL_PATH["asr"]
        processor = WhisperProcessor.from_pretrained(base_model_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_path).to(device)
        base_model.generation_config.forced_decoder_ids = None
        
        model_path = MODEL_PATH["asr"][model_type]
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"[ASR] 检测到 LoRA Adapter, 正在挂载微调权重")
            # 使用 PeftModel 加载 Adapter
            peft_model = PeftModel.from_pretrained(base_model, model_path)
            
            model = peft_model.merge_and_unload()
        else:
            # 如果是全量微调（没有 adapter 文件），直接用 base_model 变量即可
            # 但针对你的截图，肯定会走上面那个 if 分支
            model = base_model
        # global_models["asr"]["configuration"] = WhisperConfig.from_pretrained(model_path)
        global_models["asr"]["processor"] = processor
        global_models["asr"]["model"] = model
        global_models["asr"]["current_type"] = model_type
        print(f"[ASR] 加载完成")
        return True, "success"
    except Exception as e:
        print(f"[ASR] 加载失败: {e}")
        return False, str(e)

# 加载mt模型
def load_mt(model_type: str):
    """切换mt模型

    Args:
        model_type (str): 需要的asr模型

    Returns:
        (bool): 模型是否成功加载
        
        (str): 加载状态/失败原因
    """
    if global_models["mt"]["current_type"] == model_type:
        return True, "mt模型已加载完成"
    print(f"[MT] 正在加载 {model_type} 模型")
    
    if global_models["mt"]["model"] is not None:
        del global_models["mt"]["model"]
        del global_models["mt"]["tokenizer"]
        global_models["mt"]["model"] = None
        global_models["mt"]["tokenizer"] = None
        
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        base_model_path = BASE_MODEL_PATH["mt"]
        model_path = MODEL_PATH["mt"][model_type]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path).to(device)
        
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"[MT] 检测到 LoRA Adapter, 正在挂载微调权重...")
            # 使用 PeftModel 加载 Adapter
            peft_model = PeftModel.from_pretrained(base_model, model_path)
            
            model = peft_model.merge_and_unload()
        else:
            # 如果是全量微调（没有 adapter 文件），直接用 base_model 变量即可
            # 但针对你的截图，肯定会走上面那个 if 分支
            model = base_model
            
        global_models["mt"]["tokenizer"] = tokenizer
        global_models["mt"]["model"] = model
        global_models["mt"]["current_type"] = model_type
        print(f"[MT] 加载完成")
        return True, "success"
    except Exception as e:
        print(f"[MT] 加载失败: {e}")
        return False, str(e)   

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # 【启动逻辑 (Startup)】: yield 之前
    print("🚀 系统启动中... 正在加载默认模型")
    
    # 加载默认模型 (微调前)
    success_asr, msg_asr = load_asr("whisper-large")
    success_mt, msg_mt = load_mt("nllb-200-distilled-600M")
    
    if success_asr and success_mt:
        print("默认模型加载完毕")
    else:
        print(f"模型加载错误: ASR={msg_asr}, MT={msg_mt}")

    yield  # 应用程序运行期间，代码会停在这里

    print("系统关闭，正在清理显存")
    
    # 清空全局引用
    global_models["asr"]["model"] = None
    # global_models["asr"]["configuration"] = None
    global_models["asr"]["processor"] = None
    global_models["mt"]["model"] = None
    global_models["mt"]["tokenizer"] = None
    
    gc.collect()
    torch.cuda.empty_cache()
    print("系统结束")

@app.post("/switch_asr")
async def api_switch_asr(model_name: str = Form(...)):
    """接收参数: asrmodel1 或 asrmodel2"""
    success, msg = load_asr(model_name)
    if success:
        return {"status": "success", "current": model_name}
    else:
        return {"status": "error", "message": msg}

@app.post("/switch_mt")
async def api_switch_mt(model_name: str = Form(...)):
    """接收参数: mtmodel1 或 mtmodel2"""
    success, msg = load_mt(model_name)
    if success:
        return {"status": "success", "current": model_name}
    else:
        return {"status": "error", "message": msg}

# @app.post("/asr")
# async def asr(
#     src_lang: str = Form(...), 
#     audio_file: UploadFile = File(...)
#     ):
#     model = global_models["asr"]["model"]
#     processor = global_models["asr"]["processor"]
#     # configuration = global_models["asr"]["configuration"]
    
#     if model is None:
#         return {"error": "[ASR] 模型未加载"}
    
#     temp_filename = f"temp_{audio_file.filename}"
#     try:
#         with open(temp_filename, "wb") as b:
#             shutil.copyfileobj(audio_file.file, b)
#         lang_map = {
#             "中文": "chinese",
#             "粤语": "cantonese",
#             "英语": "english"
#         }
#         lang = lang_map.get(src_lang)
#         audio_array, _ = librosa.load(temp_filename, sr=16000)

#         # 记录音频时长和开始时间
#         audio_time = librosa.get_duration(y=audio_array, sr=16000)
#         start_time = time.time()
        
#         inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
#         input_features = inputs.input_features.to(device)
#         predicted_ids = model.generate(input_features, language=lang, task="transcribe")
#         transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
#         # 计算推理时长以及RTF(Real Time Factor)
#         inference_time = time.time() - start_time
#         rtf = inference_time / audio_time
#         logging.info(f"[ASR] 音频时长: {audio_time:.2f}s, 推理时长: {inference_time:.2f}, RTF: {rtf:.4f}")
        
#     finally:
#         if os.path.exists(temp_filename):
#             os.remove(temp_filename)
#     return{
#         "text": transcription,
#         "asr_performance":{
#             "audio_time": audio_time,
#             "inference_time": inference_time,
#             "RTF": rtf
#         }
#         }


@app.post("/asr")
async def asr(
    src_lang: str = Form(...), 
    audio_file: UploadFile = File(...)
    ):
    model = global_models["asr"]["model"]
    processor = global_models["asr"]["processor"]
    
    if model is None:
        return {"error": "[ASR] 模型未加载"}
    
    temp_filename = f"temp_{audio_file.filename}"
    try:
        with open(temp_filename, "wb") as b:
            shutil.copyfileobj(audio_file.file, b)
        
        # 1. 语言代码映射修正
        # Whisper 的 language 参数需要的是简单的代码，比如 'zh', 'en'
        # 而不是 'chinese', 'english' (虽然 processor 有时能自动识别，但用标准代码最稳)
        lang_map = {
            "中文": "Chinese",
            "粤语": "zh", # Whisper 没有专门的粤语代码，通常归类为 zh，或者是 model specific
            "英语": "English"
        }
        lang = lang_map.get(src_lang, "zh") # 默认 zh

        audio_array, _ = librosa.load(temp_filename, sr=16000)
        audio_time = librosa.get_duration(y=audio_array, sr=16000)
        start_time = time.time()
        
        # 2. 处理输入特征
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        input_features = inputs.input_features.to(device)
        
        # if "attention_mask" in inputs:
        #     attention_mask = inputs.attention_mask.to(device)
        # else:
        #     # 对于 Whisper，通常 feature 是定长的 (3000帧)，mask 全为 1 即可
        #     attention_mask = torch.ones(input_features.shape[0], input_features.shape[2], dtype=torch.long).to(device)
        
        # 【核心修改点】显式获取 forced_decoder_ids
        # 这样既避免了 config 里的冲突，又明确告诉了模型用什么语言开始
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=lang, 
            task="transcribe"
        )
        
        # 3. 生成配置优化 (防止无限循环)
        # no_repeat_ngram_size: 防止 "你你你" 这种重复
        # max_new_tokens: 强制限制生成长度，防止 "專專專..." 无限输出
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=256,
            
            # 【修复4】抑制重复生成的强力参数
            no_repeat_ngram_size=3,    # 禁止连续3个字重复
            repetition_penalty=1.1,    # 惩罚重复内容
            temperature=0.2,           # 降低随机性，让输出更稳定
            
            # 【消除警告】显式关闭不需要的自动设置，防止 LogitsProcessor 冲突警告
            use_cache=True
        )
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # transcription = 
        
        inference_time = time.time() - start_time
        rtf = inference_time / audio_time
        logging.info(f"[ASR] 音频时长: {audio_time:.2f}s, 推理时长: {inference_time:.2f}, RTF: {rtf:.4f}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    return{
        "text": transcription,
        "asr_performance":{
            "audio_time": audio_time,
            "inference_time": inference_time,
            "RTF": rtf
        }
    }

@app.post("/mt")
async def mt(
    src_lang: str = Form(...),
    tar_lang: str = Form(...),
    text: str = Form(...)
):
    model = global_models["mt"]["model"]
    tokenizer = global_models["mt"]["tokenizer"]
    
    if model is None:
        return {"error": "[MT] 模型未加载"}
    
    lang_map = {
        "中文": "zho_Hans",
        "粤语": "yue_Hant",
        "英语": "eng_Latn"
    }
    if not text:
        return {"error": "Text cannot be empty"}
    
    try:
        tokenizer.src_lang = lang_map[src_lang]
        inputs = tokenizer(text, return_tensors="pt").to(device)
        try:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(lang_map[tar_lang])
        except KeyError:
            return {"error": f"Unsupported target language code: {tar_lang}"}

        start_time = time.time()
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs, 
                forced_bos_token_id=forced_bos_token_id, 
                max_length=100  # 根据需要调整最大长度
            )
        
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # 计算推理时间以及TPS(Token/s)
        inference_time = time.time() - start_time
        token_nums = len(generated_tokens[0])
        tps = token_nums / inference_time
        logging.info(f"[MT] 输入长度: {len(text)}, 输出Token长度: {token_nums}, 推理时间: {inference_time:.2f}s, TPS: {tps:.2f}")
        
        return {
            "translated_text": result,
            "mt_performance": {
                "input_length": len(text),
                "token_length": token_nums,
                "inference_time": inference_time,
                "token_per_sec": tps
            }
            }

    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)

