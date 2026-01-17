import sys
import os
from pathlib import Path

# ================= 配置路径 (最关键的步骤) =================
# 1. 获取当前脚本的绝对路径
current_file_path = Path(__file__).resolve()

project_root = current_file_path.parent.parent.parent

# 3. 定位 CosyVoice 的根目录
cosyvoice_root = project_root / "model" / "tts" / "CosyVoice"

# 4. 检查路径是否存在 (方便调试)
if not cosyvoice_root.exists():
    raise FileNotFoundError(f"找不到 CosyVoice 目录，请检查路径: {cosyvoice_root}")

# 5. 将 CosyVoice 目录加入系统路径，这样 Python 才能找到 cosyvoice 包
sys.path.insert(0, str(cosyvoice_root))

# 注意：CosyVoice 经常依赖 third_party 下的模块 (如 Matcha-TTS)，也需要加上
matcha_path = cosyvoice_root / "third_party" / "Matcha-TTS"
if matcha_path.exists():
    sys.path.append(str(matcha_path))

from fastapi import FastAPI,Form,UploadFile,File
from fastapi.responses import StreamingResponse
from cosyvoice.cli.cosyvoice import AutoModel

import tempfile
import torch
import torchaudio
import io
import soundfile as sf

model_dir = cosyvoice_root / "pretrained_models" / "CosyVoice2-0.5B"

app = FastAPI()

print("正在加载 CosyVoice2 模型")
model = AutoModel(model_dir=str(model_dir))
print("模型加载完成")

@app.post("/tts")
async def TTS(
    tar_text: str = Form(...),
    tar_lang: str = Form(...),
    prompt_audio: UploadFile = File(...),
    prompt_text: str = Form(...)
    ):
    """
    接收目标文本、参考文本和参考音频，进行零样本语音克隆
    """
    prompt_wav_path = ""
    try:
        # 1. 将上传的音频保存为临时文件，因为CosyVoice读取的是文件路径
        suffix = os.path.splitext(prompt_audio.filename)[-1]
        if not suffix:
            suffix = ".wav"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await prompt_audio.read()
            tmp.write(content)
            prompt_wav_path = tmp.name

        # 2. 调用 CosyVoice2 进行推理
        # inference_zero_shot 参数: (目标文本, 参考音频文本, 参考音频路径)
        # 注意：这里返回的是一个生成器，我们需要获取生成的音频数据
        # 如果是短文本，通常只有一个生成结果
        
        audio = None

        print("📦 临时音频大小:", os.path.getsize(prompt_wav_path))

        waveform, sr = torchaudio.load(prompt_wav_path)
        print("🎧 torchaudio load success:", waveform.shape, sr)

        # 使用 inference_zero_shot 进行克隆
        # stream=False 表示一次性生成，适合非流式API
        print("语音生成开始")
        
        instruct_text = f"用{tar_lang}说这句话<|endofprompt|>"
        
        responses = model.inference_instruct2(tar_text, instruct_text, prompt_wav_path, stream=False)
        
        for response in responses:
            audio = response['tts_speech']
            # 这里我们只取第一段结果（如果文本很长可能需要拼接）
            break 

        if audio is None:
            return {"error": "Generation failed"}

        # 3. 将 Tensor 转换为 Bytes (WAV格式) 返回
        buffer = io.BytesIO()
        audio_numpy = audio.cpu().squeeze().numpy()
        sf.write(buffer, audio_numpy, model.sample_rate, format='WAV')
        # torchaudio.save(buffer, audio.cpu(), model.sample_rate, format="wav")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        return {"error": str(e)}
        
    finally:
        # 清理临时文件
        if prompt_wav_path and os.path.exists(prompt_wav_path):
            os.remove(prompt_wav_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6008)

