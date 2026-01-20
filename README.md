# Real-Time Speech Translation System
一个基于本地大模型的前后端分离的实时语音翻译系统。该系统整合了ASR(自动语音识别)、MT(机器翻译)和TTS(语音合成)三大模块，能够实现低延迟的端到端语音翻译。

## 主要功能
- **实时语音识别(ASR)**：基于openai/whisper-large-v3，无需手动触发，录音结束立即开始识别。
- **高精度翻译(MT)**：基于facebook/nllb-200-distilled-600M，提供精准、快速的翻译。
- **自克隆语音合成(TTS)**：基于FunAudioLLM/CosyVoice2-0.5B，克隆用户音色，快速语音生成。
- **前后端分离**：前端使用Streamlit库搭建，后端使用FastAPI搭建，实现全双工实时通信。

## 目录结构
```text
RealtimeTranslation/
├── model/                   # 模型权重目录 (需手动下载，微调前后模型须在不同文件夹)
│   ├── asr/                 # 语音识别模型 & LoRA
│   ├── mt/                  # 机器翻译模型 & LoRA
│   └── tts/                 # 语音合成模型 & LoRA
├── src/                     # 源代码
│   ├── backend/             # 后端核心代码 (FastAPI)
│   │   ├── CosyVoiceEnd.py  # 启动cosyvoice的api服务
│   │   ├── WhisperEnd.py    # 启动whisper和nllb的api服务
│   │   └── requirement.txt  # 后端依赖
│   └── frontend/            # 前端代码(streamlit)
│       ├── FrontEnd.py      # 启动前端
│       └── requirement.txt  # 前端依赖
├── log/                     # 运行日志
└── README.md                # 项目文档
```

# 快速开始
1. 环境准备
   - OS: Linux(recommended Ubuntu 20.04+)
   - GPU: 现存大于10GB的NVIDIA显卡
   - Python: 3.10
   - CUDA: 12.8+

2. 安装代码
```Bash
# 1.克隆项目
git clone https://github.com/gilililila/RealtimeTranslation.git
cd RealtimeTranslation

# 2. 创建并激活虚拟环境
conda create -n rtt python=3.10
conda activate rtt

# 3. 安装依赖
# 注意：请先根据自己的CUDA版本安装对应PyTorch
pip install -r src/backend/requirements.txt
pip install -r src/frontend/requirements.txt
```

3. 模型下载
由于模型文件过于庞大，未包含在仓库中，请从HuggingFace或ModelScope中下载对应基座模型，并严格按照以下结构放置：
- ASR模型：放置于model/asr/[modelname]
- MT模型：放置于model/mt/[modelname]
- TTS模型：放置于model/tts/[modelname]
(如果有LoRA微调权重，请在asr、mt、tts目录下新建文件夹放置)

4. 运行服务
- 修改backend文件夹下的python文件，将最后一行`uvicorn.run(app, host="0.0.0.0", port=6006)`中host改为127.0.0.1。
- 修改BASE_MODEL_PATH于MODEL_PATH，将内容修改为你的模型文件夹的命名。
- 修改frontend/FrontEnd.py文件，将其中API_BASE修改为你的WhiperEnd.py中暴露的API接口；将其中TTS_API修改为你的CosyVoiceEnd.py中暴露的API接口
