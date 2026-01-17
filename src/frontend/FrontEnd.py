import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import requests

st.set_page_config(layout="wide")

API_BASE = "https://u862905-8fee-1d9fbb2f.westb.seetacloud.com:8443"

ASR_API = API_BASE + "/asr"
MT_API = API_BASE + "/mt"
TTS_API = "https://uu862905-8fee-1d9fbb2f.westb.seetacloud.com:8443/tts"

type_select=["微调前", "微调后", "手动选择"]
asr_model_select=["whisper-large", "whisper-large-finetune"]
mt_model_select=["nllb-200-distilled-600M", "nllb-200-distilled-600M-finetune"]
tts_model_select=["cosyvoice2"]
language_select = ["中文","粤语","英语"]

# 初始化 Session State
if "preset_option" not in st.session_state:
    st.session_state.preset_option = "手动选择"
if "asr_option" not in st.session_state:
    st.session_state.asr_option = asr_model_select[0]
if "mt_option" not in st.session_state:
    st.session_state.mt_option = mt_model_select[0]
if "tts_option" not in st.session_state:
    st.session_state.tts_option = tts_model_select[0]

if "src_lang" not in st.session_state:
    st.session_state.src_lang = language_select[0]
if "tar_lang" not in st.session_state:
    st.session_state.tar_lang = language_select[0]

if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None
if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = None
if "translated_text" not in st.session_state:
    st.session_state.translated_text = None
if "generated_audio" not in st.session_state:
    st.session_state.generated_audio = None

if "asr_performance" not in st.session_state:
    st.session_state.asr_performance = None
if "mt_performance" not in st.session_state:
    st.session_state.mt_performance = None

def on_asr_change():
    asr_model_name = st.session_state.asr_option
    st.session_state.preset_option = "手动选择"
    with st.spinner(f"正在切换 ASR 模型为 {asr_model_name}"):
        try:
            response = requests.post(f"{API_BASE}/switch_asr", data={"model_name": asr_model_name})
            if response.status_code == 200:
                st.success(f"ASR 模型已切换为: {asr_model_name}")
            else:
                st.error("ASR 模型切换失败")
        except Exception as e:
            st.error(f"连接失败 {e}")

def on_mt_change():
    mt_model_name = st.session_state.mt_option
    st.session_state.preset_option = "手动选择"
    
    with st.spinner(f"正在切换 MT 模型为 {mt_model_name}..."):
        try:
            resp = requests.post(f"{API_BASE}/switch_mt", data={"model_name": mt_model_name})
            if resp.status_code == 200:
                st.success(f"MT 模型已切换为: {mt_model_name}")
            else:
                st.error("MT 模型切换失败")
        except Exception as e:
            st.error(f"连接失败: {e}")

# 预设模型切换函数
def on_preset_change():
    selection = st.session_state.preset_option
    
    if selection == "手动选择":
        return
    
    if selection == "微调前":
        cur_asr = asr_model_select[0]
        cur_mt = mt_model_select[0]
        
    elif selection == "微调后":
        cur_asr = asr_model_select[1]
        cur_mt = mt_model_select[1]
    
    st.session_state.asr_option = cur_asr
    st.session_state.mt_option = cur_mt
    
    with st.spinner("正在应用预设配置..."):
        try:
            requests.post(f"{API_BASE}/switch_asr", data={"model_name": cur_asr})
            requests.post(f"{API_BASE}/switch_mt", data={"model_name": cur_mt})
            st.success(f"已切换为 {selection} 模式")
        except Exception as e:
            st.error(f"预设应用失败: {e}")


# 侧边栏页面布局
with st.sidebar:
    # 标题栏
    title_container = st.container(border=False)
    with title_container:
        title_container.markdown("<h1 style='text-align: center'>💬实时语音翻译系统</h1>", unsafe_allow_html=True)
    
    st.divider()

    st.subheader(":point_right: 选择可用模型")
    # 预设选择框：绑定 key 并设置 on_change
    st.selectbox(
        label=":thinking: 预设选择",
        options=type_select,
        key="preset_option",      # 绑定到 session_state
        on_change=on_preset_change # 绑定回调函数
    )
    # 子模型选择框：绑定 key 并设置 on_change
    st.selectbox(
        label=":ear: ASR 模型选择",
        options=asr_model_select,
        key="asr_option",
        on_change=on_asr_change
    )
    st.selectbox(
        label=":robot: 机器翻译模型选择",
        options=mt_model_select,
        key="mt_option",
        on_change=on_mt_change
    )
    st.selectbox(
        label=":loud_sound: 语音生成模型选择",
        options=tts_model_select,
        key="tts_option",
    )

    st.divider()
    
# 调用语音识别
def call_asr_api(audio_file, language):
    try:
        files = {
            "audio_file":(audio_file.name, audio_file, "audio/wav")
        }
        data = {
            "src_lang": language
        }
        if st.session_state["asr_option"] == asr_model_select[0]:
            response = requests.post(ASR_API, files=files, data=data)
        elif st.session_state["asr_option"] == asr_model_select[1]:
            response = requests.post(ASR_API, files=files, data=data)
        if response.status_code == 200:
            st.session_state["asr_performance"] = response.json().get("asr_performance")
            return response.json().get("text", "")
        else:
            st.error(f"语音识别服务器错误: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"语音识别连接失败: {e}")
        return None

# 调用机器翻译
def call_mt_api(src_text, src_lang, tar_lang):
    data = {
        "text": src_text,
        "src_lang": src_lang,
        "tar_lang": tar_lang
    }
    try:
        response = requests.post(MT_API, data=data) 

        try:
            res_json = response.json()
        except Exception:
            st.error(f"服务端返回了非 JSON 数据: {response.text[:100]}")
            return None

        if "error" in res_json:
            error_msg = res_json["error"]
            print(f"Server Error: {error_msg}") # 打印到终端方便调试
            
            # --- 在界面上显示错误 ---
            st.error(f"❌ 翻译服务报错: {error_msg}") 
            return None

        st.session_state["mt_performance"] = res_json.get("mt_performance")
        
        return res_json.get("translated_text", "")   
    
    except Exception as e:
        st.error(f"机器翻译连接失败: {e}")
        return None

# 调用语音生成
def call_tts_api(audio_file, prompt_text, tar_text, tar_lang):
    if hasattr(audio_file, "seek"):
        audio_file.seek(0)
    
    try:
        files = {
            "prompt_audio": (
                audio_file.name, 
                audio_file, 
                "audio/wav"
            )
        }
        data = {
            "tar_text": tar_text,       # 目标内容 (翻译后的文本)
            "tar_lang": tar_lang,
            "prompt_text": prompt_text # 参考内容 (源语音对应的文本，用于克隆音色)
        }
        
        # 发送请求
        response = requests.post(TTS_API, files=files, data=data)
        
        if response.status_code == 200:
            return response.content  # 直接返回二进制音频数据
        else:
            st.error(f"TTS 服务端错误: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"TTS 连接异常: {e}")
        return None
    

# 主页面
io_column, detail_column = st.columns([7,3],width="stretch")

# 状态检测区
with detail_column:
    st.subheader("⚙️ 处理状态监控")
    status_container = st.container() # 专门用于显示 spinner 和状态文字
    
    performance_container = st.container()
    with performance_container:
        st.write("#### ASR 模型性能")
        if st.session_state.get("asr_performance"):
            asr_performance = st.session_state.asr_performance
            c1, c2 = st.columns(2)
            # 3. 使用 st.metric 显示指标
            with c1:
                st.metric(
                    label="音频时长", 
                    value=f"{asr_performance['audio_time']:.2f} s",
                    help="输入语音的时长"
                )
            with c2:
                st.metric(
                    label="推理耗时", 
                    value=f"{asr_performance['inference_time']:.2f} s",
                    help="模型识别这段语音所花费的时间"
                )
            c3, c4 = st.columns(2)
            with c3:
                rtf_val = asr_performance['RTF']
                st.metric(
                    label="RTF (实时率)", 
                    value=f"{rtf_val:.4f}",
                    delta="实时" if rtf_val < 1.0 else "非实时",
                    delta_color="inverse", # 如果 < 1 显示绿色(good)，否则红色
                    help="RTF = 推理耗时 / 音频时长。小于 1 表示处理速度快于说话速度"
                )
        else:
            st.info("等待 ASR 任务执行...")
        
        st.divider()
        st.write("#### MT 模型性能")
        if st.session_state.get("mt_performance"):
            mt_data = st.session_state.mt_performance
            
            # 第一行：显示核心性能指标 (时间 & 速度)
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label="推理耗时", 
                    value=f"{mt_data['inference_time']:.2f} s",
                    help="模型生成完整翻译所花费的总时间"
                )
            with m2:
                tps_val = mt_data['token_per_sec']
                st.metric(
                    label="TPS (生成速度)", 
                    value=f"{tps_val:.1f}",
                    delta=f"{tps_val:.1f} tok/s", # 把单位放在 delta 里显示更酷，或者直接放在 value 里也可以
                    delta_color="normal",         # 灰色 delta，表示这是补充信息
                    help="Tokens Per Second: 每秒生成的 Token 数量，越高越快"
                )
                
            # 第二行：显示数据量指标 (输入 & 输出)
            m3, m4 = st.columns(2)
            with m3:
                st.metric(
                    label="输入字符长度", 
                    value=mt_data['input_length'],
                    help="源文本的字符数量"
                )
            with m4:
                st.metric(
                    label="生成 Token 数", 
                    value=mt_data['token_length'],
                    help="模型输出的 Tokens 总数"
                )   
        
        else:
            st.info("等待 MT 任务执行...")
    
    st.subheader("⚙️ 原始数据")
    with st.expander("查看原始数据"):
        st.write(st.session_state)
    
# 输入输出区
with io_column:
    language_select_area = st.container()
    with language_select_area:
        src_column, emoji_column, tar_column = st.columns([4,2,4],width="stretch")
        # 源语言设置
        with src_column:
            src_lang = st.selectbox(
                label="源语言选择", 
                options=language_select,
                key="src_lang"
                )
        with emoji_column:
            st.markdown("<p style='text-align: center'>➡️</p>", unsafe_allow_html=True)
        # 目标语言设置
        with tar_column:
            tar_lang = st.selectbox(
                label="目标语言选择", 
                options=language_select,
                key="tar_lang"
                )
    
    st.divider()
    
    # 语音区
    audio_area = st.container()
    with audio_area:
        in_audio, emoji_column, out_audio = st.columns([4,2,4],width="stretch")
        with in_audio:
            st.subheader("语音输入")
            # audio_value = st.audio_input(label="")
            audio_value = st.file_uploader("上传音频文件")
            if audio_value is not None and audio_value != st.session_state.last_processed_audio:
                with detail_column:
                    with st.spinner("正在进行语音识别"):
                        recognized_text = call_asr_api(audio_value, src_lang)
                        if recognized_text is not None:
                            st.session_state.recognized_text = recognized_text
                            st.session_state.last_processed_audio = audio_value
                            st.rerun()
        with emoji_column:
            st.markdown("<p style='text-align: center'>➡️</p>", unsafe_allow_html=True)
        with out_audio:
            st.subheader("翻译语音输出")
            if st.session_state.generated_audio is not None:
                st.audio(st.session_state['generated_audio'], format="audio/wav", autoplay=True)
            else:
                st.info("等待生成语音")

    st.divider()

    # 文本区
    txt_area = st.container()
    with txt_area:
        in_text, emoji_column, out_text = st.columns([4,2,4],width="stretch")
        with in_text:
            st.subheader("识别文本")
            input_text = st.text_area(
                label="",
                height=150,
                key="recognized_text",
                help="可手动编辑输入文本"
            )
            if input_text != st.session_state.recognized_text:
                st.session_state.recognized_text = input_text
        with emoji_column:
            st.markdown("<p style='text-align: center'>➡️</p>", unsafe_allow_html=True)
        with out_text:
            st.subheader("翻译结果")
            # 实时翻译结果显示
            translation_text_area = st.text_area(
                label="",
                value=st.session_state.translated_text,
                height=150,
                disabled=True
            )

    st.divider()

    # 控制按钮
    if st.button(
        "🚀 开始翻译", 
        use_container_width=True,
        type="primary"
    ):
        current_text = st.session_state.recognized_text
        if not current_text:
            st.warning("请先上传语音或输入文本！")
        else:
            # --- 逻辑修改 2: 机器翻译过程在 detail_column ---
            with detail_column:
                with status_container:
                    # 步骤 A: 机器翻译
                    with st.spinner(f"正在进行机器翻译"):
                        trans_result = call_mt_api(
                            current_text, 
                            st.session_state.src_lang, 
                            st.session_state.tar_lang
                        )
                    
                    print(f"trans_result: {trans_result}")
                    
                    if trans_result:
                        st.session_state.translated_text = trans_result
                        st.success(f"机器翻译完成")
                        
                        # 步骤 B: 语音生成 (仅在翻译成功后执行)
                        # --- 逻辑修改 3: 语音生成过程在 detail_column ---
                        with st.spinner(f"正在克隆声音并生成语音"):
                            # 注意：根据你的逻辑，TTS需要用到原本的音频作为 prompt
                            # 确保 file 指针回到开头
                            if st.session_state.last_processed_audio:
                                st.session_state.last_processed_audio.seek(0)
                                
                            audio_data = call_tts_api(
                                tar_text=trans_result,
                                tar_lang=st.session_state.tar_lang,
                                prompt_text=current_text,
                                audio_file=st.session_state.last_processed_audio
                            )
                        
                        if audio_data:
                            st.session_state.generated_audio = audio_data
                            st.success("语音生成完成")
                            st.rerun()
                    else:
                        st.error("翻译失败，终止流程")

        
    # 清空按钮
    if st.button(
        "🗑️ 清空所有", 
        use_container_width=True,
        type="secondary"
    ):
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

