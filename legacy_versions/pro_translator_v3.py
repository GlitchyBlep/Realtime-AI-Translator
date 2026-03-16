"""
实时同传翻译器 V3.0 - 热重载完整版
功能：实时监听系统音频/麦克风 → 智能断句 → 上下文隔离翻译 → 左右分栏显示
架构：多线程队列 + 热重载 + 防吞字机制
使用方法：python pro_translator_v3.py
"""

import sys
import os
import time
import json
import re
import threading
import queue
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# ============== 第一部分：导入所有库 ==============
try:
    # 音频处理
    import sounddevice as sd
    import numpy as np
    
    # 语音识别
    from faster_whisper import WhisperModel
    
    # 大模型翻译
    import ollama
    
    # GUI界面
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                                 QPushButton, QComboBox, QVBoxLayout, QHBoxLayout,
                                 QGroupBox, QTextEdit, QFileDialog, QLineEdit,
                                 QMessageBox, QSystemTrayIcon, QMenu, QAction,
                                 QSplitter, QFrame)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QPoint
    from PyQt5.QtGui import QIcon, QFont, QTextCursor
    
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    print("请运行: pip install sounddevice numpy faster-whisper ollama PyQt5")
    sys.exit(1)

# ============== 第二部分：系统配置 ==============
@dataclass
class Config:
    """系统配置"""
    # 音频设置
    SAMPLE_RATE: int = 16000  # Whisper需要的采样率
    CHUNK_DURATION: float = 3.0  # 每次处理的音频时长（秒）
    CHUNK_SIZE: int = int(SAMPLE_RATE * CHUNK_DURATION)  # 每个音频块的大小
    
    # 队列设置
    AUDIO_QUEUE_SIZE: int = 10  # 音频队列最大大小
    TEXT_QUEUE_SIZE: int = 9999  # 文本队列最大大小
    
    # 模型设置
    WHISPER_MODEL_SIZE: str = "small"  # tiny, base, small, medium
    WHISPER_MODEL_DEVICE: str = "cpu"  # macOS用cpu
    WHISPER_MODEL_COMPUTE_TYPE: str = "int8"  # 量化减少内存
    
    TRANSLATION_MODEL: str = "qwen3.5:9b"  # 使用标准模型名称
    
    # 界面设置
    FLOATING_WINDOW_WIDTH: int = 900  # 窗口宽度（屏幕1/3）
    FLOATING_WINDOW_HEIGHT: int = 300  # 固定高度
    FONT_SIZE: int = 18  # 字体大小
    OPACITY: float = 0.9  # 窗口透明度
    MAX_HISTORY_LINES: int = 50  # 每侧最大历史行数
    
    # 语言支持 - 升级点1：None表示自动检测
    LANGUAGES = {
        "自动检测": None,
        "中文": "zh",
        "英文": "en",
        "日文": "ja",
        "韩文": "ko",
        "法文": "fr",
        "西班牙文": "es",
        "葡萄牙文": "pt",
        "德文": "de",
        "意大利文": "it",
        "俄文": "ru",
        "越南文": "vi",
        "泰文": "th",
        "阿拉伯文": "ar",
        "印尼文": "id",
        "马来文": "ms",
        "土耳其文": "tr",
        "荷兰文": "nl",
        "波兰文": "pl",
        "捷克文": "cs",
        "瑞典文": "sv",
        "罗马尼亚文": "ro",
        "印地文": "hi",
        "孟加拉文": "bn",
        "波斯文": "fa",
        "乌尔都文": "ur",
        "希腊文": "el",
        "芬兰文": "fi",
        "丹麦文": "da",
        "匈牙利文": "hu"
    }
    
    # 句末标点（用于智能断句）
    SENTENCE_ENDINGS = ['.', '?', '!', '。', '？', '！', ';', '；']
    
    # 文件存储
    OUTPUT_DIR: Path = Path.home() / "Documents" / "实时翻译记录"
    
    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== 第三部分：多线程队列架构 ==============
class AudioRecorder(QThread):
    """音频录制线程 - 独立线程录制音频"""
    
    audio_ready = pyqtSignal(np.ndarray)  # 音频数据信号
    
    def __init__(self, config: Config, device_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.device_id = device_id
        self.is_running = False
        self.audio_buffer = []
        self.buffer_size = 0
        self.target_buffer_size = config.CHUNK_SIZE
        
    def run(self):
        """运行音频录制线程"""
        self.is_running = True
        
        def audio_callback(indata, frames, time_info, status):
            """音频回调函数"""
            if status:
                print(f"音频状态: {status}")
            
            if self.is_running:
                # 添加音频数据到缓冲区
                audio_data = indata.copy()
                self.audio_buffer.append(audio_data)
                self.buffer_size += len(audio_data)
                
                # 当缓冲区达到目标大小时发送
                if self.buffer_size >= self.target_buffer_size:
                    # 合并缓冲区数据
                    full_audio = np.concatenate(self.audio_buffer, axis=0)[:self.target_buffer_size]
                    
                    # 发送音频数据
                    self.audio_ready.emit(full_audio)
                    
                    # 重置缓冲区（保留剩余部分）
                    remaining = self.buffer_size - self.target_buffer_size
                    if remaining > 0:
                        self.audio_buffer = [np.concatenate(self.audio_buffer, axis=0)[self.target_buffer_size:]]
                        self.buffer_size = remaining
                    else:
                        self.audio_buffer = []
                        self.buffer_size = 0
        
        try:
            # 启动音频流
            with sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.config.SAMPLE_RATE,
                callback=audio_callback,
                blocksize=int(self.config.SAMPLE_RATE * 0.1),  # 100ms块
                latency='low'
            ):
                print(f"✅ 音频录制线程启动，设备ID: {self.device_id}")
                # 保持线程运行
                while self.is_running:
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"❌ 音频录制失败: {e}")
    
    def stop(self):
        """停止音频录制"""
        self.is_running = False
        print("🛑 音频录制线程停止")

class SpeechRecognizer(QThread):
    """语音识别线程 - 独立线程处理语音识别"""
    
    text_recognized = pyqtSignal(str)  # 识别文本信号
    status_update = pyqtSignal(str)     # 状态更新信号
    
    def __init__(self, config: Config, audio_queue: queue.Queue, session_id: str):
        super().__init__()
        self.config = config
        self.audio_queue = audio_queue
        self.session_id = session_id
        self.is_running = False
        self.language = None  # 初始化为None
        self.model = None
        
    def initialize_model(self):
        """初始化Whisper模型"""
        try:
            self.model = WhisperModel(
                self.config.WHISPER_MODEL_SIZE,
                device=self.config.WHISPER_MODEL_DEVICE,
                compute_type=self.config.WHISPER_MODEL_COMPUTE_TYPE,
                cpu_threads=4
            )
            print(f"[{self.session_id[:8]}] ✅ Whisper模型加载成功")
        except Exception as e:
            print(f"[{self.session_id[:8]}] ❌ Whisper模型加载失败: {e}")
            raise
    
    def set_language(self, language: str):
        """设置识别语言 - 升级点1：直接赋值"""
        self.language = language  # 可以是None或语言代码
    
    def run(self):
        """运行语音识别线程"""
        self.is_running = True
        
        # 初始化模型
        self.initialize_model()
        self.status_update.emit("语音识别就绪")
        
        while self.is_running:
            try:
                # 从队列获取音频数据（带超时）
                audio_data = self.audio_queue.get(timeout=0.1)
                
                if audio_data is None:  # 停止信号
                    break
                    
                # 语音识别
                self.status_update.emit("识别中...")
                
                # 保存为临时文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    # 保存为WAV文件
                    import scipy.io.wavfile
                    scipy.io.wavfile.write(tmp_file.name, self.config.SAMPLE_RATE, audio_data)
                    
                    # 进行转录
                    segments, _ = self.model.transcribe(
                        tmp_file.name,
                        language=self.language,  # None表示自动检测
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    # 合并所有片段
                    text = " ".join([segment.text for segment in segments]).strip()
                    
                    # 删除临时文件
                    os.unlink(tmp_file.name)
                
                if text:
                    self.text_recognized.emit(text)
                    self.status_update.emit("识别完成")
                else:
                    self.status_update.emit("等待语音...")
                    
            except queue.Empty:
                continue  # 队列为空，继续等待
            except Exception as e:
                print(f"[{self.session_id[:8]}] ❌ 语音识别错误: {e}")
                self.status_update.emit(f"识别错误: {str(e)[:30]}")
    
    def stop(self):
        """停止语音识别"""
        self.is_running = False
        self.status_update.emit("语音识别停止")

class StreamTranslator(QThread):
    """流式翻译线程 - 独立线程处理翻译"""
    
    translation_chunk = pyqtSignal(str)  # 翻译片段信号
    translation_complete = pyqtSignal(str)  # 完整翻译信号
    status_update = pyqtSignal(str)     # 状态更新信号
    
    def __init__(self, config: Config, text_queue: queue.Queue, session_id: str):
        super().__init__()
        self.config = config
        self.text_queue = text_queue
        self.session_id = session_id
        self.is_running = False
        self.source_lang = "自动检测"
        self.target_lang = "中文"
        self.scene_context = ""  # 场景/领域上下文
        self.client = None
        self.translation_history = []
        
        # 升级点2：滑动窗口记忆
        self.last_sentence = {
            "original": "",
            "translation": "",
            "timestamp": ""
        }
        
        # 语言显示名称映射
        self.lang_display_map = {
            "自动检测": "自动检测",
            "中文": "中文",
            "英文": "英文",
            "日文": "日文",
            "韩文": "韩文",
            "法文": "法文",
            "西班牙文": "西班牙文",
            "葡萄牙文": "葡萄牙文",
            "德文": "德文",
            "意大利文": "意大利文",
            "俄文": "俄文",
            "越南文": "越南文",
            "泰文": "泰文",
            "阿拉伯文": "阿拉伯文",
            "印尼文": "印尼文",
            "马来文": "马来文",
            "土耳其文": "土耳其文",
            "荷兰文": "荷兰文",
            "波兰文": "波兰文",
            "捷克文": "捷克文",
            "瑞典文": "瑞典文",
            "罗马尼亚文": "罗马尼亚文",
            "印地文": "印地文",
            "孟加拉文": "孟加拉文",
            "波斯文": "波斯文",
            "乌尔都文": "乌尔都文",
            "希腊文": "希腊文",
            "芬兰文": "芬兰文",
            "丹麦文": "丹麦文",
            "匈牙利文": "匈牙利文"
        }
    
    def set_languages(self, source_lang: str, target_lang: str, scene_context: str = ""):
        """设置翻译语言和场景上下文 - 升级点2：清空记忆"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.scene_context = scene_context.strip()
        
        # 升级点2：清空历史记录和滑动窗口
        self.translation_history = []
        self.last_sentence = {
            "original": "",
            "translation": "",
            "timestamp": ""
        }
        print(f"[{self.session_id[:8]}] ✅ 语言设置已更新: {source_lang} → {target_lang} | 场景: {scene_context}")
    
    def initialize_client(self):
        """初始化Ollama客户端"""
        try:
            self.client = ollama.Client()
            # 测试连接
            self.client.list()
            print(f"[{self.session_id[:8]}] ✅ Ollama客户端初始化成功")
        except Exception as e:
            print(f"[{self.session_id[:8]}] ❌ Ollama连接失败: {e}")
            raise
    
    def build_prompt(self, text: str) -> str:
        """构建翻译提示词 - 严格上下文隔离"""
        # 获取语言显示名称
        source_lang_display = self.lang_display_map.get(self.source_lang, self.source_lang)
        target_lang_display = self.lang_display_map.get(self.target_lang, self.target_lang)
        
        # 构建场景上下文部分
        scene_part = f"当前场景/领域：【{self.scene_context}】\n" if self.scene_context else ""
        
        # 构建背景参考部分（滑动窗口记忆）
        background_part = ""
        if self.last_sentence["original"] and self.last_sentence["translation"]:
            background_part = f"""=== 背景参考（仅用于辅助理解代词、时态和语境，切勿将以下内容混入当前翻译） ===
前文原文：{self.last_sentence["original"]}
前文译文：{self.last_sentence["translation"]}
"""
        
        # 构建完整提示词
        prompt = f"""你是一个顶级的同传翻译专家。
{scene_part}{background_part}=== 当前翻译任务（你的唯一目标） ===
请精准、流畅、地道地翻译以下【{source_lang_display}】，译为【{target_lang_display}】。
绝对不要重复原文，不要加解释，不要擅自脑补或延续前文剧情，只翻译当前给出的句子：

当前原文：{text}
当前译文："""
        
        return prompt
    
    def run(self):
        """运行翻译线程"""
        self.is_running = True
        
        # 初始化Ollama客户端
        try:
            self.initialize_client()
        except Exception as e:
            self.status_update.emit(f"Ollama错误: {str(e)[:30]}")
            return
        
        self.status_update.emit("翻译就绪")
        
        while self.is_running:
            try:
                # 从队列获取文本（带超时）
                text_item = self.text_queue.get(timeout=0.1)
                
                if text_item is None:  # 停止信号
                    break
                    
                # 提取文本和时间戳
                text, timestamp = text_item
                
                if not text.strip():
                    continue
                
                # 开始翻译
                self.status_update.emit("翻译中...")
                
                # 构建提示词
                prompt = self.build_prompt(text)
                
                # 流式翻译 - think=False关闭思考过程
                full_translation = ""
                try:
                    # 调用Ollama进行流式生成
                    response = self.client.generate(
                        model=self.config.TRANSLATION_MODEL,
                        prompt=prompt,
                        stream=True,
                        think=False,  # 强制关闭思考过程
                        options={
                            "temperature": 0.1,  # 低温度保证准确性
                            "top_p": 0.9,
                            "top_k": 40
                        }
                    )
                    
                    # 处理流式响应
                    for chunk in response:
                        if 'response' in chunk:
                            chunk_text = chunk['response']
                            full_translation += chunk_text
                            # 实时发送翻译片段
                            self.translation_chunk.emit(full_translation)
                            
                    # 翻译完成
                    if full_translation.strip():
                        # 保存到历史记录
                        self.translation_history.append({
                            "timestamp": timestamp,
                            "source": text,
                            "translation": full_translation.strip(),
                            "source_lang": self.source_lang,
                            "target_lang": self.target_lang,
                            "scene": self.scene_context
                        })
                        
                        # 更新滑动窗口记忆
                        self.last_sentence = {
                            "original": text,
                            "translation": full_translation.strip(),
                            "timestamp": timestamp
                        }
                        
                        # 发送完整翻译信号
                        self.translation_complete.emit(full_translation.strip())
                        self.status_update.emit("翻译完成")
                    else:
                        self.status_update.emit("翻译为空")
                        
                except Exception as e:
                    print(f"[{self.session_id[:8]}] ❌ 翻译错误: {e}")
                    # 发送错误信息
                    self.translation_complete.emit(f"[翻译错误: {str(e)[:30]}]")
                    self.status_update.emit(f"翻译错误: {str(e)[:30]}")
                    
            except queue.Empty:
                continue  # 队列为空，继续等待
            except Exception as e:
                print(f"[{self.session_id[:8]}] ❌ 翻译线程错误: {e}")
                self.status_update.emit(f"线程错误: {str(e)[:30]}")
    
    def stop(self):
        """停止翻译"""
        self.is_running = False
        self.status_update.emit("翻译停止")
    
    def get_translation_history(self):
        """获取翻译历史记录"""
        return self.translation_history
    
    def clear_history(self):
        """清空历史记录和滑动窗口"""
        self.translation_history = []
        self.last_sentence = {
            "original": "",
            "translation": "",
            "timestamp": ""
        }

class TranslationPipeline(QThread):
    """翻译流水线管理器 - 协调所有线程"""
    
    # 信号定义
    original_text_signal = pyqtSignal(str)  # 原文信号
    translation_chunk_signal = pyqtSignal(str)  # 翻译片段信号
    translation_complete_signal = pyqtSignal(str)  # 完整翻译信号
    status_signal = pyqtSignal(str)  # 状态信号
    error_signal = pyqtSignal(str)   # 错误信号
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.is_running = False
        self.current_session_id = ""  # 当前会话ID
        
        # 创建队列
        self.audio_queue = queue.Queue(maxsize=config.AUDIO_QUEUE_SIZE)
        self.text_queue = queue.Queue(maxsize=config.TEXT_QUEUE_SIZE)
        
        # 初始化组件
        self.audio_recorder = None
        self.speech_recognizer = None
        self.translator = None
        
        # 数据存储
        self.text_buffer = ""  # 文本缓冲区
        self.subtitles = []  # 字幕历史
        
        # 语言设置
        self.source_lang = "自动检测"
        self.target_lang = "中文"
        self.scene_context = ""  # 场景/领域上下文
        
        # 音频设备
        self.audio_device_id = None
    
    def audio_callback(self, audio_data: np.ndarray):
        """音频回调函数 - 将音频数据放入队列"""
        if self.is_running and not self.audio_queue.full():
            self.audio_queue.put(audio_data)
    
    def start_pipeline(self, device_id: Optional[int] = None):
        """启动整个流水线"""
        # 生成新的会话ID
        self.current_session_id = str(uuid.uuid4())
        print(f"[{self.current_session_id[:8]}] 🚀 启动新翻译会话")
        
        self.is_running = True
        
        # 1. 启动音频录制线程
        self.audio_recorder = AudioRecorder(self.config, device_id)
        self.audio_recorder.audio_ready.connect(self.audio_callback)
        self.audio_recorder.start()
        
        # 2. 启动语音识别线程
        self.speech_recognizer = SpeechRecognizer(self.config, self.audio_queue, self.current_session_id)
        self.speech_recognizer.text_recognized.connect(self.handle_recognized_text)
        self.speech_recognizer.status_update.connect(self.status_signal)
        self.speech_recognizer.set_language(self.config.LANGUAGES.get(self.source_lang))
        self.speech_recognizer.start()
        
        # 3. 启动翻译线程
        self.translator = StreamTranslator(self.config, self.text_queue, self.current_session_id)
        self.translator.translation_chunk.connect(self.handle_translation_chunk)
        self.translator.translation_complete.connect(self.handle_translation_complete)
        self.translator.status_update.connect(self.status_signal)
        self.translator.set_languages(self.source_lang, self.target_lang, self.scene_context)
        self.translator.start()
        
        self.status_signal.emit("流水线启动")
    
    def handle_recognized_text(self, text: str):
        """处理识别到的文本 - 升级点4：防吞字机制"""
        if not text.strip():
            return
            
        # 添加到缓冲区
        self.text_buffer += " " + text.strip()
        
        # 升级点4：强制防吞字机制
        force_send = False
        if len(self.text_buffer) > 100:  # 超过100个字符强制发送
            force_send = True
            print(f"[{self.current_session_id[:8]}] ⚠️ 防吞字机制触发: buffer长度={len(self.text_buffer)}")
        
        # 检查缓冲区是否包含完整句子
        sentences = self.split_into_sentences(self.text_buffer)
        
        if len(sentences) > 1 or force_send:
            # 处理所有完整句子
            if force_send:
                # 强制将整个缓冲区作为一个句子
                sentence = self.text_buffer.strip()
                if sentence:
                    # 发送原文信号
                    self.original_text_signal.emit(sentence)
                    
                    # 将文本放入翻译队列
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if not self.text_queue.full():
                        self.text_queue.put((sentence, timestamp))
                    
                    # 清空缓冲区
                    self.text_buffer = ""
            else:
                # 正常处理完整句子
                for i in range(len(sentences) - 1):
                    sentence = sentences[i].strip()
                    if sentence:  # 非空句子
                        # 发送原文信号
                        self.original_text_signal.emit(sentence)
                        
                        # 将文本放入翻译队列
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        if not self.text_queue.full():
                            self.text_queue.put((sentence, timestamp))
                
                # 保留最后一个未完成的句子
                self.text_buffer = sentences[-1] if sentences else ""
        else:
            # 没有完整句子，暂时不发送
            pass
    
    def split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子（基于标点符号）"""
        if not text:
            return []
        
        # 使用正则表达式分割句子
        # 匹配句末标点（包括中文和英文）
        pattern = r'([.!?。！？]+)'
        parts = re.split(pattern, text)
        
        sentences = []
        current_sentence = ""
        
        for i in range(0, len(parts)-1, 2):
            sentence_part = parts[i]
            punctuation = parts[i+1] if i+1 < len(parts) else ""
            
            current_sentence += sentence_part + punctuation
            
            # 如果当前部分以句末标点结束，则作为一个完整句子
            if punctuation and any(p in punctuation for p in self.config.SENTENCE_ENDINGS):
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # 添加最后部分
        if current_sentence:
            sentences.append(current_sentence.strip())
        elif len(parts) % 2 == 1:  # 奇数个部分，最后一个部分是未完成的句子
            sentences.append(parts[-1].strip())
        
        return sentences
    
    def handle_translation_chunk(self, chunk: str):
        """处理翻译片段"""
        # 会话检查：确保当前会话仍然活跃
        if not self.is_running:
            return
        self.translation_chunk_signal.emit(chunk)
    
    def handle_translation_complete(self, translation: str):
        """处理翻译完成的文本"""
        # 会话检查：确保当前会话仍然活跃
        if not self.is_running:
            return
            
        if translation and translation.strip():
            self.translation_complete_signal.emit(translation)
    
    def set_languages(self, source_lang: str, target_lang: str, scene_context: str = ""):
        """设置翻译语言和场景上下文 - 升级点2：清空缓冲区"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.scene_context = scene_context
        
        # 升级点2：清空文本缓冲区
        self.text_buffer = ""
        
        # 更新语音识别语言
        if self.speech_recognizer:
            self.speech_recognizer.set_language(self.config.LANGUAGES.get(source_lang))
        
        # 更新翻译语言和场景
        if self.translator:
            self.translator.set_languages(source_lang, target_lang, scene_context)
    
    def set_audio_device(self, device_id: int):
        """设置音频设备"""
        self.audio_device_id = device_id
    
    def switch_audio_device(self, device_id: int):
        """切换音频设备 - 升级点3：热重载音频录制"""
        if not self.is_running:
            return
        
        print(f"[{self.current_session_id[:8]}] 🔄 切换音频设备到: {device_id}")
        
        # 停止当前的音频录制线程
        if self.audio_recorder:
            self.audio_recorder.stop()
            self.audio_recorder.wait()
        
        # 启动新的音频录制线程
        self.audio_recorder = AudioRecorder(self.config, device_id)
        self.audio_recorder.audio_ready.connect(self.audio_callback)
        self.audio_recorder.start()
        
        self.status_signal.emit(f"音频设备已切换")
    
    def run(self):
        """运行流水线"""
        self.start_pipeline(self.audio_device_id)
        
        # 保持线程运行
        while self.is_running:
            time.sleep(0.1)
    
    def stop(self):
        """停止流水线"""
        print(f"[{self.current_session_id[:8]}] 🛑 停止翻译会话")
        self.is_running = False
        
        # 清空所有队列
        self.clear_queues()
        
        # 发送停止信号到队列
        self.audio_queue.put(None)
        self.text_queue.put(None)
        
        # 停止所有线程
        if self.audio_recorder:
            self.audio_recorder.stop()
            self.audio_recorder.wait()
        
        if self.speech_recognizer:
            self.speech_recognizer.stop()
            self.speech_recognizer.wait()
        
        if self.translator:
            self.translator.stop()
            self.translator.wait()
        
        # 重置会话ID
        self.current_session_id = ""
        
        self.status_signal.emit("流水线停止")
    
    def clear_queues(self):
        """清空所有队列"""
        # 清空音频队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # 清空文本队列
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break
        
        # 清空缓冲区
        self.text_buffer = ""
        if self.current_session_id:
            print(f"[{self.current_session_id[:8]}] ✅ 所有队列已清空")
    
    def save_subtitles(self, file_path: str):
        """保存字幕到文件"""
        try:
            if self.translator:
                history = self.translator.get_translation_history()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"实时翻译记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"源语言: {self.source_lang} → 目标语言: {self.target_lang}\n")
                    if self.scene_context:
                        f.write(f"场景/领域: {self.scene_context}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for i, item in enumerate(history, 1):
                        f.write(f"{i}. [{item['timestamp']}]\n")
                        f.write(f"   原文: {item['source']}\n")
                        f.write(f"   译文: {item['translation']}\n")
                        if item.get('scene'):
                            f.write(f"   场景: {item['scene']}\n")
                        f.write("\n")
                
                return True
            else:
                return False
        except Exception as e:
            print(f"❌ 保存字幕失败: {e}")
            return False
    
    def clear_subtitles(self):
        """清空字幕记录"""
        self.subtitles = []
        self.text_buffer = ""
        if self.translator:
            self.translator.clear_history()

# ============== 第四部分：GUI界面 ==============
class FloatingWindow(QWidget):
    """悬浮字幕窗口 - 左右分栏设计"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.init_ui()
        self.setup_window_properties()
        
        # 历史记录
        self.original_history = []
        self.translation_history = []
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("实时翻译字幕")
        
        # 设置窗口样式
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 200);
                color: white;
                border: none;
                border-radius: 10px;
            }
            QTextEdit {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 5px;
                padding: 10px;
                font-size: 18px;
                font-family: 'PingFang SC', 'Helvetica Neue', Arial, sans-serif;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton {
                background-color: rgba(70, 130, 180, 150);
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 12px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 200);
            }
        """)
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # 标题栏
        title_layout = QHBoxLayout()
        title_label = QLabel("🎬 实时同传翻译 V3.0")
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        
        # 关闭按钮
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 50, 50, 150);
                color: white;
                border: none;
                border-radius: 15px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(255, 50, 50, 200);
            }
        """)
        close_btn.clicked.connect(self.hide)
        title_layout.addWidget(close_btn)
        
        main_layout.addLayout(title_layout)
        
        # 分割器布局（左右分栏）
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：原文显示框
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)
        
        original_label = QLabel("📝 原文")
        original_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(original_label)
        
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        self.original_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.original_text.setMaximumHeight(self.config.FLOATING_WINDOW_HEIGHT - 100)
        left_layout.addWidget(self.original_text)
        
        splitter.addWidget(left_widget)
        
        # 右侧：译文显示框
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)
        
        translation_label = QLabel("🌐 译文")
        translation_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(translation_label)
        
        self.translation_text = QTextEdit()
        self.translation_text.setReadOnly(True)
        self.translation_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.translation_text.setMaximumHeight(self.config.FLOATING_WINDOW_HEIGHT - 100)
        right_layout.addWidget(self.translation_text)
        
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setSizes([450, 450])
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #90EE90; font-size: 14px;")
        status_layout.addWidget(self.status_label)
        
        # 滚动按钮
        scroll_btn = QPushButton("🔄 同步滚动")
        scroll_btn.setFixedHeight(25)
        scroll_btn.clicked.connect(self.sync_scroll)
        status_layout.addWidget(scroll_btn)
        
        # 清空按钮
        clear_btn = QPushButton("🗑️ 清空")
        clear_btn.setFixedHeight(25)
        clear_btn.clicked.connect(self.clear_all)
        status_layout.addWidget(clear_btn)
        
        main_layout.addLayout(status_layout)
        
        self.setLayout(main_layout)
    
    def setup_window_properties(self):
        """设置窗口属性"""
        # 窗口置顶
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint | 
            Qt.Tool |
            Qt.X11BypassWindowManagerHint
        )
        
        # 窗口大小
        self.resize(self.config.FLOATING_WINDOW_WIDTH, 
                   self.config.FLOATING_WINDOW_HEIGHT)
        
        # 窗口透明度
        self.setWindowOpacity(self.config.OPACITY)
        
        # 初始位置（屏幕右下角）
        screen_geometry = QApplication.desktop().screenGeometry()
        self.move(
            screen_geometry.width() - self.width() - 20,
            screen_geometry.height() - self.height() - 100
        )
    
    def mousePressEvent(self, event):
        """鼠标点击事件 - 支持拖动"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 拖动窗口"""
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
    
    def update_subtitle(self, original_text: str):
        """更新原文显示"""
        if not original_text.strip():
            return
            
        # 添加到历史记录
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_text = f"[{timestamp}] {original_text}"
        self.original_history.append(formatted_text)
        
        # 限制历史记录长度
        if len(self.original_history) > self.config.MAX_HISTORY_LINES:
            self.original_history.pop(0)
        
        # 更新显示
        self.original_text.setText("\n\n".join(self.original_history))
        
        # 自动滚动到底部
        self.original_text.moveCursor(QTextCursor.End)
    
    def update_translation_chunk(self, chunk: str):
        """更新流式翻译片段"""
        if not chunk:
            return
            
        # 获取当前译文文本
        current_text = self.translation_text.toPlainText()
        lines = current_text.split("\n")
        
        if lines and lines[-1].startswith("[翻译中]"):
            # 更新最后一行
            lines[-1] = f"[翻译中] {chunk}"
        else:
            # 添加新行
            timestamp = datetime.now().strftime("%H:%M:%S")
            lines.append(f"[翻译中] {chunk}")
        
        # 限制历史记录长度
        if len(lines) > self.config.MAX_HISTORY_LINES:
            lines = lines[-self.config.MAX_HISTORY_LINES:]
        
        self.translation_history = lines.copy()
        self.translation_text.setText("\n".join(lines))
        self.translation_text.moveCursor(QTextCursor.End)
    
    def update_translation_complete(self, translation: str):
        """更新完整翻译"""
        if not translation.strip() or translation.startswith("[翻译错误"):
            return
            
        # 获取当前译文文本
        current_text = self.translation_text.toPlainText()
        lines = current_text.split("\n")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 查找并替换最后一行"[翻译中]"的行
        for i in range(len(lines)-1, -1, -1):
            if "[翻译中]" in lines[i]:
                lines[i] = f"[{timestamp}] {translation}"
                break
        else:
            # 如果没有找到，添加新行
            lines.append(f"[{timestamp}] {translation}")
        
        # 限制历史记录长度
        if len(lines) > self.config.MAX_HISTORY_LINES:
            lines = lines[-self.config.MAX_HISTORY_LINES:]
        
        self.translation_history = lines.copy()
        self.translation_text.setText("\n".join(lines))
        self.translation_text.moveCursor(QTextCursor.End)
        
        # 更新状态
        self.status_label.setText(f"✓ {timestamp} 翻译完成")
    
    def update_status(self, status: str):
        """更新状态显示"""
        self.status_label.setText(status)
    
    def sync_scroll(self):
        """同步滚动原文和译文"""
        original_scrollbar = self.original_text.verticalScrollBar()
        translation_scrollbar = self.translation_text.verticalScrollBar()
        
        # 获取原文滚动位置
        original_value = original_scrollbar.value()
        original_max = original_scrollbar.maximum()
        
        # 计算比例并设置译文滚动位置
        if original_max > 0:
            ratio = original_value / original_max
            translation_max = translation_scrollbar.maximum()
            translation_scrollbar.setValue(int(ratio * translation_max))
    
    def clear_all(self):
        """清空所有内容"""
        self.original_history = []
        self.translation_history = []
        self.original_text.clear()
        self.translation_text.clear()
        self.status_label.setText("已清空")

class ControlWindow(QMainWindow):
    """控制窗口"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.init_ui()
        
        # 初始化组件
        self.floating_window = FloatingWindow(config)
        self.translation_pipeline = TranslationPipeline(config)
        
        # 连接信号
        self.connect_signals()
        
        # 系统托盘
        self.setup_system_tray()
        
        # 显示悬浮窗
        self.floating_window.show()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("专业级同传翻译器 V3.0 - 控制台")
        self.setGeometry(100, 100, 600, 550)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # 1. 语言设置组
        lang_group = QGroupBox("🌐 语言设置")
        lang_layout = QVBoxLayout()
        
        # 源语言
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("源语言:"))
        self.source_lang_combo = QComboBox()
        for lang_name in self.config.LANGUAGES.keys():
            self.source_lang_combo.addItem(lang_name)
        self.source_lang_combo.setCurrentText("英文")
        source_layout.addWidget(self.source_lang_combo)
        lang_layout.addLayout(source_layout)
        
        # 目标语言
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("目标语言:"))
        self.target_lang_combo = QComboBox()
        for lang_name, lang_code in self.config.LANGUAGES.items():
            if lang_name != "自动检测":
                self.target_lang_combo.addItem(lang_name)
        self.target_lang_combo.setCurrentText("中文")
        target_layout.addWidget(self.target_lang_combo)
        lang_layout.addLayout(target_layout)
        
        lang_group.setLayout(lang_layout)
        main_layout.addWidget(lang_group)
        
        # 场景/领域输入框
        scene_group = QGroupBox("🎭 场景/领域设置")
        scene_layout = QVBoxLayout()
        
        self.scene_input = QLineEdit()
        self.scene_input.setPlaceholderText("例如：MIT神经生物学、韩国爱情剧、日常闲聊...")
        self.scene_input.setToolTip("输入翻译场景或领域，AI会针对性优化翻译效果（可选）")
        scene_layout.addWidget(self.scene_input)
        
        scene_note = QLabel("💡 提示：输入场景描述可以让翻译更符合特定领域语境")
        scene_note.setStyleSheet("color: #666; font-size: 12px;")
        scene_note.setWordWrap(True)
        scene_layout.addWidget(scene_note)
        
        scene_group.setLayout(scene_layout)
        main_layout.addWidget(scene_group)
        
        # 2. 音频设置组
        audio_group = QGroupBox("🎤 音频设置")
        audio_layout = QVBoxLayout()
        
        # 音频设备选择
        audio_layout.addWidget(QLabel("音频输入源:"))
        self.audio_device_combo = QComboBox()
        self.refresh_audio_devices()
        audio_layout.addWidget(self.audio_device_combo)
        
        # 音频设备说明
        audio_note = QLabel("💡 建议：")
        audio_note.setStyleSheet("color: #666; font-size: 12px;")
        audio_note.setWordWrap(True)
        audio_note.setText("• 监听系统声音：选择 BlackHole 2ch\n• 麦克风录音：选择 Built-in Microphone")
        audio_layout.addWidget(audio_note)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # 3. 控制按钮组
        control_group = QGroupBox("🕹️ 控制")
        control_layout = QVBoxLayout()
        
        # 开始/停止按钮
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("▶️ 开始翻译")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_button.clicked.connect(self.start_translation)
        
        self.stop_button = QPushButton("⏸️ 停止翻译")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 12px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_button.clicked.connect(self.stop_translation)
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        control_layout.addLayout(button_layout)
        
        # 操作按钮
        operation_layout = QHBoxLayout()
        
        clear_btn = QPushButton("🗑️ 清空字幕")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 8px;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        clear_btn.clicked.connect(self.clear_all)
        operation_layout.addWidget(clear_btn)
        
        self.save_button = QPushButton("💾 保存字幕")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.save_button.clicked.connect(self.save_subtitles)
        self.save_button.setEnabled(False)
        operation_layout.addWidget(self.save_button)
        
        control_layout.addLayout(operation_layout)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # 4. 状态显示组
        status_group = QGroupBox("📊 状态信息")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("状态: 等待开始")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        status_layout.addWidget(self.status_label)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        self.info_text.setStyleSheet("font-size: 12px; background-color: #f5f5f5;")
        self.info_text.setText("就绪，请选择语言和音频源后点击开始。\n\n使用提示：\n1. 确保Ollama服务正在运行\n2. 选择正确的音频输入源\n3. 可输入场景描述优化翻译效果\n4. 点击开始后，悬浮窗会显示实时翻译")
        status_layout.addWidget(self.info_text)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # 底部说明
        footer_label = QLabel("💡 提示：悬浮窗可以拖动，右键系统托盘图标可退出程序")
        footer_label.setStyleSheet("color: #888; font-size: 12px; padding: 10px;")
        footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer_label)
        
        central_widget.setLayout(main_layout)
    
    def refresh_audio_devices(self):
        """刷新音频设备列表"""
        self.audio_device_combo.clear()
        
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # 只显示输入设备
                    device_name = device['name']
                    # 高亮显示BlackHole设备
                    if 'blackhole' in device_name.lower():
                        self.audio_device_combo.addItem(f"{i}: {device_name} (推荐: 系统声音)", i)
                    elif 'built-in' in device_name.lower():
                        self.audio_device_combo.addItem(f"{i}: {device_name} (内置麦克风)", i)
                    else:
                        self.audio_device_combo.addItem(f"{i}: {device_name}", i)
        except Exception as e:
            print(f"❌ 获取音频设备失败: {e}")
            self.audio_device_combo.addItem("默认设备", None)
        
        # 默认选择第一个设备
        if self.audio_device_combo.count() > 0:
            self.audio_device_combo.setCurrentIndex(0)
    
    def connect_signals(self):
        """连接信号与槽 - 升级点3：连接音频设备切换信号"""
        # 翻译流水线信号
        self.translation_pipeline.original_text_signal.connect(
            self.floating_window.update_subtitle
        )
        self.translation_pipeline.translation_chunk_signal.connect(
            self.floating_window.update_translation_chunk
        )
        self.translation_pipeline.translation_complete_signal.connect(
            self.floating_window.update_translation_complete
        )
        self.translation_pipeline.status_signal.connect(self.update_status)
        self.translation_pipeline.error_signal.connect(self.show_error)
        
        # 语言切换信号
        self.source_lang_combo.currentTextChanged.connect(self.on_language_changed)
        self.target_lang_combo.currentTextChanged.connect(self.on_language_changed)
        self.scene_input.textChanged.connect(self.on_scene_changed)
        
        # 升级点3：音频设备切换信号
        self.audio_device_combo.currentIndexChanged.connect(self.on_audio_device_changed)
    
    def on_language_changed(self):
        """语言切换处理"""
        source_lang = self.source_lang_combo.currentText()
        target_lang = self.target_lang_combo.currentText()
        scene_context = self.scene_input.text().strip()
        
        # 更新流水线语言设置
        self.translation_pipeline.set_languages(source_lang, target_lang, scene_context)
        
        # 更新状态显示
        status_text = f"语言设置已更新: {source_lang} → {target_lang}"
        if scene_context:
            status_text += f" | 场景: {scene_context}"
        self.update_status(status_text)
    
    def on_scene_changed(self):
        """场景描述变化处理"""
        # 触发语言更新（包含场景）
        self.on_language_changed()
    
    def on_audio_device_changed(self):
        """音频设备切换处理 - 升级点3：热重载音频录制"""
        if self.audio_device_combo.currentData() is None:
            return
            
        audio_device_index = self.audio_device_combo.currentData()
        device_name = self.audio_device_combo.currentText()
        
        # 如果翻译正在进行中，热切换音频设备
        if self.translation_pipeline.isRunning():
            self.translation_pipeline.switch_audio_device(audio_device_index)
            self.update_status(f"音频设备已切换到: {device_name}")
    
    def update_status(self, status: str):
        """更新状态显示"""
        self.status_label.setText(f"状态: {status}")
        current_info = self.info_text.toPlainText()
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.append(f"[{timestamp}] {status}")
        
        # 保持最后15行
        lines = current_info.split('\n')
        if len(lines) > 15:
            self.info_text.setText('\n'.join(lines[-15:]))
        
        # 更新悬浮窗状态
        self.floating_window.update_status(status)
    
    def show_error(self, error: str):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", error)
        self.update_status(f"错误: {error}")
    
    def start_translation(self):
        """开始翻译"""
        try:
            # 获取设置
            source_lang = self.source_lang_combo.currentText()
            target_lang = self.target_lang_combo.currentText()
            scene_context = self.scene_input.text().strip()
            audio_device_index = self.audio_device_combo.currentData()
            
            # 检查Ollama连接
            try:
                import ollama
                client = ollama.Client()
                models = client.list()
                
                # 稳健的模型名称检查
                model_found = False
                for model_info in models['models']:
                    model_name = model_info.get('model', '')
                    # 检查是否包含qwen3.5:9b（支持不同变体）
                    if 'qwen3.5' in model_name.lower() and '9b' in model_name.lower():
                        model_found = True
                        print(f"✅ 找到兼容模型: {model_name}")
                        break
                
                if not model_found:
                    QMessageBox.warning(self, "警告", 
                        "未找到qwen3.5:9b模型或兼容模型\n请运行: ollama pull qwen3.5:9b")
                    return
                    
            except Exception as e:
                QMessageBox.critical(self, "Ollama错误", 
                    f"无法连接Ollama: {e}\n请确保Ollama服务正在运行。")
                return
            
            # 清空所有缓存和历史记录
            self.floating_window.clear_all()
            self.translation_pipeline.clear_subtitles()
            
            # 设置翻译流水线
            self.translation_pipeline.set_languages(source_lang, target_lang, scene_context)
            self.translation_pipeline.set_audio_device(audio_device_index)
            
            # 启动流水线
            self.translation_pipeline.start()
            
            # 更新按钮状态
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.save_button.setEnabled(False)
            
            status_text = f"翻译已启动: {source_lang} → {target_lang}"
            if scene_context:
                status_text += f" | 场景: {scene_context}"
            self.update_status(status_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动失败: {str(e)}")
    
    def stop_translation(self):
        """停止翻译"""
        # 停止流水线
        self.translation_pipeline.stop()
        
        # 等待流水线完全停止
        self.translation_pipeline.wait()
        
        # 更新按钮状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)
        
        self.update_status("翻译已停止")
        
        # 询问是否保存
        reply = QMessageBox.question(
            self, '保存字幕',
            '是否保存本次翻译的字幕记录？\n\n选择"否"将清空记录（阅后即焚）。',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self.save_subtitles()
        else:
            self.translation_pipeline.clear_subtitles()
            self.floating_window.clear_all()
            self.update_status("字幕已清空（阅后即焚）")
    
    def clear_all(self):
        """清空所有内容"""
        self.floating_window.clear_all()
        self.translation_pipeline.clear_subtitles()
        self.update_status("所有内容已清空")
    
    def save_subtitles(self):
        """保存字幕到文件"""
        # 生成默认文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_lang = self.source_lang_combo.currentText()
        target_lang = self.target_lang_combo.currentText()
        default_name = f"翻译_{source_lang}_{target_lang}_{timestamp}.txt"
        default_path = self.config.OUTPUT_DIR / default_name
        
        # 选择保存位置
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存字幕文件",
            str(default_path),
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if file_path:
            success = self.translation_pipeline.save_subtitles(file_path)
            if success:
                QMessageBox.information(self, "成功", f"字幕已保存到:\n{file_path}")
                self.update_status(f"字幕已保存: {Path(file_path).name}")
            else:
                QMessageBox.critical(self, "错误", "保存失败，可能没有翻译历史记录")
    
    def setup_system_tray(self):
        """设置系统托盘"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        
        self.tray_icon = QSystemTrayIcon(self)
        
        # 创建托盘菜单
        tray_menu = QMenu()
        
        show_action = QAction("显示控制台", self)
        show_action.triggered.connect(self.showNormal)
        tray_menu.addAction(show_action)
        
        hide_action = QAction("隐藏控制台", self)
        hide_action.triggered.connect(self.hide)
        tray_menu.addAction(hide_action)
        
        show_floating_action = QAction("显示悬浮窗", self)
        show_floating_action.triggered.connect(self.floating_window.show)
        tray_menu.addAction(show_floating_action)
        
        hide_floating_action = QAction("隐藏悬浮窗", self)
        hide_floating_action.triggered.connect(self.floating_window.hide)
        tray_menu.addAction(hide_floating_action)
        
        tray_menu.addSeparator()
        
        exit_action = QAction("退出程序", self)
        exit_action.triggered.connect(self.close_all)
        tray_menu.addAction(exit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
    
    def close_all(self):
        """关闭所有窗口并退出"""
        if self.translation_pipeline.isRunning():
            self.translation_pipeline.stop()
            self.translation_pipeline.wait()
        self.floating_window.close()
        QApplication.quit()
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.translation_pipeline.isRunning():
            reply = QMessageBox.question(
                self, '确认退出',
                '翻译正在进行中，是否退出？\n\n退出将停止所有翻译并关闭悬浮窗。',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.close_all()
                event.accept()
            else:
                event.ignore()
        else:
            self.close_all()
            event.accept()

# ============== 第五部分：主函数 ==============
def main():
    """主函数"""
    print("=" * 60)
    print("🎬 专业级实时同传翻译器 V3.0 - 终极热重载版")
    print("=" * 60)
    print("💡 使用说明：")
    print("1. 确保Ollama服务正在运行：ollama serve")
    print("2. 确保已下载模型：ollama pull qwen3.5:9b")
    print("3. 选择源语言和目标语言（支持29种语言）")
    print("4. 可选：输入场景/领域提示词优化翻译")
    print("5. 选择音频输入源（BlackHole监听系统声音）")
    print("6. 点击开始翻译")
    print("7. 运行时可以随时切换语言、场景、音频设备")
    print("8. 悬浮窗会分栏显示原文和译文")
    print("=" * 60)
    
    # 检查Ollama
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        
        # 稳健的模型检查
        model_found = False
        for model_info in models['models']:
            model_name = model_info.get('model', '')
            if 'qwen3.5' in model_name.lower() and '9b' in model_name.lower():
                model_found = True
                print(f"✅ 找到兼容模型: {model_name}")
                break
        
        if not model_found:
            print("⚠️  未找到qwen3.5:9b模型或兼容模型")
            print("请运行: ollama pull qwen3.5:9b")
            print("如果已经安装，请确认模型名称是否正确")
            
    except Exception as e:
        print(f"❌ 无法连接Ollama: {e}")
        print("请确保已安装并运行Ollama：")
        print("1. 访问 https://ollama.ai 下载安装")
        print("2. 新终端窗口运行: ollama serve")
        print("3. 另一个终端运行: ollama pull qwen3.5:9b")
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("专业级同传翻译器 V3.0")
    app.setApplicationDisplayName("专业级同传翻译器 V3.0")
    
    # 创建配置和控制窗口
    config = Config()
    control_window = ControlWindow(config)
    control_window.show()
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()