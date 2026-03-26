"""
🎬 专业级实时同传翻译器 V6.0 - WebRTC VAD重构版
功能：实时监听系统音频/麦克风 → 智能VAD断句 → 零记忆翻译 → 左右分栏显示
架构：多线程队列 + C++底层推理引擎 + WebRTC VAD
引擎：whisper.cpp + llama.cpp（腾讯混元翻译模型）
重构重点：
  1. WebRTC VAD智能语音端点检测（mode=3，500ms静音断句）
  2. 显式内存管理，解决反复启动OOM问题
  3. 极简Prompt适配1.8B小模型
使用方法：WHISPER_CPP_LIB=./libwhisper.dylib python3 v6_1.py
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
import gc
warnings.filterwarnings("ignore")

# ============== 第一部分：导入所有库 ==============
try:
    # 音频处理
    import sounddevice as sd
    import numpy as np
    
    # 语音识别 - whisper.cpp
    from whisper_cpp_python import Whisper
    
    # 大模型翻译 - llama.cpp
    from llama_cpp import Llama
    
    # GUI界面
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                                 QPushButton, QComboBox, QVBoxLayout, QHBoxLayout,
                                 QGroupBox, QTextEdit, QFileDialog, QLineEdit,
                                 QMessageBox, QSystemTrayIcon, QMenu, QAction,
                                 QSplitter, QFrame, QSlider, QDial, QProgressBar)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QPoint
    from PyQt5.QtGui import QIcon, QFont, QTextCursor, QColor
    
    # WebRTC VAD语音活动检测
    import webrtcvad
    
    # 音频格式转换
    import wave
    import struct
    
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    print("请运行: pip install sounddevice numpy whisper-cpp-python llama-cpp-python PyQt5 webrtcvad scipy")
    sys.exit(1)

# ============== 第二部分：系统配置 ==============
@dataclass
class Config:
    """系统配置 - V6.1：WebRTC VAD优化版（增加强制兜底机制）"""
    
    # 音频设置
    SAMPLE_RATE: int = 16000  # Whisper和VAD需要的采样率
    CHANNELS: int = 1  # 单声道
    BIT_DEPTH: int = 16  # 16位深度
    
    # WebRTC VAD设置 - V6.1优化
    VAD_MODE: int = 3  # 0-3，3为最敏感，误报最少
    VAD_FRAME_DURATION_MS: int = 30  # 帧持续时间（毫秒）
    
    # V6.1重构：静音超时改为300ms（追求极速上屏体验）
    VAD_SILENCE_TIMEOUT_MS: int = 300  # 静音持续300ms判定为语句结束
    
    # V6.1新增：强制兜底机制，防止音频块无限增长
    VAD_MAX_SPEECH_MS: int = 5000  # 最大语音持续时间5秒（强制兜底防卡死）
    
    VAD_MIN_SPEECH_MS: int = 100  # 最小语音持续时间（毫秒）
    
    # VAD帧大小计算
    FRAME_SIZE: int = int(SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)  # 每帧的样本数
    
    # 队列设置
    AUDIO_QUEUE_SIZE: int = 10  # 音频队列最大大小
    TEXT_QUEUE_SIZE: int = 9999  # 文本队列最大大小
    
    # C++引擎设置
    WHISPER_MODEL_PATH: str = "models/ggml-small.bin"  # whisper.cpp模型路径
    LLM_MODEL_PATH: str = "models/HY-MT1.5-1.8B-Q4_K_M.gguf"  # llama.cpp模型路径
    WHISPER_THREADS: int = 4  # whisper推理线程数
    LLM_THREADS: int = 4      # llama推理线程数
    LLM_CONTEXT_SIZE: int = 2048  # 上下文窗口大小
    LLM_N_GPU_LAYERS: int = 0    # GPU加速层数（0表示纯CPU）
    
    # 界面设置
    FLOATING_WINDOW_WIDTH: int = 900  # 窗口宽度（屏幕1/3）
    FLOATING_WINDOW_HEIGHT: int = 300  # 固定高度
    FONT_SIZE: int = 18  # 字体大小
    OPACITY: float = 0.9  # 窗口透明度
    MAX_HISTORY_LINES: int = 50  # 每侧最大历史行数
    
    # 语言支持
    LANGUAGES = {
        "自动检测": "auto",
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
    SENTENCE_ENDINGS = ['.', '?', '!', '。', '？', '！']
    
    # 文件存储
    OUTPUT_DIR: Path = Path.home() / "Documents" / "实时翻译记录"
    
    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== 第三部分：多线程队列架构 ==============
class AudioRecorder(QThread):
    """音频录制线程 - WebRTC VAD优化版（增加强制兜底机制）"""
    
    audio_ready = pyqtSignal(np.ndarray)  # 音频数据信号
    
    def __init__(self, config: Config, device_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.device_id = device_id
        self.is_running = False
        
        # WebRTC VAD初始化
        self.vad = webrtcvad.Vad(config.VAD_MODE)
        
        # 语音活动检测状态
        self.speech_buffer = []  # 当前语音块缓冲区
        self.silence_frames = 0  # 连续静音帧计数
        self.speech_start_time = 0  # 语音开始时间
        
        # 音频流
        self.stream = None
        
    def pcm_to_wav_format(self, audio_data: np.ndarray) -> bytes:
        """将PCM音频数据转换为VAD需要的16位小端格式"""
        # 确保是单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        # 确保是16位整数
        if audio_data.dtype != np.int16:
            # 归一化到[-32768, 32767]
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # 转换为字节
        pcm_bytes = audio_data.tobytes()
        return pcm_bytes
    
    def run(self):
        """运行音频录制线程 - V6.1重构：增加强制兜底机制"""
        self.is_running = True
        print(f"✅ 音频录制线程启动，设备ID: {self.device_id}")
        print(f"🔊 VAD配置: mode={self.config.VAD_MODE}")
        print(f"🕐 静音超时={self.config.VAD_SILENCE_TIMEOUT_MS}ms")
        print(f"🛑 强制兜底={self.config.VAD_MAX_SPEECH_MS}ms")
        
        try:
            # 启动音频流
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.config.CHANNELS,
                samplerate=self.config.SAMPLE_RATE,
                blocksize=self.config.FRAME_SIZE,
                latency='low',
                dtype=np.float32
            )
            
            self.stream.start()
            
            while self.is_running:
                # 读取一帧音频
                audio_frame, overflowed = self.stream.read(self.config.FRAME_SIZE)
                
                if overflowed:
                    print("⚠️ 音频缓冲区溢出")
                
                # 转换为PCM格式用于VAD
                pcm_data = self.pcm_to_wav_format(audio_frame)
                
                # VAD检测：判断当前帧是否包含语音
                try:
                    is_speech = self.vad.is_speech(pcm_data, self.config.SAMPLE_RATE)
                except Exception as e:
                    # VAD可能对某些帧格式敏感，错误时保守处理
                    is_speech = False
                
                # VAD状态机
                if is_speech:
                    # 检测到语音
                    self.silence_frames = 0  # 重置静音计数
                    
                    # 如果是语音的开始，记录时间
                    if len(self.speech_buffer) == 0:
                        self.speech_start_time = time.time()
                    
                    # 添加到语音缓冲区
                    self.speech_buffer.append(audio_frame.copy())
                    
                else:
                    # 检测到静音
                    self.silence_frames += 1
                    
                    # 如果已经有语音，静音帧也要添加到缓冲区（避免截断）
                    if len(self.speech_buffer) > 0:
                        self.speech_buffer.append(audio_frame.copy())
                    
                    # 计算静音持续时间
                    silence_duration_ms = self.silence_frames * self.config.VAD_FRAME_DURATION_MS
                    
                # ============ V6.1重构核心逻辑 ============
                # 实时计算当前语音缓冲区的音频时长（毫秒）
                current_speech_duration_ms = len(self.speech_buffer) * self.config.VAD_FRAME_DURATION_MS
                
                # 判断是否应该发送音频（双重断句机制）
                should_send_audio = False
                reason = ""
                
                # 条件A：静音超时断句（常规换气断句）
                if len(self.speech_buffer) > 0 and is_speech == False:
                    if silence_duration_ms >= self.config.VAD_SILENCE_TIMEOUT_MS:
                        should_send_audio = True
                        reason = f"静音断句（{silence_duration_ms:.0f}ms静音）"
                
                # 条件B：强制兜底断句（防卡死机制）
                if current_speech_duration_ms >= self.config.VAD_MAX_SPEECH_MS:
                    should_send_audio = True
                    reason = f"到达{self.config.VAD_MAX_SPEECH_MS}ms兜底断句"
                
                # 如果满足任一断句条件，立即发送音频
                if should_send_audio:
                    # 检查是否满足最小语音长度要求
                    if current_speech_duration_ms >= self.config.VAD_MIN_SPEECH_MS:
                        # 合并所有音频帧
                        speech_audio = np.concatenate(self.speech_buffer, axis=0)
                        
                        # 确保是单声道
                        if len(speech_audio.shape) > 1:
                            speech_audio = speech_audio[:, 0]
                        
                        # 转换为16位PCM格式
                        speech_audio = np.clip(speech_audio, -1.0, 1.0)
                        speech_audio = (speech_audio * 32767).astype(np.int16)
                        
                        # 发送完整语义块，附带断句原因日志
                        print(f"🎤 VAD检测到完整语句: {current_speech_duration_ms:.0f}ms音频 - {reason}")
                        self.audio_ready.emit(speech_audio)
                    
                    # 清空缓冲区，重置状态
                    self.speech_buffer = []
                    self.silence_frames = 0
                
                # 避免过度占用CPU
                time.sleep(0.001)
                
        except Exception as e:
            print(f"❌ 音频录制失败: {e}")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
    
    def stop(self):
        """停止音频录制"""
        self.is_running = False
        print("🛑 音频录制线程停止")

class SpeechRecognizer(QThread):
    """语音识别线程 - whisper.cpp版本（适配最新API）"""
    
    text_recognized = pyqtSignal(str)  # 识别文本信号
    status_update = pyqtSignal(str)     # 状态更新信号
    
    def __init__(self, config: Config, audio_queue: queue.Queue, session_id: str):
        super().__init__()
        self.config = config
        self.audio_queue = audio_queue
        self.session_id = session_id
        self.is_running = False
        self.language = "auto"
        self.model = None
        self.temp_dir = tempfile.mkdtemp(prefix="whisper_")
        
    def initialize_model(self):
        """初始化whisper.cpp模型"""
        try:
            # 检查模型文件是否存在
            model_path = Path(self.config.WHISPER_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"whisper.cpp模型文件不存在: {model_path}")
            
            # 初始化whisper.cpp（移除print_progress参数）
            self.model = Whisper(
                model_path=str(model_path),
                n_threads=self.config.WHISPER_THREADS,
            )
            print(f"[{self.session_id[:8]}] ✅ whisper.cpp模型加载成功: {model_path.name}")
        except Exception as e:
            print(f"[{self.session_id[:8]}] ❌ whisper.cpp模型加载失败: {e}")
            raise
    
    def save_audio_to_temp_wav(self, audio_data: np.ndarray) -> str:
        """将音频数据保存为临时wav文件"""
        # 创建临时文件
        temp_wav = tempfile.NamedTemporaryFile(
            dir=self.temp_dir,
            suffix=".wav",
            delete=False
        )
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        # 写入wav文件
        with wave.open(temp_wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位 = 2字节
            wav_file.setframerate(self.config.SAMPLE_RATE)
            
            # 确保音频数据是16位
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_wav_path
    
    def set_language(self, language: str):
        """设置识别语言"""
        # whisper.cpp的语言代码映射
        language_map = {
            "自动检测": "auto",
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
        self.language = language_map.get(language, "auto")
    
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
                
                # 保存为临时wav文件（必须传入文件路径）
                wav_path = self.save_audio_to_temp_wav(audio_data)
                
                try:
                    # 使用transcribe方法传入文件路径（最新API要求）
                    result = self.model.transcribe(
                        wav_path,
                        language=self.language
                    )
                    
                    # 处理返回结果
                    text = ""
                    if isinstance(result, dict):
                        text = result.get("text", "").strip()
                    elif isinstance(result, str):
                        text = result.strip()
                    else:
                        # 尝试从可能的结构中提取文本
                        text = str(result).strip()
                    
                    if text:
                        self.text_recognized.emit(text)
                        self.status_update.emit("识别完成")
                    else:
                        self.status_update.emit("等待语音...")
                        
                finally:
                    # 清理临时文件
                    try:
                        os.unlink(wav_path)
                    except:
                        pass
                
            except queue.Empty:
                continue  # 队列为空，继续等待
            except Exception as e:
                print(f"[{self.session_id[:8]}] ❌ 语音识别错误: {e}")
                self.status_update.emit(f"识别错误: {str(e)[:30]}")
    
    def stop(self):
        """停止语音识别 - 显式释放内存"""
        self.is_running = False
        self.status_update.emit("语音识别停止")
        
        # 显式释放模型
        if self.model:
            print(f"[{self.session_id[:8]}] 🧹 释放whisper.cpp模型内存")
            del self.model
            self.model = None
        
        # 清理临时目录
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
        
        # 强制垃圾回收
        gc.collect()
        
        print(f"[{self.session_id[:8]}] ✅ 语音识别资源已释放")

class StreamTranslator(QThread):
    """流式翻译线程 - llama.cpp版本（极简Zero-shot模式）"""
    
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
        self.model = None
        
        # V6.0重构：移除所有上下文缓存逻辑
        # 使用Zero-shot模式，每次翻译都是独立的
    
    def set_languages(self, source_lang: str, target_lang: str):
        """设置翻译语言"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        print(f"[{self.session_id[:8]}] ✅ 语言设置已更新: {source_lang} → {target_lang}")
    
    def initialize_model(self):
        """初始化llama.cpp模型"""
        try:
            # 检查模型文件是否存在
            model_path = Path(self.config.LLM_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"llama.cpp模型文件不存在: {model_path}")
            
            # 初始化llama.cpp
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.LLM_CONTEXT_SIZE,
                n_threads=self.config.LLM_THREADS,
                n_gpu_layers=self.config.LLM_N_GPU_LAYERS,
                verbose=False
            )
            print(f"[{self.session_id[:8]}] ✅ llama.cpp模型加载成功: {model_path.name}")
        except Exception as e:
            print(f"[{self.session_id[:8]}] ❌ llama.cpp模型加载失败: {e}")
            raise
    
    def build_prompt(self, text: str) -> str:
        """构建翻译提示词 - 极简Zero-shot模式"""
        # V6.0重构：使用极简Prompt适配1.8B小模型
        # 移除所有上下文缓存，实现Zero-shot翻译
        
        prompt = f"""You are a machine translation engine. Output ONLY the translation. No explanations, no filler words, no repetitions.

Translate this to {self.target_lang}: {text}

Translation:"""
        
        return prompt
    
    def run(self):
        """运行翻译线程"""
        self.is_running = True
        
        # 初始化模型
        try:
            self.initialize_model()
        except Exception as e:
            self.status_update.emit(f"模型错误: {str(e)[:30]}")
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
                
                # 构建极简提示词
                prompt = self.build_prompt(text)
                
                # llama.cpp流式生成
                full_translation = ""
                try:
                    # 调用llama.cpp进行流式生成
                    stream = self.model(
                        prompt,
                        max_tokens=256,
                        temperature=0.1,
                        top_p=0.9,
                        top_k=40,
                        repeat_penalty=1.1,
                        stop=["\n\n", "---", "==="],
                        stream=True
                    )
                    
                    # 处理流式响应
                    for output in stream:
                        chunk_text = output['choices'][0]['text']
                        full_translation += chunk_text
                        # 实时发送翻译片段
                        self.translation_chunk.emit(chunk_text)
                        
                    # 翻译完成
                    if full_translation.strip():
                        # 清理翻译结果
                        cleaned_translation = self.clean_translation(full_translation.strip())
                        
                        # 发送完整翻译信号
                        self.translation_complete.emit(cleaned_translation)
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
    
    def clean_translation(self, translation: str) -> str:
        """清理翻译结果 - 移除可能的幻觉内容"""
        # 移除常见的幻觉标记
        patterns_to_remove = [
            r'^[\[\{\(].*?[\]\}\)]\s*',  # 开头的括号内容
            r'\s*[\[\{\(].*?[\]\}\)]$',  # 结尾的括号内容
            r'^译文[：:]\s*',  # 开头的"译文："标记
            r'^翻译[：:]\s*',  # 开头的"翻译："标记
            r'^[A-Za-z]+[：:]\s*',  # 开头的英文标签
            r'^Assistant:\s*',  # 开头的Assistant标记
            r'^assistant:\s*',  # 小写assistant标记
            r'^\d+\.\s*',  # 开头的数字列表
        ]
        
        cleaned = translation
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # 移除多余的空格和换行
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def stop(self):
        """停止翻译 - 显式释放内存"""
        self.is_running = False
        self.status_update.emit("翻译停止")
        
        # 显式释放模型
        if self.model:
            print(f"[{self.session_id[:8]}] 🧹 释放llama.cpp模型内存")
            del self.model
            self.model = None
        
        # 强制垃圾回收
        gc.collect()
        
        print(f"[{self.session_id[:8]}] ✅ 翻译资源已释放")

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
        self.buffer_start_time = 0  # 缓冲区开始时间
        
        # 语言设置
        self.source_lang = "自动检测"
        self.target_lang = "中文"
        
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
        print(f"[{self.current_session_id[:8]}] 🔧 语言设置: {self.source_lang} → {self.target_lang}")
        
        self.is_running = True
        
        # 1. 启动音频录制线程
        self.audio_recorder = AudioRecorder(self.config, device_id)
        self.audio_recorder.audio_ready.connect(self.audio_callback)
        self.audio_recorder.start()
        
        # 2. 启动语音识别线程
        self.speech_recognizer = SpeechRecognizer(self.config, self.audio_queue, self.current_session_id)
        self.speech_recognizer.text_recognized.connect(self.handle_recognized_text)
        self.speech_recognizer.status_update.connect(self.status_signal)
        self.speech_recognizer.set_language(self.source_lang)
        self.speech_recognizer.start()
        
        # 3. 启动翻译线程
        self.translator = StreamTranslator(self.config, self.text_queue, self.current_session_id)
        self.translator.translation_chunk.connect(self.handle_translation_chunk)
        self.translator.translation_complete.connect(self.handle_translation_complete)
        self.translator.status_update.connect(self.status_signal)
        self.translator.set_languages(self.source_lang, self.target_lang)
        self.translator.start()
        
        # 初始化缓冲区时间
        self.buffer_start_time = time.time()
        
        self.status_signal.emit("流水线启动")
    
    def handle_recognized_text(self, text: str):
        """处理识别到的文本 - 直接送入翻译"""
        if not text.strip():
            return
        
        # 发送原文信号
        self.original_text_signal.emit(text)
        
        # 将文本放入翻译队列
        timestamp = datetime.now().strftime("%H:%M:%S")
        if not self.text_queue.full():
            self.text_queue.put((text, timestamp))
        else:
            print(f"[{self.current_session_id[:8]}] ⚠️ 文本队列已满，丢弃文本")
    
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
    
    def set_languages(self, source_lang: str, target_lang: str):
        """设置翻译语言"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # 清空文本缓冲区
        self.text_buffer = ""
        self.buffer_start_time = 0
        
        # 更新语音识别语言
        if self.speech_recognizer:
            self.speech_recognizer.set_language(source_lang)
        
        # 更新翻译语言
        if self.translator:
            self.translator.set_languages(source_lang, target_lang)
    
    def set_audio_device(self, device_id: int):
        """设置音频设备"""
        self.audio_device_id = device_id
    
    def switch_audio_device(self, device_id: int):
        """切换音频设备 - 热重载音频录制"""
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
        self.buffer_start_time = 0
        if self.current_session_id:
            print(f"[{self.current_session_id[:8]}] ✅ 所有队列已清空")

# ============== 第四部分：GUI界面 ==============
class FloatingWindow(QWidget):
    """悬浮字幕窗口 - 独立置顶窗口"""
    
    def __init__(self, config: Config):
        # 不要传入parent，让它成为独立的顶级窗口
        super().__init__(None)
        
        self.config = config
        self.init_ui()
        self.setup_window_properties()
        
        # 历史记录
        self.original_history = []
        self.translation_history = []
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("🎬 实时同传翻译")
        
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
        title_label = QLabel("🎬 实时同传翻译 V6.0")
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
        # 窗口置顶属性
        self.setWindowFlags(
            Qt.Window |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
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
    
    def update_opacity(self, opacity: float):
        """更新窗口透明度"""
        if opacity < 0.0:
            opacity = 0.0
        elif opacity > 1.0:
            opacity = 1.0
        self.setWindowOpacity(opacity)
    
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
        if not chunk.strip():
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
        
        # 必须先创建子组件，再初始化UI
        self.floating_window = FloatingWindow(config)
        self.translation_pipeline = TranslationPipeline(config)
        
        # 然后初始化UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
        
        # 系统托盘
        self.setup_system_tray()
        
        # 显示悬浮窗
        self.floating_window.show()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("专业级同传翻译器 V6.0 - WebRTC VAD重构版")
        self.setGeometry(100, 100, 650, 550)
        
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
        
        # V6.0重构：移除场景/领域设置（极简模式）
        
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
        audio_note.setText("• 监听系统声音：选择 BlackHole 2ch\n• 麦克风录音：选择 Built-in Microphone\n• V6.0特性：WebRTC VAD智能断句，500ms静音判定语句结束")
        audio_layout.addWidget(audio_note)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # 3. 模型设置组
        model_group = QGroupBox("🤖 模型设置")
        model_layout = QVBoxLayout()
        
        # whisper.cpp模型路径
        whisper_layout = QHBoxLayout()
        whisper_layout.addWidget(QLabel("Whisper模型:"))
        self.whisper_model_path = QLineEdit(self.config.WHISPER_MODEL_PATH)
        self.whisper_model_path.setPlaceholderText("ggml-base.bin 或 ggml-small.bin")
        whisper_layout.addWidget(self.whisper_model_path)
        
        whisper_btn = QPushButton("浏览")
        whisper_btn.clicked.connect(self.browse_whisper_model)
        whisper_layout.addWidget(whisper_btn)
        
        model_layout.addLayout(whisper_layout)
        
        # llama.cpp模型路径
        llm_layout = QHBoxLayout()
        llm_layout.addWidget(QLabel("翻译模型:"))
        self.llm_model_path = QLineEdit(self.config.LLM_MODEL_PATH)
        self.llm_model_path.setPlaceholderText("HY-MT1.5-1.8B-GGUF-q8_0.gguf")
        llm_layout.addWidget(self.llm_model_path)
        
        llm_btn = QPushButton("浏览")
        llm_btn.clicked.connect(self.browse_llm_model)
        llm_layout.addWidget(llm_btn)
        
        model_layout.addLayout(llm_layout)
        
        # 模型说明
        model_note = QLabel("💡 V6.0特性：显式内存管理，解决反复启动OOM问题\n下载模型：whisper.cpp模型从 https://huggingface.co/ggerganov/whisper.cpp/tree/main\n腾讯混元翻译模型：https://huggingface.co/Tencent/HY-MT1.5-1.8B-GGUF")
        model_note.setStyleSheet("color: #666; font-size: 12px;")
        model_note.setWordWrap(True)
        model_layout.addWidget(model_note)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # 悬浮窗控制组
        floating_group = QGroupBox("🪟 悬浮窗控制")
        floating_layout = QVBoxLayout()
        
        # 显示/隐藏按钮
        floating_btn_layout = QHBoxLayout()
        
        self.show_floating_btn = QPushButton("👁️ 显示悬浮窗")
        self.show_floating_btn.setStyleSheet("""
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
        """)
        
        self.hide_floating_btn = QPushButton("👁️​🗨️ 隐藏悬浮窗")
        self.hide_floating_btn.setStyleSheet("""
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
        
        floating_btn_layout.addWidget(self.show_floating_btn)
        floating_btn_layout.addWidget(self.hide_floating_btn)
        floating_layout.addLayout(floating_btn_layout)
        
        # 透明度滑块
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("透明度:"))
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(10)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self.config.OPACITY * 100))
        self.opacity_slider.setTickInterval(10)
        self.opacity_slider.setTickPosition(QSlider.TicksBelow)
        
        self.opacity_label = QLabel(f"{int(self.config.OPACITY * 100)}%")
        self.opacity_label.setFixedWidth(40)
        
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_label)
        floating_layout.addLayout(opacity_layout)
        
        floating_group.setLayout(floating_layout)
        main_layout.addWidget(floating_group)
        
        # 4. 控制按钮组
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
        self.save_button.setEnabled(False)
        
        operation_layout.addWidget(clear_btn)
        operation_layout.addWidget(self.save_button)
        control_layout.addLayout(operation_layout)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # 5. 状态显示组
        status_group = QGroupBox("📊 状态信息")
        status_layout = QVBoxLayout()
        
        # 状态标签
        self.status_label = QLabel("状态: 等待开始")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        status_layout.addWidget(self.status_label)
        
        # 进度指示器
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """)
        status_layout.addWidget(self.progress_bar)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # 底部说明
        footer_label = QLabel("💡 V6.0特性：WebRTC VAD智能断句 + 显式内存管理 + 极简Zero-shot翻译\n提示：悬浮窗可以拖动，右键系统托盘图标可退出程序")
        footer_label.setStyleSheet("color: #888; font-size: 12px; padding: 10px;")
        footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer_label)
        
        central_widget.setLayout(main_layout)
    
    def browse_whisper_model(self):
        """浏览whisper.cpp模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择whisper.cpp模型文件",
            str(Path.home()),
            "模型文件 (*.bin);;所有文件 (*.*)"
        )
        if file_path:
            self.whisper_model_path.setText(file_path)
            self.config.WHISPER_MODEL_PATH = file_path
    
    def browse_llm_model(self):
        """浏览llama.cpp模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择llama.cpp模型文件",
            str(Path.home()),
            "GGUF模型 (*.gguf *.bin);;所有文件 (*.*)"
        )
        if file_path:
            self.llm_model_path.setText(file_path)
            self.config.LLM_MODEL_PATH = file_path
    
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
        """连接信号与槽"""
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
        
        # 状态栏信号连接
        self.translation_pipeline.status_signal.connect(self.update_status)
        
        # 语言切换信号
        self.source_lang_combo.currentTextChanged.connect(self.on_language_changed)
        self.target_lang_combo.currentTextChanged.connect(self.on_language_changed)
        
        # 音频设备切换信号
        self.audio_device_combo.currentIndexChanged.connect(self.on_audio_device_changed)
        
        # 修复：在这里绑定按钮信号（此时floating_window已存在）
        self.show_floating_btn.clicked.connect(self.floating_window.show)
        self.hide_floating_btn.clicked.connect(self.floating_window.hide)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        self.start_button.clicked.connect(self.start_translation)
        self.stop_button.clicked.connect(self.stop_translation)
        self.save_button.clicked.connect(self.save_subtitles)
        
        # 清空按钮
        clear_btn = self.findChild(QPushButton, None)
        if clear_btn and clear_btn.text() == "🗑️ 清空字幕":
            clear_btn.clicked.connect(self.clear_all)
    
    def on_opacity_changed(self, value: int):
        """透明度滑块变化处理"""
        self.opacity_label.setText(f"{value}%")
        # 确保将滑块值除以100.0，得到0.0-1.0之间的浮点数
        opacity_value = value / 100.0
        self.floating_window.update_opacity(opacity_value)
    
    def on_language_changed(self):
        """语言切换处理"""
        source_lang = self.source_lang_combo.currentText()
        target_lang = self.target_lang_combo.currentText()
        
        # 更新流水线语言设置
        self.translation_pipeline.set_languages(source_lang, target_lang)
        
        # 更新状态显示
        status_text = f"语言设置已更新: {source_lang} → {target_lang}"
        self.update_status(status_text)
    
    def on_audio_device_changed(self):
        """音频设备切换处理"""
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
        
        # 根据状态更新进度条
        if "识别中" in status:
            self.progress_bar.setValue(30)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    background-color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #FF9800;
                    border-radius: 5px;
                }
            """)
        elif "翻译中" in status:
            self.progress_bar.setValue(70)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    background-color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #2196F3;
                    border-radius: 5px;
                }
            """)
        elif "完成" in status or "就绪" in status:
            self.progress_bar.setValue(100)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    background-color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 5px;
                }
            """)
        elif "错误" in status:
            self.progress_bar.setValue(0)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    background-color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #f44336;
                    border-radius: 5px;
                }
            """)
        
        # 更新悬浮窗状态
        self.floating_window.update_status(status)
    
    def start_translation(self):
        """开始翻译"""
        try:
            # 获取设置
            source_lang = self.source_lang_combo.currentText()
            target_lang = self.target_lang_combo.currentText()
            audio_device_index = self.audio_device_combo.currentData()
            
            # 检查模型文件是否存在
            whisper_model_path = Path(self.config.WHISPER_MODEL_PATH)
            llm_model_path = Path(self.config.LLM_MODEL_PATH)
            
            if not whisper_model_path.exists():
                QMessageBox.critical(self, "模型文件错误", 
                    f"whisper.cpp模型文件不存在:\n{whisper_model_path}\n\n请下载: https://huggingface.co/ggerganov/whisper.cpp/tree/main")
                return
                
            if not llm_model_path.exists():
                QMessageBox.critical(self, "模型文件错误", 
                    f"llama.cpp模型文件不存在:\n{llm_model_path}\n\n推荐使用: HY-MT1.5-1.8B-GGUF-q8_0.gguf\n下载: https://huggingface.co/Tencent/HY-MT1.5-1.8B-GGUF")
                return
            
            # 清空所有缓存和历史记录
            self.floating_window.clear_all()
            
            # 设置翻译流水线
            self.translation_pipeline.set_languages(source_lang, target_lang)
            self.translation_pipeline.set_audio_device(audio_device_index)
            
            # 启动流水线
            self.translation_pipeline.start()
            
            # 更新按钮状态
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.save_button.setEnabled(False)
            
            status_text = f"翻译已启动: {source_lang} → {target_lang}"
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
            self.floating_window.clear_all()
            self.update_status("字幕已清空（阅后即焚）")
    
    def clear_all(self):
        """清空所有内容"""
        self.floating_window.clear_all()
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
            # 获取当前窗口中的原文和译文
            original_text = self.floating_window.original_text.toPlainText()
            translation_text = self.floating_window.translation_text.toPlainText()
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"实时翻译记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"源语言: {source_lang} → 目标语言: {target_lang}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("原文:\n")
                    f.write(original_text + "\n\n")
                    f.write("译文:\n")
                    f.write(translation_text + "\n")
                
                QMessageBox.information(self, "成功", f"字幕已保存到:\n{file_path}")
                self.update_status(f"字幕已保存: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
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
    print("🎬 专业级实时同传翻译器 V6.0 - WebRTC VAD重构版")
    print("=" * 60)
    print("💡 使用说明：")
    print("1. 安装依赖包：pip install sounddevice numpy whisper-cpp-python llama-cpp-python PyQt5 webrtcvad scipy")
    print("2. 下载模型文件：")
    print("   - whisper.cpp模型：从 https://huggingface.co/ggerganov/whisper.cpp/tree/main 下载")
    print("   - 腾讯混元翻译模型：https://huggingface.co/Tencent/HY-MT1.5-1.8B-GGUF")
    print("3. 将模型文件放在 models/ 目录下，或通过界面浏览选择")
    print("4. 选择源语言和目标语言（支持29种语言）")
    print("5. 选择音频输入源（BlackHole监听系统声音）")
    print("6. 点击开始翻译")
    print("7. V6.0新特性：")
    print("   - WebRTC VAD智能断句（500ms静音自动断句）")
    print("   - 显式内存管理，解决反复启动OOM问题")
    print("   - 极简Zero-shot翻译，适配1.8B小模型")
    print("=" * 60)
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("专业级同传翻译器 V6.0")
    app.setApplicationDisplayName("专业级同传翻译器 V6.0")
    
    # 创建配置和控制窗口
    config = Config()
    control_window = ControlWindow(config)
    control_window.show()
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()