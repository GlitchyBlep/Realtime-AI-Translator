"""
实时同传翻译器 - qwen2.5：14b版
功能：实时监听系统音频/麦克风 → 语音识别 → 流式翻译 → 悬浮窗显示
架构：完全解耦的三线程队列系统
使用方法：python realtime_translator_fixed.py
"""

import sys
import os
import time
import json
import threading
import queue
import wave
import tempfile
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
                                 QGroupBox, QTextEdit, QFileDialog,
                                 QMessageBox, QSystemTrayIcon, QMenu, QAction)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
    from PyQt5.QtGui import QIcon
    
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
    CHUNK_DURATION: float = 4.0  # 每次处理的音频时长（秒）
    CHUNK_SIZE: int = int(SAMPLE_RATE * CHUNK_DURATION)  # 每个音频块的大小
    
    # 队列设置
    AUDIO_QUEUE_SIZE: int = 10  # 音频队列最大大小
    TEXT_QUEUE_SIZE: int = 9999  # 文本队列最大大小
    
    # 模型设置
    WHISPER_MODEL_SIZE: str = "small"  # tiny, base, small, medium
    WHISPER_MODEL_DEVICE: str = "cpu"  # macOS用cpu
    WHISPER_MODEL_COMPUTE_TYPE: str = "int8"  # 量化减少内存
    
    TRANSLATION_MODEL: str = "qwen2.5:14b"  # 使用标准模型名称
    
    # 界面设置
    FLOATING_WINDOW_WIDTH: int = 800
    FLOATING_WINDOW_HEIGHT: int = 180
    FONT_SIZE: int = 24
    MAX_LINES: int = 3  # 悬浮窗最多显示3行
    OPACITY: float = 0.9  # 窗口透明度
    
    # 语言支持
    LANGUAGES = {
        "自动检测": "auto",
        "中文": "zh",
        "英文": "en",
        "德文": "de",
        "法文": "fr",
        "日文": "ja",
        "韩文": "ko"
    }
    
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
    
    def __init__(self, config: Config, audio_queue: queue.Queue):
        super().__init__()
        self.config = config
        self.audio_queue = audio_queue
        self.is_running = False
        self.language = "auto"
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
            print("✅ Whisper模型加载成功")
        except Exception as e:
            print(f"❌ Whisper模型加载失败: {e}")
            raise
    
    def set_language(self, language: str):
        """设置识别语言"""
        self.language = language if language != "自动检测" else None
    
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
                        language=self.language,
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
                print(f"❌ 语音识别错误: {e}")
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
    
    def __init__(self, config: Config, text_queue: queue.Queue):
        super().__init__()
        self.config = config
        self.text_queue = text_queue
        self.is_running = False
        self.source_lang = "auto"
        self.target_lang = "zh"
        self.client = None
        self.translation_history = []
        
    def set_languages(self, source_lang: str, target_lang: str):
        """设置翻译语言"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # 构建目标语言的中文显示名称
        lang_display_map = {
            "中文": "中文",
            "英文": "英文",
            "德文": "德文",
            "法文": "法文",
            "日文": "日文",
            "韩文": "韩文"
        }
        
        self.target_lang_display = lang_display_map.get(target_lang, "中文")
    
    def initialize_client(self):
        """初始化Ollama客户端"""
        try:
            self.client = ollama.Client()
            # 测试连接
            self.client.list()
            print("✅ Ollama客户端初始化成功")
        except Exception as e:
            print(f"❌ Ollama连接失败: {e}")
            raise
    
    def build_prompt(self, text: str) -> str:
        """构建翻译提示词 - 修复：动态目标语言"""
        # 暴力中文指令，根据目标语言动态调整
        return f"你是一个极其专业的同传翻译。请直接给出下面这段话的{self.target_lang_display}翻译，绝对不要重复原文，不要加任何标点和解释：\n\n原文：{text}\n\n译文："
    
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
                
                # 流式翻译
                full_translation = ""
                try:
                    # 调用Ollama进行流式生成 - 修复：移除num_predict
                    response = self.client.generate(
                        model=self.config.TRANSLATION_MODEL,
                        prompt=prompt,
                        stream=True,
                        options={
                            "temperature": 0.1  # 只保留temperature
                        }
                    )
                    
                    # 处理流式响应
                    for chunk in response:
                        if 'response' in chunk:
                            chunk_text = chunk['response']
                            full_translation += chunk_text
                            # 实时发送翻译片段
                            self.translation_chunk.emit(chunk_text)
                            
                    # 翻译完成
                    if full_translation.strip():
                        # 保存到历史记录
                        self.translation_history.append({
                            "timestamp": timestamp,
                            "source": text,
                            "translation": full_translation,
                            "source_lang": self.source_lang,
                            "target_lang": self.target_lang
                        })
                        
                        # 发送完整翻译信号 - 修复：确保发射信号
                        self.translation_complete.emit(full_translation.strip())
                        self.status_update.emit("翻译完成")
                    else:
                        self.status_update.emit("翻译为空")
                        
                except Exception as e:
                    print(f"❌ 翻译错误: {e}")
                    # 发送错误信息
                    self.translation_complete.emit(f"[翻译错误: {str(e)[:30]}]")
                    self.status_update.emit(f"翻译错误: {str(e)[:30]}")
                    
            except queue.Empty:
                continue  # 队列为空，继续等待
            except Exception as e:
                print(f"❌ 翻译线程错误: {e}")
                self.status_update.emit(f"线程错误: {str(e)[:30]}")
    
    def stop(self):
        """停止翻译"""
        self.is_running = False
        self.status_update.emit("翻译停止")

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
        
        # 创建队列
        self.audio_queue = queue.Queue(maxsize=config.AUDIO_QUEUE_SIZE)
        self.text_queue = queue.Queue(maxsize=config.TEXT_QUEUE_SIZE)
        
        # 初始化组件
        self.audio_recorder = None
        self.speech_recognizer = None
        self.translator = None
        
        # 数据存储
        self.subtitles = []  # 字幕历史
        
        # 语言设置
        self.source_lang = "auto"
        self.target_lang = "zh"
        
        # 音频设备
        self.audio_device_id = None
        
    def audio_callback(self, audio_data: np.ndarray):
        """音频回调函数 - 将音频数据放入队列"""
        if self.is_running and not self.audio_queue.full():
            self.audio_queue.put(audio_data)
    
    def start_pipeline(self, device_id: Optional[int] = None):
        """启动整个流水线"""
        self.is_running = True
        
        # 1. 启动音频录制线程
        self.audio_recorder = AudioRecorder(self.config, device_id)
        self.audio_recorder.audio_ready.connect(self.audio_callback)
        self.audio_recorder.start()
        
        # 2. 启动语音识别线程
        self.speech_recognizer = SpeechRecognizer(self.config, self.audio_queue)
        self.speech_recognizer.text_recognized.connect(self.handle_recognized_text)
        self.speech_recognizer.status_update.connect(self.status_signal)
        self.speech_recognizer.set_language(self.config.LANGUAGES.get(self.source_lang, "auto"))
        self.speech_recognizer.start()
        
        # 3. 启动翻译线程
        self.translator = StreamTranslator(self.config, self.text_queue)
        self.translator.translation_chunk.connect(self.translation_chunk_signal)
        self.translator.translation_complete.connect(self.handle_translation_complete)
        self.translator.status_update.connect(self.status_signal)
        self.translator.set_languages(self.source_lang, self.target_lang)
        self.translator.start()
        
        self.status_signal.emit("流水线启动")
    
    def handle_recognized_text(self, text: str):
        """处理识别到的文本"""
        if text:
            # 发送原文信号
            self.original_text_signal.emit(text)
            
            # 将文本放入翻译队列
            timestamp = datetime.now().strftime("%H:%M:%S")
            if not self.text_queue.full():
                self.text_queue.put((text, timestamp))
    
    def handle_translation_complete(self, translation: str):
        """处理翻译完成的文本 - 修复：正确发射信号"""
        # 发射翻译完成信号到UI
        if translation and translation.strip():
            self.translation_complete_signal.emit(translation)
    
    def set_languages(self, source_lang: str, target_lang: str):
        """设置翻译语言"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # 更新语音识别语言
        if self.speech_recognizer:
            self.speech_recognizer.set_language(self.config.LANGUAGES.get(source_lang, "auto"))
        
        # 更新翻译语言
        if self.translator:
            self.translator.set_languages(source_lang, target_lang)
    
    def set_audio_device(self, device_id: int):
        """设置音频设备"""
        self.audio_device_id = device_id
    
    def run(self):
        """运行流水线"""
        self.start_pipeline(self.audio_device_id)
        
        # 保持线程运行
        while self.is_running:
            time.sleep(0.1)
    
    def stop(self):
        """停止流水线"""
        self.is_running = False
        
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
        
        self.status_signal.emit("流水线停止")
    
    def save_subtitles(self, file_path: str):
        """保存字幕到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"实时翻译记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"源语言: {self.source_lang} → 目标语言: {self.target_lang}\n")
                f.write("=" * 60 + "\n\n")
                
                # 这里可以保存翻译历史
                # 需要从translator中获取
                
            return True
        except Exception as e:
            print(f"❌ 保存字幕失败: {e}")
            return False
    
    def clear_subtitles(self):
        """清空字幕记录"""
        self.subtitles = []

# ============== 第四部分：GUI界面 ==============
class FloatingWindow(QWidget):
    """悬浮字幕窗口 - 修复：简化显示逻辑"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.init_ui()
        self.setup_window_properties()
        
        # 字幕缓冲区
        self.subtitle_lines = []
        self.current_translation = ""
        
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
        """)
        
        # 主布局
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)
        
        # 字幕显示标签（多行）
        self.subtitle_label = QLabel("等待翻译...")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: white;
                font-size: {self.config.FONT_SIZE}px;
                font-weight: bold;
                font-family: 'PingFang SC', 'Helvetica Neue', Arial, sans-serif;
                background-color: transparent;
                padding: 5px;
                line-height: 1.5;
            }}
        """)
        self.subtitle_label.setWordWrap(True)
        
        layout.addWidget(self.subtitle_label)
        self.setLayout(layout)
    
    def setup_window_properties(self):
        """设置窗口属性"""
        # 窗口置顶
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint | 
            Qt.Tool |
            Qt.X11BypassWindowManagerHint  # 在某些系统上更好的置顶效果
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
        """更新字幕显示 - 只显示原文"""
        if not original_text.strip():
            return
            
        # 限制最多显示3行
        if len(self.subtitle_lines) >= self.config.MAX_LINES:
            self.subtitle_lines.pop(0)
        
        # 添加新字幕
        timestamp = datetime.now().strftime("%H:%M:%S")
        new_line = f"[{timestamp}] {original_text}"
        self.subtitle_lines.append(new_line)
        
        # 更新显示
        display_text = "\n".join(self.subtitle_lines[-self.config.MAX_LINES:])
        self.subtitle_label.setText(display_text)
        
        # 清空当前翻译缓存
        self.current_translation = ""
        
        # 自动调整高度
        line_count = display_text.count('\n') + 1
        new_height = max(100, min(300, line_count * 50 + 20))
        self.resize(self.width(), new_height)
    
    def update_translation_chunk(self, chunk: str):
        """更新流式翻译片段"""
        if not chunk.strip():
            return
            
        self.current_translation += chunk
        
        # 构建显示文本
        if self.subtitle_lines:
            # 显示原文最后一行 + 当前翻译
            display_lines = self.subtitle_lines[-self.config.MAX_LINES+1:] if len(self.subtitle_lines) > 1 else self.subtitle_lines
            display_text = "\n".join(display_lines)
            
            # 如果有当前翻译，添加到末尾
            if self.current_translation:
                display_text += f"\n→ {self.current_translation}"
                
            self.subtitle_label.setText(display_text)
    
    def update_translation_complete(self, translation: str):
        """更新完整翻译 - 修复：确保显示译文"""
        if not translation.strip() or translation.startswith("[翻译错误"):
            return
            
        # 清空当前翻译缓存
        self.current_translation = ""
        
        # 如果最后一行是原文，添加译文
        if self.subtitle_lines:
            self.subtitle_lines[-1] = f"{self.subtitle_lines[-1]}\n→ {translation}"
            
            # 限制行数
            if len(self.subtitle_lines) > self.config.MAX_LINES:
                self.subtitle_lines = self.subtitle_lines[-self.config.MAX_LINES:]
            
            display_text = "\n".join(self.subtitle_lines[-self.config.MAX_LINES:])
            self.subtitle_label.setText(display_text)
    
    def clear_subtitles(self):
        """清空字幕"""
        self.subtitle_lines = []
        self.current_translation = ""
        self.subtitle_label.setText("等待翻译...")

class ControlWindow(QMainWindow):
    """控制窗口"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.init_ui()
        
        # 初始化组件
        self.floating_window = FloatingWindow(config)
        self.translation_pipeline = TranslationPipeline(config)
        
        # 连接信号 - 修复：确保所有信号都连接
        self.connect_signals()
        
        # 系统托盘
        self.setup_system_tray()
        
        # 显示悬浮窗
        self.floating_window.show()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("实时同传翻译器 - 控制台")
        self.setGeometry(100, 100, 500, 450)
        
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
        self.source_lang_combo.addItems(self.config.LANGUAGES.keys())
        self.source_lang_combo.setCurrentText("英文")
        source_layout.addWidget(self.source_lang_combo)
        lang_layout.addLayout(source_layout)
        
        # 目标语言
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("目标语言:"))
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems([lang for lang in self.config.LANGUAGES.keys() if lang != "自动检测"])
        self.target_lang_combo.setCurrentText("中文")
        target_layout.addWidget(self.target_lang_combo)
        lang_layout.addLayout(target_layout)
        
        lang_group.setLayout(lang_layout)
        main_layout.addWidget(lang_group)
        
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
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_button.clicked.connect(self.start_translation)
        
        self.stop_button = QPushButton("⏸️ 停止翻译")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.stop_button.clicked.connect(self.stop_translation)
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        control_layout.addLayout(button_layout)
        
        # 保存按钮
        self.save_button = QPushButton("💾 保存字幕")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.save_button.clicked.connect(self.save_subtitles)
        self.save_button.setEnabled(False)
        control_layout.addWidget(self.save_button)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # 4. 状态显示组
        status_group = QGroupBox("📊 状态信息")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("状态: 等待开始")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        status_layout.addWidget(self.status_label)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        self.info_text.setStyleSheet("font-size: 12px; background-color: #f5f5f5;")
        self.info_text.setText("就绪，请选择语言和音频源后点击开始。\n\n使用提示：\n1. 确保Ollama服务正在运行\n2. 选择正确的音频输入源\n3. 点击开始后，悬浮窗会显示实时翻译")
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
        """连接信号与槽 - 修复：确保所有信号都连接"""
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
                    if 'qwen2.5' in model_name.lower() and '14b' in model_name.lower():
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
            
            # 设置翻译流水线
            self.translation_pipeline.set_languages(source_lang, target_lang)
            self.translation_pipeline.set_audio_device(audio_device_index)
            
            # 启动流水线
            self.translation_pipeline.start()
            
            # 更新按钮状态
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.save_button.setEnabled(False)
            
            # 清空悬浮窗
            self.floating_window.clear_subtitles()
            
            self.update_status("翻译已启动")
            
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
            self.floating_window.clear_subtitles()
            self.update_status("字幕已清空（阅后即焚）")
    
    def save_subtitles(self):
        """保存字幕到文件"""
        # 生成默认文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"翻译记录_{timestamp}.txt"
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
                QMessageBox.critical(self, "错误", "保存失败")
    
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
        
        tray_menu.addSeparator()
        
        exit_action = QAction("退出程序", self)
        exit_action.triggered.connect(self.close_all)
        tray_menu.addAction(exit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
    
    def close_all(self):
        """关闭所有窗口并退出"""
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
    print("🎬 实时视频同传翻译器 - 最终修复版")
    print("=" * 60)
    print("💡 使用说明：")
    print("1. 确保Ollama服务正在运行：ollama serve")
    print("2. 确保已下载模型：ollama pull qwen2.5:14b")
    print("3. 选择源语言和目标语言")
    print("4. 选择音频输入源（BlackHole监听系统声音）")
    print("5. 点击开始翻译")
    print("6. 悬浮窗将显示实时翻译字幕")
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
            print("⚠️  未找到qwen2.5:14b模型或兼容模型")
            print("请运行: ollama pull qwen3.5:9b")
            print("如果已经安装，请确认模型名称是否正确")
            
    except Exception as e:
        print(f"❌ 无法连接Ollama: {e}")
        print("请确保已安装并运行Ollama：")
        print("1. 访问 https://ollama.ai 下载安装")
        print("2. 新终端窗口运行: ollama serve")
        print("3. 另一个终端运行: ollama pull qwen2.5:14b")
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("实时同传翻译器")
    app.setApplicationDisplayName("实时同传翻译器")
    
    # 创建配置和控制窗口
    config = Config()
    control_window = ControlWindow(config)
    control_window.show()
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()