"""
Realtime Translator V4.1 - Ultimate Fix Version
Functionality: Real-time listening to system audio/microphone → Precise punctuation segmentation → Strict translation → Independent always-on-top floating window
Architecture: Multi-threaded queue + Precise segmentation + Anti-hallucination mechanism + Independent floating window
How to use: python pro_translator_v4_1.py
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

# ============== Part 1: Import All Libraries ==============
try:
    # Audio Processing
    import sounddevice as sd
    import numpy as np
    
    # Speech Recognition
    from faster_whisper import WhisperModel
    
    # Large Model Translation
    import ollama
    
    # GUI Interface
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                                 QPushButton, QComboBox, QVBoxLayout, QHBoxLayout,
                                 QGroupBox, QTextEdit, QFileDialog, QLineEdit,
                                 QMessageBox, QSystemTrayIcon, QMenu, QAction,
                                 QSplitter, QFrame, QSlider, QDial, QProgressBar)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QPoint
    from PyQt5.QtGui import QIcon, QFont, QTextCursor, QColor
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please run: pip install sounddevice numpy faster-whisper ollama PyQt5")
    sys.exit(1)

# ============== Part 2: System Configuration ==============
@dataclass
class Config:
    """System Configuration - Issue 5: Back to 3.0s ultra-fast mode"""
    # Audio Settings
    SAMPLE_RATE: int = 16000  # Sample rate required by Whisper
    CHUNK_DURATION: float = 3.0  # Issue 5: Back to 3.0s for ultra-fast translation
    CHUNK_SIZE: int = int(SAMPLE_RATE * CHUNK_DURATION)  # Size of each audio chunk
    
    # Queue Settings
    AUDIO_QUEUE_SIZE: int = 10  # Max size of the audio queue
    TEXT_QUEUE_SIZE: int = 9999  # Max size of the text queue
    
    # Buffer Timeout Settings
    BUFFER_TIMEOUT_SECONDS: float = 8.0  # Buffer timeout in seconds
    BUFFER_MAX_CHARS: int = 200  # Max characters in the buffer (fallback mechanism)
    
    # Model Settings
    WHISPER_MODEL_SIZE: str = "small"  # tiny, base, small, medium
    WHISPER_MODEL_DEVICE: str = "cpu"  # Use cpu on macOS
    WHISPER_MODEL_COMPUTE_TYPE: str = "int8"  # Quantization to reduce memory
    
    TRANSLATION_MODEL: str = "qwen3.5:9b"  # Use standard model name
    
    # UI Settings
    FLOATING_WINDOW_WIDTH: int = 900  # Window width (1/3 of the screen)
    FLOATING_WINDOW_HEIGHT: int = 300  # Fixed height
    FONT_SIZE: int = 18  # Font size
    OPACITY: float = 0.9  # Window opacity
    MAX_HISTORY_LINES: int = 50  # Max history lines on each side
    
    # Language Support - Use None for auto-detection
    LANGUAGES = {
        "Auto-detect": None,
        "Chinese": "zh",
        "English": "en",
        "Japanese": "ja",
        "Korean": "ko",
        "French": "fr",
        "Spanish": "es",
        "Portuguese": "pt",
        "German": "de",
        "Italian": "it",
        "Russian": "ru",
        "Vietnamese": "vi",
        "Thai": "th",
        "Arabic": "ar",
        "Indonesian": "id",
        "Malay": "ms",
        "Turkish": "tr",
        "Dutch": "nl",
        "Polish": "pl",
        "Czech": "cs",
        "Swedish": "sv",
        "Romanian": "ro",
        "Hindi": "hi",
        "Bengali": "bn",
        "Persian": "fa",
        "Urdu": "ur",
        "Greek": "el",
        "Finnish": "fi",
        "Danish": "da",
        "Hungarian": "hu"
    }
    
    # End-of-sentence punctuation (for intelligent segmentation)
    SENTENCE_ENDINGS = ['.', '?', '!', '。', '？', '！']
    
    # File Storage
    OUTPUT_DIR: Path = Path.home() / "Documents" / "Realtime Translation Logs"
    
    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== Part 3: Multi-Threaded Queue Architecture ==============
class AudioRecorder(QThread):
    """Audio Recording Thread - Independent thread for recording audio"""
    
    audio_ready = pyqtSignal(np.ndarray)  # Audio data signal
    
    def __init__(self, config: Config, device_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.device_id = device_id
        self.is_running = False
        self.audio_buffer = []
        self.buffer_size = 0
        self.target_buffer_size = config.CHUNK_SIZE
        
    def run(self):
        """Run the audio recording thread"""
        self.is_running = True
        
        def audio_callback(indata, frames, time_info, status):
            """Audio callback function"""
            if status:
                print(f"Audio status: {status}")
            
            if self.is_running:
                # Add audio data to the buffer
                audio_data = indata.copy()
                self.audio_buffer.append(audio_data)
                self.buffer_size += len(audio_data)
                
                # When the buffer reaches the target size
                if self.buffer_size >= self.target_buffer_size:
                    # Combine the buffer data
                    full_audio = np.concatenate(self.audio_buffer, axis=0)[:self.target_buffer_size]
                    
                    # Send the audio data
                    self.audio_ready.emit(full_audio)
                    
                    # Reset the buffer (keep the remainder)
                    remaining = self.buffer_size - self.target_buffer_size
                    if remaining > 0:
                        self.audio_buffer = [np.concatenate(self.audio_buffer, axis=0)[self.target_buffer_size:]]
                        self.buffer_size = remaining
                    else:
                        self.audio_buffer = []
                        self.buffer_size = 0
        
        try:
            # Start the audio stream
            with sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.config.SAMPLE_RATE,
                callback=audio_callback,
                blocksize=int(self.config.SAMPLE_RATE * 0.1),  # 100ms blocks
                latency='low'
            ):
                print(f"✅ Audio recording thread started, device ID: {self.device_id}")
                # Keep the thread running
                while self.is_running:
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"❌ Audio recording failed: {e}")
    
    def stop(self):
        """Stop audio recording"""
        self.is_running = False
        print("🛑 Audio recording thread stopped")

class SpeechRecognizer(QThread):
    """Speech Recognition Thread - Independent thread for speech recognition"""
    
    text_recognized = pyqtSignal(str)  # Recognized text signal
    status_update = pyqtSignal(str)     # Status update signal
    
    def __init__(self, config: Config, audio_queue: queue.Queue, session_id: str):
        super().__init__()
        self.config = config
        self.audio_queue = audio_queue
        self.session_id = session_id
        self.is_running = False
        self.language = None  # Initialize to None
        self.model = None
        
    def initialize_model(self):
        """Initialize the Whisper model"""
        try:
            self.model = WhisperModel(
                self.config.WHISPER_MODEL_SIZE,
                device=self.config.WHISPER_MODEL_DEVICE,
                compute_type=self.config.WHISPER_MODEL_COMPUTE_TYPE,
                cpu_threads=4
            )
            print(f"[{self.session_id[:8]}] ✅ Whisper model loaded successfully")
        except Exception as e:
            print(f"[{self.session_id[:8]}] ❌ Whisper model loading failed: {e}")
            raise
    
    def set_language(self, language: str):
        """Set the recognition language"""
        self.language = language  # Can be None or a language code
    
    def run(self):
        """Run the speech recognition thread"""
        self.is_running = True
        
        # Initialize model
        self.initialize_model()
        self.status_update.emit("Speech recognition ready")
        
        while self.is_running:
            try:
                # Get audio data from the queue (with timeout)
                audio_data = self.audio_queue.get(timeout=0.1)
                
                if audio_data is None:  # Stop signal
                    break
                    
                # Speech recognition
                self.status_update.emit("Recognizing...")
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    # Save as WAV file
                    import scipy.io.wavfile
                    scipy.io.wavfile.write(tmp_file.name, self.config.SAMPLE_RATE, audio_data)
                    
                    # Perform transcription
                    segments, _ = self.model.transcribe(
                        tmp_file.name,
                        language=self.language,  # None for auto-detection
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    # Combine all segments
                    text = " ".join([segment.text for segment in segments]).strip()
                    
                    # Delete the temporary file
                    os.unlink(tmp_file.name)
                
                if text:
                    self.text_recognized.emit(text)
                    self.status_update.emit("Recognition complete")
                else:
                    self.status_update.emit("Waiting for speech...")
                    
            except queue.Empty:
                continue  # Queue is empty, keep waiting
            except Exception as e:
                print(f"[{self.session_id[:8]}] ❌ Speech recognition error: {e}")
                self.status_update.emit(f"Error: {str(e)[:30]}")
    
    def stop(self):
        """Stop speech recognition"""
        self.is_running = False
        self.status_update.emit("Speech recognition stopped")

class StreamTranslator(QThread):
    """Stream Translation Thread - Independent thread for translation"""
    
    translation_chunk = pyqtSignal(str)  # Translation chunk signal
    translation_complete = pyqtSignal(str)  # Complete translation signal
    status_update = pyqtSignal(str)     # Status update signal
    
    def __init__(self, config: Config, text_queue: queue.Queue, session_id: str):
        super().__init__()
        self.config = config
        self.text_queue = text_queue
        self.session_id = session_id
        self.is_running = False
        self.source_lang = "Auto-detect"
        self.target_lang = "Chinese"
        self.scene_context = ""  # Scene/domain context
        self.client = None
        self.translation_history = []
        
        # Sliding window memory
        self.last_sentence = {
            "original": "",
            "translation": "",
            "timestamp": ""
        }
        
        # Language display name mapping
        self.lang_display_map = {
            "Auto-detect": "Auto-detect",
            "Chinese": "Chinese",
            "English": "English",
            "Japanese": "Japanese",
            "Korean": "Korean",
            "French": "French",
            "Spanish": "Spanish",
            "Portuguese": "Portuguese",
            "German": "German",
            "Italian": "Italian",
            "Russian": "Russian",
            "Vietnamese": "Vietnamese",
            "Thai": "Thai",
            "Arabic": "Arabic",
            "Indonesian": "Indonesian",
            "Malay": "Malay",
            "Turkish": "Turkish",
            "Dutch": "Dutch",
            "Polish": "Polish",
            "Czech": "Czech",
            "Swedish": "Swedish",
            "Romanian": "Romanian",
            "Hindi": "Hindi",
            "Bengali": "Bengali",
            "Persian": "Persian",
            "Urdu": "Urdu",
            "Greek": "Greek",
            "Finnish": "Finnish",
            "Danish": "Danish",
            "Hungarian": "Hungarian"
        }
    
    def set_languages(self, source_lang: str, target_lang: str, scene_context: str = ""):
        """Set the translation language and scene context"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.scene_context = scene_context.strip()
        
        # Clear history
        self.translation_history = []
        self.last_sentence = {
            "original": "",
            "translation": "",
            "timestamp": ""
        }
        print(f"[{self.session_id[:8]}] ✅ Language settings updated: {source_lang} → {target_lang} | Scene: {scene_context}")
    
    def initialize_client(self):
        """Initialize the Ollama client"""
        try:
            self.client = ollama.Client()
            # Test the connection
            self.client.list()
            print(f"[{self.session_id[:8]}] ✅ Ollama client initialized successfully")
        except Exception as e:
            print(f"[{self.session_id[:8]}] ❌ Ollama connection failed: {e}")
            raise
    
    def build_prompt(self, text: str) -> str:
        """Build the translation prompt - Absolute instruction format to prevent hallucinations"""
        # Get language display names
        source_lang_display = self.lang_display_map.get(self.source_lang, self.source_lang)
        target_lang_display = self.lang_display_map.get(self.target_lang, self.target_lang)
        
        # Build the scene context part
        scene_part = f"Current scene: {self.scene_context}\n" if self.scene_context else ""
        
        # Absolute instruction format, strip away redundant context, prevent hallucinations
        prompt = f"""<|im_start|>system
You are a professional translation machine.  Please strictly follow these instructions:

1. Translation rules:
   - Translate from [{source_lang_display}] to [{target_lang_display}]
   - Only output the translation result, do not explain
   - Do not output pinyin
   - Do not continue the conversation
   - Do not output any punctuation or line breaks other than the translation
   - Maintain accurate meaning and natural language

2. Additional requirements:
   - If it is professional terminology, please maintain accuracy
   - If it is colloquial expression, please translate it into natural colloquial language
   - Absolutely do not add any of your own comments or explanations

{scene_part}<|im_end|>
<|im_start|>user
Please translate the following text:

{text}

<|im_end|>
<|im_start|>assistant
"""
        
        return prompt
    
    def run(self):
        """Run the translation thread"""
        self.is_running = True
        
        # Initialize Ollama client
        try:
            self.initialize_client()
        except Exception as e:
            self.status_update.emit(f"Ollama error: {str(e)[:30]}")
            return
        
        self.status_update.emit("Translation ready")
        
        while self.is_running:
            try:
                # Get text from the queue (with timeout)
                text_item = self.text_queue.get(timeout=0.1)
                
                if text_item is None:  # Stop signal
                    break
                    
                # Extract text and timestamp
                text, timestamp = text_item
                
                if not text.strip():
                    continue
                
                # Start translation
                self.status_update.emit("Translating...")
                
                # Build the prompt
                prompt = self.build_prompt(text)
                
                # Streaming translation - think=False disables the thinking process
                full_translation = ""
                try:
                    # Call Ollama for streaming generation
                    response = self.client.generate(
                        model=self.config.TRANSLATION_MODEL,
                        prompt=prompt,
                        stream=True,
                        think=False,  # Forcefully disable the thinking process
                        options={
                            "temperature": 0.1,  # Low temperature ensures accuracy
                            "top_p": 0.9,
                            "top_k": 40,
                            "repeat_penalty": 1.1  # Prevent repetition
                        }
                    )
                    
                    # Process the streaming response
                    for chunk in response:
                        if 'response' in chunk:
                            chunk_text = chunk['response']
                            full_translation += chunk_text
                            # Send the translation chunk in real-time
                            self.translation_chunk.emit(chunk_text)
                            
                    # Translation complete
                    if full_translation.strip():
                        # Clean up the translation result (remove possible brackets or extra content)
                        cleaned_translation = self.clean_translation(full_translation.strip())
                        
                        # Save to history
                        self.translation_history.append({
                            "timestamp": timestamp,
                            "source": text,
                            "translation": cleaned_translation,
                            "source_lang": self.source_lang,
                            "target_lang": self.target_lang,
                            "scene": self.scene_context
                        })
                        
                        # Update sliding window memory
                        self.last_sentence = {
                            "original": text,
                            "translation": cleaned_translation,
                            "timestamp": timestamp
                        }
                        
                        # Send complete translation signal
                        self.translation_complete.emit(cleaned_translation)
                        self.status_update.emit("Translation complete")
                    else:
                        self.status_update.emit("Translation is empty")
                        
                except Exception as e:
                    print(f"[{self.session_id[:8]}] ❌ Translation error: {e}")
                    # Send error information
                    self.translation_complete.emit(f"[Translation error: {str(e)[:30]}]")
                    self.status_update.emit(f"Translation error: {str(e)[:30]}")
                    
            except queue.Empty:
                continue  # Queue is empty, keep waiting
            except Exception as e:
                print(f"[{self.session_id[:8]}] ❌ Translation thread error: {e}")
                self.status_update.emit(f"Thread error: {str(e)[:30]}")
    
    def clean_translation(self, translation: str) -> str:
        """Clean up the translation result - Remove possible hallucination content"""
        # Remove common hallucination markers
        patterns_to_remove = [
            r'^[\[\{\(].*?[\]\}\)]\s*',  # Opening brackets
            r'\s*[\[\{\(].*?[\]\}\)]$',  # Closing brackets
            r'^Translation[：:]\s*',  # Opening "Translation:" marker
            r'^译文[：:]\s*',  # Opening "译文:" marker
            r'^[A-Za-z]+[：:]\s*',  # Opening English labels
        ]
        
        cleaned = translation
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra spaces and line breaks
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def stop(self):
        """Stop translation"""
        self.is_running = False
        self.status_update.emit("Translation stopped")
    
    def get_translation_history(self):
        """Get translation history"""
        return self.translation_history
    
    def clear_history(self):
        """Clear history and sliding window"""
        self.translation_history = []
        self.last_sentence = {
            "original": "",
            "translation": "",
            "timestamp": ""
        }

class TranslationPipeline(QThread):
    """Translation Pipeline Manager - Coordinates all threads"""
    
    # Signals
    original_text_signal = pyqtSignal(str)  # Original text signal
    translation_chunk_signal = pyqtSignal(str)  # Translation chunk signal
    translation_complete_signal = pyqtSignal(str)  # Complete translation signal
    status_signal = pyqtSignal(str)  # Status signal
    error_signal = pyqtSignal(str)   # Error signal
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.is_running = False
        self.current_session_id = ""  # Current session ID
        
        # Create queues
        self.audio_queue = queue.Queue(maxsize=config.AUDIO_QUEUE_SIZE)
        self.text_queue = queue.Queue(maxsize=config.TEXT_QUEUE_SIZE)
        
        # Initialize components
        self.audio_recorder = None
        self.speech_recognizer = None
        self.translator = None
        
        # Data storage
        self.text_buffer = ""  # Text buffer
        self.buffer_start_time = 0  # Buffer start time
        self.subtitles = []  # Subtitle history
        
        # Language settings
        self.source_lang = "Auto-detect"
        self.target_lang = "Chinese"
        self.scene_context = ""  # Scene/domain context
        
        # Audio device
        self.audio_device_id = None
    
    def audio_callback(self, audio_data: np.ndarray):
        """Audio callback function - Put audio data into the queue"""
        if self.is_running and not self.audio_queue.full():
            self.audio_queue.put(audio_data)
    
    def start_pipeline(self, device_id: Optional[int] = None):
        """Start the pipeline"""
        # Generate a new session ID
        self.current_session_id = str(uuid.uuid4())
        print(f"[{self.current_session_id[:8]}] 🚀 Starting new translation session")
        
        self.is_running = True
        
        # 1. Start audio recording thread
        self.audio_recorder = AudioRecorder(self.config, device_id)
        self.audio_recorder.audio_ready.connect(self.audio_callback)
        self.audio_recorder.start()
        
        # 2. Start speech recognition thread
        self.speech_recognizer = SpeechRecognizer(self.config, self.audio_queue, self.current_session_id)
        self.speech_recognizer.text_recognized.connect(self.handle_recognized_text)
        self.speech_recognizer.status_update.connect(self.status_signal)
        self.speech_recognizer.set_language(self.config.LANGUAGES.get(self.source_lang))
        self.speech_recognizer.start()
        
        # 3. Start translation thread
        self.translator = StreamTranslator(self.config, self.text_queue, self.current_session_id)
        self.translator.translation_chunk.connect(self.handle_translation_chunk)
        self.translator.translation_complete.connect(self.handle_translation_complete)
        self.translator.status_update.connect(self.status_signal)
        self.translator.set_languages(self.source_lang, self.target_lang, self.scene_context)
        self.translator.start()
        
        # Initialize the buffer time
        self.buffer_start_time = time.time()
        
        self.status_signal.emit("Pipeline started")
    
    def handle_recognized_text(self, text: str):
        """Process the recognized text - Issue 1: Precise punctuation segmentation"""
        if not text.strip():
            return
            
        # If the buffer is empty, record the start time
        if not self.text_buffer:
            self.buffer_start_time = time.time()
        
        # Add to the buffer, using strip() to remove whitespace interference
        self.text_buffer += " " + text.strip()
        
        # Check if the timeout has been reached (fallback mechanism)
        current_time = time.time()
        buffer_age = current_time - self.buffer_start_time
        
        # Check if the buffer is too long (character count fallback)
        buffer_too_long = len(self.text_buffer) > self.config.BUFFER_MAX_CHARS
        
        # Check if the timeout has been reached (time fallback)
        buffer_timeout = buffer_age > self.config.BUFFER_TIMEOUT_SECONDS
        
        # Issue 1: First attempt precise punctuation segmentation
        sentences = self.split_by_punctuation(self.text_buffer)
        
        # If there are complete sentences (sentence count greater than 1) or a fallback is needed
        if len(sentences) > 1:
            # Issue 1: Process all complete sentences (except the last one)
            for i in range(len(sentences) - 1):
                sentence = sentences[i].strip()
                if sentence:  # Non-empty sentence
                    print(f"[{self.current_session_id[:8]}] ✅ Punctuation segmentation successful: [{sentence[:30]}...]")
                    self.send_sentence_to_translation(sentence)
            
            # Keep the last incomplete sentence
            self.text_buffer = sentences[-1].strip() if sentences and sentences[-1].strip() else ""
            
            # Reset the buffer time
            if self.text_buffer:
                self.buffer_start_time = current_time
            else:
                self.buffer_start_time = 0
                
        elif buffer_too_long or buffer_timeout:
            # Fallback mechanism: Force send the entire buffer
            sentence = self.text_buffer.strip()
            if sentence:
                reason = "Character limit exceeded" if buffer_too_long else f"Timeout of {buffer_age:.1f}s"
                print(f"[{self.current_session_id[:8]}] ⚠️ Fallback mechanism triggered: {reason}, buffer length={len(self.text_buffer)}")
                self.send_sentence_to_translation(sentence)
                self.text_buffer = ""
                self.buffer_start_time = 0
        else:
            # No complete sentences, do not send yet
            pass
    
    def split_by_punctuation(self, text: str) -> List[str]:
        """Split text based on punctuation - Issue 1: Precise semantic segmentation"""
        if not text:
            return []
        
        # Issue 1: Use simple punctuation detection to avoid over-segmentation with regular expressions
        sentences = []
        current_sentence = ""
        
        # Flag to indicate if inside quotes (avoid segmentation within quotes)
        in_quotes = False
        quote_char = None
        
        for i, char in enumerate(text):
            current_sentence += char
            
            # Handle quotes
            if char in ['"', "'", '「', '」', '『', '』', '《', '》']:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            
            # Issue 1: Detect end-of-sentence punctuation (when not inside quotes)
            if not in_quotes and char in self.config.SENTENCE_ENDINGS:
                # Check if there is a space or end of string after punctuation
                if i == len(text) - 1 or text[i+1].isspace():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Add the last part (if any)
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def send_sentence_to_translation(self, sentence: str):
        """Send a sentence to translation"""
        if not sentence.strip():
            return
            
        # Send original text signal
        self.original_text_signal.emit(sentence)
        
        # Put the text into the translation queue
        timestamp = datetime.now().strftime("%H:%M:%S")
        if not self.text_queue.full():
            self.text_queue.put((sentence, timestamp))
        else:
            print(f"[{self.current_session_id[:8]}] ⚠️ Text queue is full, dropping text")
    
    def handle_translation_chunk(self, chunk: str):
        """Process the translation chunk"""
        # Session check: Ensure the current session is still active
        if not self.is_running:
            return
        self.translation_chunk_signal.emit(chunk)
    
    def handle_translation_complete(self, translation: str):
        """Process the completed translation"""
        # Session check: Ensure the current session is still active
        if not self.is_running:
            return
            
        if translation and translation.strip():
            self.translation_complete_signal.emit(translation)
    
    def set_languages(self, source_lang: str, target_lang: str, scene_context: str = ""):
        """Set the translation language and scene context"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.scene_context = scene_context
        
        # Clear the text buffer
        self.text_buffer = ""
        self.buffer_start_time = 0
        
        # Update speech recognition language
        if self.speech_recognizer:
            self.speech_recognizer.set_language(self.config.LANGUAGES.get(source_lang))
        
        # Update translation language and scene
        if self.translator:
            self.translator.set_languages(source_lang, target_lang, scene_context)
    
    def set_audio_device(self, device_id: int):
        """Set the audio device"""
        self.audio_device_id = device_id
    
    def switch_audio_device(self, device_id: int):
        """Switch audio device - Hot reload audio recording"""
        if not self.is_running:
            return
        
        print(f"[{self.current_session_id[:8]}] 🔄 Switching audio device to: {device_id}")
        
        # Stop the current audio recording thread
        if self.audio_recorder:
            self.audio_recorder.stop()
            self.audio_recorder.wait()
        
        # Start a new audio recording thread
        self.audio_recorder = AudioRecorder(self.config, device_id)
        self.audio_recorder.audio_ready.connect(self.audio_callback)
        self.audio_recorder.start()
        
        self.status_signal.emit(f"Audio device switched")
    
    def run(self):
        """Run the pipeline"""
        self.start_pipeline(self.audio_device_id)
        
        # Keep the thread running
        while self.is_running:
            time.sleep(0.1)
    
    def stop(self):
        """Stop the pipeline"""
        print(f"[{self.current_session_id[:8]}] 🛑 Stopping translation session")
        self.is_running = False
        
        # Clear all queues
        self.clear_queues()
        
        # Send stop signals to queues
        self.audio_queue.put(None)
        self.text_queue.put(None)
        
        # Stop all threads
        if self.audio_recorder:
            self.audio_recorder.stop()
            self.audio_recorder.wait()
        
        if self.speech_recognizer:
            self.speech_recognizer.stop()
            self.speech_recognizer.wait()
        
        if self.translator:
            self.translator.stop()
            self.translator.wait()
        
        # Reset the session ID
        self.current_session_id = ""
        
        self.status_signal.emit("Pipeline stopped")
    
    def clear_queues(self):
        """Clear all queues"""
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear the text queue
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear the buffer
        self.text_buffer = ""
        self.buffer_start_time = 0
        if self.current_session_id:
            print(f"[{self.current_session_id[:8]}] ✅ All queues cleared")
    
    def save_subtitles(self, file_path: str):
        """Save subtitles to file"""
        try:
            if self.translator:
                history = self.translator.get_translation_history()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Realtime Translation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Source Language: {self.source_lang} → Target Language: {self.target_lang}\n")
                    if self.scene_context:
                        f.write(f"Scene/Domain: {self.scene_context}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for i, item in enumerate(history, 1):
                        f.write(f"{i}. [{item['timestamp']}]\n")
                        f.write(f"   Original: {item['source']}\n")
                        f.write(f"   Translation: {item['translation']}\n")
                        if item.get('scene'):
                            f.write(f"   Scene: {item['scene']}\n")
                        f.write("\n")
                
                return True
            else:
                return False
        except Exception as e:
            print(f"❌ Save subtitles failed: {e}")
            return False
    
    def clear_subtitles(self):
        """Clear subtitle records"""
        self.subtitles = []
        self.text_buffer = ""
        self.buffer_start_time = 0
        if self.translator:
            self.translator.clear_history()

class FloatingWindow(QWidget):
    """Floating Subtitle Window - Issue 4: Truly always-on-top"""
    
    def __init__(self, config: Config):
        # Issue 4: Do not pass a parent, make it an independent top-level window
        super().__init__(None)  # Key fix: parent=None
        
        self.config = config
        self.init_ui()
        self.setup_window_properties()
        
        # History
        self.original_history = []
        self.translation_history = []
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("🎬 Realtime Simultaneous Translation")
        
        # Set window style
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
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # Title bar
        title_layout = QHBoxLayout()
        title_label = QLabel("🎬 Realtime Simultaneous Translation V4.1")
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        
        # Close button
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
        
        # Splitter layout (left and right panels)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Original text display
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)
        
        original_label = QLabel("📝 Original")
        original_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(original_label)
        
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        self.original_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.original_text.setMaximumHeight(self.config.FLOATING_WINDOW_HEIGHT - 100)
        left_layout.addWidget(self.original_text)
        
        splitter.addWidget(left_widget)
        
        # Right: Translated text display
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)
        
        translation_label = QLabel("🌐 Translation")
        translation_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(translation_label)
        
        self.translation_text = QTextEdit()
        self.translation_text.setReadOnly(True)
        self.translation_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.translation_text.setMaximumHeight(self.config.FLOATING_WINDOW_HEIGHT - 100)
        right_layout.addWidget(self.translation_text)
        
        splitter.addWidget(right_widget)
        
        # Set splitter sizes
        splitter.setSizes([450, 450])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #90EE90; font-size: 14px;")
        status_layout.addWidget(self.status_label)
        
        # Scroll button
        scroll_btn = QPushButton("🔄 Sync Scroll")
        scroll_btn.setFixedHeight(25)
        scroll_btn.clicked.connect(self.sync_scroll)
        status_layout.addWidget(scroll_btn)
        
        # Clear button
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.setFixedHeight(25)
        clear_btn.clicked.connect(self.clear_all)
        status_layout.addWidget(clear_btn)
        
        main_layout.addLayout(status_layout)
        
        self.setLayout(main_layout)
    
    def setup_window_properties(self):
        """Set window properties - Issue 4: Truly always-on-top"""
        # Issue 4: Most powerful top-level flags
        self.setWindowFlags(
            Qt.Window |  # Normal window type
            Qt.FramelessWindowHint |   # No border
            Qt.WindowStaysOnTopHint |  # Always on top
            Qt.Tool  # Tool window (prevents being obscured by other apps on macOS)
        )
        
        # Window size
        self.resize(self.config.FLOATING_WINDOW_WIDTH, 
                   self.config.FLOATING_WINDOW_HEIGHT)
        
        # Window opacity - Issue 3: Ensure opacity value is between 0.0-1.0
        self.setWindowOpacity(self.config.OPACITY)
        
        # Initial position (bottom right of the screen)
        screen_geometry = QApplication.desktop().screenGeometry()
        self.move(
            screen_geometry.width() - self.width() - 20,
            screen_geometry.height() - self.height() - 100
        )
    
    def mousePressEvent(self, event):
        """Mouse press event - Support dragging"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Mouse move event - Drag the window"""
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
    
    def update_opacity(self, opacity: float):
        """Update window opacity - Issue 3: Ensure the value is between 0.0-1.0"""
        # Issue 3: Make sure the slider value is converted to a float between 0.0 and 1.0
        if opacity < 0.0:
            opacity = 0.0
        elif opacity > 1.0:
            opacity = 1.0
        self.setWindowOpacity(opacity)
    
    def update_subtitle(self, original_text: str):
        """Update original text display"""
        if not original_text.strip():
            return
            
        # Add to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_text = f"[{timestamp}] {original_text}"
        self.original_history.append(formatted_text)
        
        # Limit history length
        if len(self.original_history) > self.config.MAX_HISTORY_LINES:
            self.original_history.pop(0)
        
        # Update display
        self.original_text.setText("\n\n".join(self.original_history))
        
        # Auto scroll to the bottom
        self.original_text.moveCursor(QTextCursor.End)
    
    def update_translation_chunk(self, chunk: str):
        """Update streaming translation chunk"""
        if not chunk.strip():
            return
            
        # Get the current translation text
        current_text = self.translation_text.toPlainText()
        lines = current_text.split("\n")
        
        if lines and lines[-1].startswith("[Translating]"):
            # Update the last line
            lines[-1] = f"[Translating] {chunk}"
        else:
            # Add a new line
            timestamp = datetime.now().strftime("%H:%M:%S")
            lines.append(f"[Translating] {chunk}")
        
        # Limit history length
        if len(lines) > self.config.MAX_HISTORY_LINES:
            lines = lines[-self.config.MAX_HISTORY_LINES:]
        
        self.translation_history = lines.copy()
        self.translation_text.setText("\n".join(lines))
        self.translation_text.moveCursor(QTextCursor.End)
    
    def update_translation_complete(self, translation: str):
        """Update complete translation"""
        if not translation.strip() or translation.startswith("[Translation error"):
            return
            
        # Get the current translation text
        current_text = self.translation_text.toPlainText()
        lines = current_text.split("\n")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Find and replace the last line that starts with "[Translating]"
        for i in range(len(lines)-1, -1, -1):
            if "[Translating]" in lines[i]:
                lines[i] = f"[{timestamp}] {translation}"
                break
        else:
            # If not found, add a new line
            lines.append(f"[{timestamp}] {translation}")
        
        # Limit history length
        if len(lines) > self.config.MAX_HISTORY_LINES:
            lines = lines[-self.config.MAX_HISTORY_LINES:]
        
        self.translation_history = lines.copy()
        self.translation_text.setText("\n".join(lines))
        self.translation_text.moveCursor(QTextCursor.End)
        
        # Update status
        self.status_label.setText(f"✓ {timestamp} Translation complete")
    
    def update_status(self, status: str):
        """Update status display"""
        self.status_label.setText(status)
    
    def sync_scroll(self):
        """Synchronize scrolling of original and translated text"""
        original_scrollbar = self.original_text.verticalScrollBar()
        translation_scrollbar = self.translation_text.verticalScrollBar()
        
        # Get the original scroll position
        original_value = original_scrollbar.value()
        original_max = original_scrollbar.maximum()
        
        # Calculate the ratio and set the translation scroll position
        if original_max > 0:
            ratio = original_value / original_max
            translation_max = translation_scrollbar.maximum()
            translation_scrollbar.setValue(int(ratio * translation_max))
    
    def clear_all(self):
        """Clear all content"""
        self.original_history = []
        self.translation_history = []
        self.original_text.clear()
        self.translation_text.clear()
        self.status_label.setText("Cleared")

class ControlWindow(QMainWindow):
    """Control Window - Fix initialization order and UI issues"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Key fix: Must create child components first, then initialize the UI
        self.floating_window = FloatingWindow(config)
        self.translation_pipeline = TranslationPipeline(config)
        
        # Then initialize the UI
        self.init_ui()
        
        # Connect signals
        self.connect_signals()
        
        # System tray
        self.setup_system_tray()
        
        # Show floating window
        self.floating_window.show()
    
    def init_ui(self):
        """Initialize the UI - Issue 2: Remove the white text box at the bottom"""
        self.setWindowTitle("Professional Realtime Translator V4.1 - Console")
        self.setGeometry(100, 100, 650, 550)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # 1. Language settings group
        lang_group = QGroupBox("🌐 Language Settings")
        lang_layout = QVBoxLayout()
        
        # Source language
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source Language:"))
        self.source_lang_combo = QComboBox()
        for lang_name in self.config.LANGUAGES.keys():
            self.source_lang_combo.addItem(lang_name)
        self.source_lang_combo.setCurrentText("English")
        source_layout.addWidget(self.source_lang_combo)
        lang_layout.addLayout(source_layout)
        
        # Target language
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Language:"))
        self.target_lang_combo = QComboBox()
        for lang_name, lang_code in self.config.LANGUAGES.items():
            if lang_name != "Auto-detect":
                self.target_lang_combo.addItem(lang_name)
        self.target_lang_combo.setCurrentText("Chinese")
        target_layout.addWidget(self.target_lang_combo)
        lang_layout.addLayout(target_layout)
        
        lang_group.setLayout(lang_layout)
        main_layout.addWidget(lang_group)
        
        # Scene/domain input box
        scene_group = QGroupBox("🎭 Scene/Domain Settings")
        scene_layout = QVBoxLayout()
        
        self.scene_input = QLineEdit()
        self.scene_input.setPlaceholderText("e.g., MIT Neuroscience, Korean Drama, Casual Chat...")
        self.scene_input.setToolTip("Enter the scene or domain to optimize translation quality (optional)")
        scene_layout.addWidget(self.scene_input)
        
        scene_note = QLabel("💡 Hint: Entering a scene description can make the translation more relevant.")
        scene_note.setStyleSheet("color: #666; font-size: 12px;")
        scene_note.setWordWrap(True)
        scene_layout.addWidget(scene_note)
        
        scene_group.setLayout(scene_layout)
        main_layout.addWidget(scene_group)
        
        # 2. Audio settings group
        audio_group = QGroupBox("🎤 Audio Settings")
        audio_layout = QVBoxLayout()
        
        # Audio device selection
        audio_layout.addWidget(QLabel("Audio Input Source:"))
        self.audio_device_combo = QComboBox()
        self.refresh_audio_devices()
        audio_layout.addWidget(self.audio_device_combo)
        
        # Audio device notes
        audio_note = QLabel("💡 Recommendation:")
        audio_note.setStyleSheet("color: #666; font-size: 12px;")
        audio_note.setWordWrap(True)
        audio_note.setText("• Listen to system audio: Select BlackHole 2ch\n• Microphone recording: Select Built-in Microphone")
        audio_layout.addWidget(audio_note)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # Floating window control group
        floating_group = QGroupBox("🪟 Floating Window Control")
        floating_layout = QVBoxLayout()
        
        # Show/hide buttons
        floating_btn_layout = QHBoxLayout()
        
        self.show_floating_btn = QPushButton("👁️ Show Floating Window")
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
        
        self.hide_floating_btn = QPushButton("👁️‍🗨️ Hide Floating Window")
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
        
        # Opacity slider
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        
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
        
        # 3. Control button group
        control_group = QGroupBox("🕹️ Control")
        control_layout = QVBoxLayout()
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("▶️ Start Translation")
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
        
        self.stop_button = QPushButton("⏸️ Stop Translation")
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
        
        # Operation buttons
        operation_layout = QHBoxLayout()
        
        clear_btn = QPushButton("🗑️ Clear Subtitles")
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
        
        self.save_button = QPushButton("💾 Save Subtitles")
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
        
        # 4. Status display group - Issue 2: Remove the white text box at the bottom
        status_group = QGroupBox("📊 Status Information")
        status_layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Status: Waiting to start")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        status_layout.addWidget(self.status_label)
        
        # Progress indicator
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
        
        # Issue 2: Remove the white text box at the bottom
        # No longer add self.info_text, directly end the status group
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Bottom note
        footer_label = QLabel("💡 Tip: The floating window can be dragged, and right-click the system tray icon to exit the program")
        footer_label.setStyleSheet("color: #888; font-size: 12px; padding: 10px;")
        footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer_label)
        
        central_widget.setLayout(main_layout)
    
    def refresh_audio_devices(self):
        """Refresh the audio device list"""
        self.audio_device_combo.clear()
        
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Only show input devices
                    device_name = device['name']
                    # Highlight BlackHole device
                    if 'blackhole' in device_name.lower():
                        self.audio_device_combo.addItem(f"{i}: {device_name} (Recommended: System Audio)", i)
                    elif 'built-in' in device_name.lower():
                        self.audio_device_combo.addItem(f"{i}: {device_name} (Built-in Microphone)", i)
                    else:
                        self.audio_device_combo.addItem(f"{i}: {device_name}", i)
        except Exception as e:
            print(f"❌ Failed to get audio devices: {e}")
            self.audio_device_combo.addItem("Default Device", None)
        
        # Select the first device by default
        if self.audio_device_combo.count() > 0:
            self.audio_device_combo.setCurrentIndex(0)
    
    def connect_signals(self):
        """Connect signals and slots"""
        # Translation pipeline signals
        self.translation_pipeline.original_text_signal.connect(
            self.floating_window.update_subtitle
        )
        self.translation_pipeline.translation_chunk_signal.connect(
            self.floating_window.update_translation_chunk
        )
        self.translation_pipeline.translation_complete_signal.connect(
            self.floating_window.update_translation_complete
        )
        
        # Status bar signals
        self.translation_pipeline.status_signal.connect(self.update_status)
        self.translation_pipeline.error_signal.connect(self.show_error)
        
        # Language change signals
        self.source_lang_combo.currentTextChanged.connect(self.on_language_changed)
        self.target_lang_combo.currentTextChanged.connect(self.on_language_changed)
        self.scene_input.textChanged.connect(self.on_scene_changed)
        
        # Audio device change signals
        self.audio_device_combo.currentIndexChanged.connect(self.on_audio_device_changed)
        
        # Fix: Bind button signals here (floating_window already exists)
        self.show_floating_btn.clicked.connect(self.floating_window.show)
        self.hide_floating_btn.clicked.connect(self.floating_window.hide)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        self.start_button.clicked.connect(self.start_translation)
        self.stop_button.clicked.connect(self.stop_translation)
        self.save_button.clicked.connect(self.save_subtitles)
        
        # Clear button
        clear_btn = self.findChild(QPushButton, None)
        if clear_btn and clear_btn.text() == "🗑️ Clear Subtitles":
            clear_btn.clicked.connect(self.clear_all)
    
    def on_opacity_changed(self, value: int):
        """Handle opacity slider changes - Issue 3"""
        self.opacity_label.setText(f"{value}%")
        # Issue 3: Ensure the slider value is divided by 100.0 to get a float between 0.0 and 1.0
        opacity_value = value / 100.0
        self.floating_window.update_opacity(opacity_value)
    
    def on_language_changed(self):
        """Handle language change"""
        source_lang = self.source_lang_combo.currentText()
        target_lang = self.target_lang_combo.currentText()
        scene_context = self.scene_input.text().strip()
        
        # Update pipeline language settings
        self.translation_pipeline.set_languages(source_lang, target_lang, scene_context)
        
        # Update status display
        status_text = f"Language settings updated: {source_lang} → {target_lang}"
        if scene_context:
            status_text += f" | Scene: {scene_context}"
        self.update_status(status_text)
    
    def on_scene_changed(self):
        """Handle scene description changes"""
        self.on_language_changed()
    
    def on_audio_device_changed(self):
        """Handle audio device change"""
        if self.audio_device_combo.currentData() is None:
            return
            
        audio_device_index = self.audio_device_combo.currentData()
        device_name = self.audio_device_combo.currentText()
        
        # If translation is in progress, hot-swap audio device
        if self.translation_pipeline.isRunning():
            self.translation_pipeline.switch_audio_device(audio_device_index)
            self.update_status(f"Audio device switched to: {device_name}")
    
    def update_status(self, status: str):
        """Update the status display"""
        self.status_label.setText(f"Status: {status}")
        
        # Update progress bar based on status
        if "Recognizing" in status:
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
        elif "Translating" in status:
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
        elif "Complete" in status or "Ready" in status:
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
        elif "Error" in status:
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
        
        # Update the floating window status
        self.floating_window.update_status(status)
    
    def show_error(self, error: str):
        """Show an error message"""
        QMessageBox.critical(self, "Error", error)
        self.update_status(f"Error: {error}")
    
    def start_translation(self):
        """Start translation"""
        try:
            # Get settings
            source_lang = self.source_lang_combo.currentText()
            target_lang = self.target_lang_combo.currentText()
            scene_context = self.scene_input.text().strip()
            audio_device_index = self.audio_device_combo.currentData()
            
            # Check Ollama connection
            try:
                import ollama
                client = ollama.Client()
                models = client.list()
                
                # Robust model check
                model_found = False
                for model_info in models['models']:
                    model_name = model_info.get('model', '')
                    if 'qwen3.5' in model_name.lower() and '9b' in model_name.lower():
                        model_found = True
                        print(f"✅ Found compatible model: {model_name}")
                        break
                
                if not model_found:
                    QMessageBox.warning(self, "Warning", 
                        "qwen3.5:9b model or compatible model not found\nRun: ollama pull qwen3.5:9b")
                    return
                    
            except Exception as e:
                QMessageBox.critical(self, "Ollama Error", 
                    f"Failed to connect to Ollama: {e}\nMake sure Ollama service is running.")
                return
            
            # Clear all caches and history
            self.floating_window.clear_all()
            self.translation_pipeline.clear_subtitles()
            
            # Set the translation pipeline
            self.translation_pipeline.set_languages(source_lang, target_lang, scene_context)
            self.translation_pipeline.set_audio_device(audio_device_index)
            
            # Start the pipeline
            self.translation_pipeline.start()
            
            # Update button states
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.save_button.setEnabled(False)
            
            status_text = f"Translation started: {source_lang} → {target_lang}"
            if scene_context:
                status_text += f" | Scene: {scene_context}"
            self.update_status(status_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Start failed: {str(e)}")
    
    def stop_translation(self):
        """Stop translation"""
        # Stop the pipeline
        self.translation_pipeline.stop()
        
        # Wait for the pipeline to completely stop
        self.translation_pipeline.wait()
        
        # Update button states
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)
        
        self.update_status("Translation stopped")
        
        # Ask if save is needed
        reply = QMessageBox.question(
            self, 'Save Subtitles',
            'Do you want to save the translation log?\n\nSelecting "No" will clear the record (use at your own risk).',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self.save_subtitles()
        else:
            self.translation_pipeline.clear_subtitles()
            self.floating_window.clear_all()
            self.update_status("Subtitles cleared (use at your own risk)")
    
    def clear_all(self):
        """Clear all content"""
        self.floating_window.clear_all()
        self.translation_pipeline.clear_subtitles()
        self.update_status("All content cleared")
    
    def save_subtitles(self):
        """Save subtitles to file"""
        # Generate default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_lang = self.source_lang_combo.currentText()
        target_lang = self.target_lang_combo.currentText()
        default_name = f"translation_{source_lang}_{target_lang}_{timestamp}.txt"
        default_path = self.config.OUTPUT_DIR / default_name
        
        # Select save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Subtitle File",
            str(default_path),
            "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            success = self.translation_pipeline.save_subtitles(file_path)
            if success:
                QMessageBox.information(self, "Success", f"Subtitles saved to:\n{file_path}")
                self.update_status(f"Subtitles saved: {Path(file_path).name}")
            else:
                QMessageBox.critical(self, "Error", "Save failed, there may be no translation history")
    
    def setup_system_tray(self):
        """Set up the system tray"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        
        self.tray_icon = QSystemTrayIcon(self)
        
        # Create tray menu
        tray_menu = QMenu()
        
        show_action = QAction("Show Console", self)
        show_action.triggered.connect(self.showNormal)
        tray_menu.addAction(show_action)
        
        hide_action = QAction("Hide Console", self)
        hide_action.triggered.connect(self.hide)
        tray_menu.addAction(hide_action)
        
        show_floating_action = QAction("Show Floating Window", self)
        show_floating_action.triggered.connect(self.floating_window.show)
        tray_menu.addAction(show_floating_action)
        
        hide_floating_action = QAction("Hide Floating Window", self)
        hide_floating_action.triggered.connect(self.floating_window.hide)
        tray_menu.addAction(hide_floating_action)
        
        tray_menu.addSeparator()
        
        exit_action = QAction("Exit Program", self)
        exit_action.triggered.connect(self.close_all)
        tray_menu.addAction(exit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
    
    def close_all(self):
        """Close all windows and exit"""
        if self.translation_pipeline.isRunning():
            self.translation_pipeline.stop()
            self.translation_pipeline.wait()
        self.floating_window.close()
        QApplication.quit()
    
    def closeEvent(self, event):
        """Close event"""
        if self.translation_pipeline.isRunning():
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                'Translation is in progress, are you sure you want to exit?\n\nExiting will stop all translation and close the floating window.',
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

# ============== Part 5: Main Function ==============
def main():
    """Main function"""
    print("=" * 60)
    print("🎬 Professional Realtime Translator V4.1 - Ultimate Fix Version")
    print("=" * 60)
    print("💡 Instructions:")
    print("1. Make sure Ollama service is running: ollama serve")
    print("2. Make sure you have downloaded the model: ollama pull qwen3.5:9b")
    print("3. Select the source language and target language (supports 29 languages)")
    print("4. Optionally, enter a scene/domain prompt to optimize translation")
    print("5. Select the audio input source (BlackHole 2ch to listen to system audio)")
    print("6. Click Start Translation")
    print("7. You can switch language, scene, and audio device at any time during runtime")
    print("8. Use the floating window controls to adjust opacity and show/hide")
    print("9. The floating window will display the original and translated text in two columns")
    print("=" * 60)
    
    # Check Ollama
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        
        # Robust model check
        model_found = False
        for model_info in models['models']:
            model_name = model_info.get('model', '')
            if 'qwen3.5' in model_name.lower() and '9b' in model_name.lower():
                model_found = True
                print(f"✅ Found compatible model: {model_name}")
                break
        
        if not model_found:
            print("⚠️ qwen3.5:9b model or compatible model not found")
            print("Please run: ollama pull qwen3.5:9b")
            print("If already installed, please verify the model name")
            
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("Please make sure Ollama is installed and running:")
        print("1. Visit https://ollama.ai to download and install")
        print("2. Run in a new terminal: ollama serve")
        print("3. Run in another terminal: ollama pull qwen3.5:9b")
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Professional Realtime Translator V4.1")
    app.setApplicationDisplayName("Professional Realtime Translator V4.1")
    
    # Create configuration and control window
    config = Config()
    control_window = ControlWindow(config)
    control_window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()