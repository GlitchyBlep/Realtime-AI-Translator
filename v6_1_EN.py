"""
🎬 Professional Realtime Interpreter V6.0 - WebRTC VAD Refactored Edition
Functionality: Real-time monitoring of system audio/microphone → Intelligent VAD segmentation → Zero-shot translation → Left/right split-screen display
Architecture: Multi-threaded queue + C++ backend inference engine + WebRTC VAD
Engines: whisper.cpp + llama.cpp (Tencent's Mixed Translation Model)
Refactoring Focus:
  1. WebRTC VAD intelligent voice activity detection (mode=3, 500ms silence segmentation)
  2. Explicit memory management, solving repeated OOM issues
  3. Minimal Prompt adaptation for 1.8B small models
Usage: WHISPER_CPP_LIB=./libwhisper.dylib python3 v6_1_EN.py
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

# ============== Part 1: Import All Libraries ==============
try:
    # Audio Processing
    import sounddevice as sd
    import numpy as np
    
    # Speech Recognition - whisper.cpp
    from whisper_cpp_python import Whisper
    
    # Large Model Translation - llama.cpp
    from llama_cpp import Llama
    
    # GUI Interface
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                                 QPushButton, QComboBox, QVBoxLayout, QHBoxLayout,
                                 QGroupBox, QTextEdit, QFileDialog, QLineEdit,
                                 QMessageBox, QSystemTrayIcon, QMenu, QAction,
                                 QSplitter, QFrame, QSlider, QDial, QProgressBar)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QPoint
    from PyQt5.QtGui import QIcon, QFont, QTextCursor, QColor
    
    # WebRTC VAD voice activity detection
    import webrtcvad
    
    # Audio format conversion
    import wave
    import struct
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please run: pip install sounddevice numpy whisper-cpp-python llama-cpp-python PyQt5 webrtcvad scipy")
    sys.exit(1)

# ============== Part 2: System Configuration ==============
@dataclass
class Config:
    """System Configuration - V6.1: WebRTC VAD Optimized Edition (with forced fallback mechanism)"""
    
    # Audio Settings
    SAMPLE_RATE: int = 16000  # Sample rate required by Whisper and VAD
    CHANNELS: int = 1  # Mono
    BIT_DEPTH: int = 16  # 16-bit depth
    
    # WebRTC VAD Settings - V6.1 optimized
    VAD_MODE: int = 3  # 0-3, 3 is the most sensitive, least false positives
    VAD_FRAME_DURATION_MS: int = 30  # Frame duration in milliseconds
    
    # V6.1 refactored: Silence timeout changed to 300ms (for ultra-responsive output)
    VAD_SILENCE_TIMEOUT_MS: int = 300  # Considered sentence-end after 300ms silence
    
    # V6.1 added: Forced fallback mechanism to prevent infinite audio block growth
    VAD_MAX_SPEECH_MS: int = 5000  # Maximum speech duration of 5 seconds (forced fallback to prevent hangs)
    
    VAD_MIN_SPEECH_MS: int = 100  # Minimum speech duration in milliseconds
    
    # VAD frame size calculation
    FRAME_SIZE: int = int(SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)  # Number of samples per frame
    
    # Queue Settings
    AUDIO_QUEUE_SIZE: int = 10  # Maximum size of the audio queue
    TEXT_QUEUE_SIZE: int = 9999  # Maximum size of the text queue
    
    # C++ Engine Settings
    WHISPER_MODEL_PATH: str = "models/ggml-small.bin"  # whisper.cpp model path
    LLM_MODEL_PATH: str = "models/HY-MT1.5-1.8B-Q4_K_M.gguf"  # llama.cpp model path
    WHISPER_THREADS: int = 4  # whisper inference threads
    LLM_THREADS: int = 4      # llama inference threads
    LLM_CONTEXT_SIZE: int = 2048  # Context window size
    LLM_N_GPU_LAYERS: int = 0    # Number of GPU layers (0 for pure CPU)
    
    # UI Settings
    FLOATING_WINDOW_WIDTH: int = 900  # Window width (1/3 of the screen)
    FLOATING_WINDOW_HEIGHT: int = 300  # Fixed height
    FONT_SIZE: int = 18  # Font size
    OPACITY: float = 0.9  # Window opacity
    MAX_HISTORY_LINES: int = 50  # Maximum number of history lines on each side
    
    # Language Support
    LANGUAGES = {
        "Auto-detect": "auto",
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
    """Audio Recording Thread - WebRTC VAD optimized Edition (with forced fallback mechanism)"""
    
    audio_ready = pyqtSignal(np.ndarray)  # Audio data signal
    
    def __init__(self, config: Config, device_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.device_id = device_id
        self.is_running = False
        
        # WebRTC VAD initialization
        self.vad = webrtcvad.Vad(config.VAD_MODE)
        
        # Voice activity detection status
        self.speech_buffer = []  # Current speech block buffer
        self.silence_frames = 0  # Consecutive silence frame counter
        self.speech_start_time = 0  # Speech start time
        
        # Audio Stream
        self.stream = None
        
    def pcm_to_wav_format(self, audio_data: np.ndarray) -> bytes:
        """Convert PCM audio data to 16-bit little-endian format required by VAD"""
        # Ensure it's mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        # Ensure it's 16-bit integer
        if audio_data.dtype != np.int16:
            # Normalize to [-32768, 32767]
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Convert to bytes
        pcm_bytes = audio_data.tobytes()
        return pcm_bytes
    
    def run(self):
        """Run the audio recording thread - V6.1 refactored: Add forced fallback mechanism"""
        self.is_running = True
        print(f"✅ Audio recording thread started, device ID: {self.device_id}")
        print(f"🔊 VAD configuration: mode={self.config.VAD_MODE}")
        print(f"🕐 Silence timeout={self.config.VAD_SILENCE_TIMEOUT_MS}ms")
        print(f"🛑 Forced fallback={self.config.VAD_MAX_SPEECH_MS}ms")
        
        try:
            # Start the audio stream
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
                # Read one frame of audio
                audio_frame, overflowed = self.stream.read(self.config.FRAME_SIZE)
                
                if overflowed:
                    print("⚠️ Audio buffer overflow")
                
                # Convert to PCM format for VAD
                pcm_data = self.pcm_to_wav_format(audio_frame)
                
                # VAD detection: determine if the current frame contains voice
                try:
                    is_speech = self.vad.is_speech(pcm_data, self.config.SAMPLE_RATE)
                except Exception as e:
                    # VAD may be sensitive to certain frame formats, handle errors conservatively
                    is_speech = False
                
                # VAD state machine
                if is_speech:
                    # Detected speech
                    self.silence_frames = 0  # Reset silence counter
                    
                    # If it's the beginning of the speech, record the time
                    if len(self.speech_buffer) == 0:
                        self.speech_start_time = time.time()
                    
                    # Add to the speech buffer
                    self.speech_buffer.append(audio_frame.copy())
                    
                else:
                    # Detected silence
                    self.silence_frames += 1
                    
                    # If there's speech already, also add the silence frame to the buffer (to avoid truncation)
                    if len(self.speech_buffer) > 0:
                        self.speech_buffer.append(audio_frame.copy())
                    
                    # Calculate the duration of the silence
                    silence_duration_ms = self.silence_frames * self.config.VAD_FRAME_DURATION_MS
                
                # ============ V6.1 Core Refactoring Logic ============
                # Calculate the current duration of the speech buffer in milliseconds
                current_speech_duration_ms = len(self.speech_buffer) * self.config.VAD_FRAME_DURATION_MS
                
                # Determine whether to send audio (double segmentation mechanism)
                should_send_audio = False
                reason = ""
                
                # Condition A: Silence timeout segmentation (normal breathing segmentation)
                if len(self.speech_buffer) > 0 and is_speech == False:
                    if silence_duration_ms >= self.config.VAD_SILENCE_TIMEOUT_MS:
                        should_send_audio = True
                        reason = f"Silence segmentation ({silence_duration_ms:.0f}ms silence)"
                
                # Condition B: Forced fallback segmentation (anti-hang mechanism)
                if current_speech_duration_ms >= self.config.VAD_MAX_SPEECH_MS:
                    should_send_audio = True
                    reason = f"Reached {self.config.VAD_MAX_SPEECH_MS}ms fallback segmentation"
                
                # If either segmentation condition is met, send the audio immediately
                if should_send_audio:
                    # Check if it meets the minimum speech length requirement
                    if current_speech_duration_ms >= self.config.VAD_MIN_SPEECH_MS:
                        # Combine all audio frames
                        speech_audio = np.concatenate(self.speech_buffer, axis=0)
                        
                        # Ensure it's mono
                        if len(speech_audio.shape) > 1:
                            speech_audio = speech_audio[:, 0]
                        
                        # Convert to 16-bit PCM format
                        speech_audio = np.clip(speech_audio, -1.0, 1.0)
                        speech_audio = (speech_audio * 32767).astype(np.int16)
                        
                        # Send the complete semantic block, with segmentation reason log
                        print(f"🎤 VAD detected complete statement: {current_speech_duration_ms:.0f}ms audio - {reason}")
                        self.audio_ready.emit(speech_audio)
                    
                    # Clear the buffer and reset the state
                    self.speech_buffer = []
                    self.silence_frames = 0
                
                # Avoid excessive CPU usage
                time.sleep(0.001)
                
        except Exception as e:
            print(f"❌ Audio recording failed: {e}")
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
    
    def stop(self):
        """Stop audio recording"""
        self.is_running = False
        print("🛑 Audio recording thread stopped")

class SpeechRecognizer(QThread):
    """Speech Recognition Thread - whisper.cpp version (adapted to the latest API)"""
    
    text_recognized = pyqtSignal(str)  # Recognized text signal
    status_update = pyqtSignal(str)     # Status update signal
    
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
        """Initialize the whisper.cpp model"""
        try:
            # Check if the model file exists
            model_path = Path(self.config.WHISPER_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"whisper.cpp model file does not exist: {model_path}")
            
            # Initialize whisper.cpp (remove print_progress parameter)
            self.model = Whisper(
                model_path=str(model_path),
                n_threads=self.config.WHISPER_THREADS,
            )
            print(f"[{self.session_id[:8]}] ✅ whisper.cpp model loaded successfully: {model_path.name}")
        except Exception as e:
            print(f"[{self.session_id[:8]}] ❌ whisper.cpp model loading failed: {e}")
            raise
    
    def save_audio_to_temp_wav(self, audio_data: np.ndarray) -> str:
        """Save audio data to a temporary wav file"""
        # Create a temporary file
        temp_wav = tempfile.NamedTemporaryFile(
            dir=self.temp_dir,
            suffix=".wav",
            delete=False
        )
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        # Write to wav file
        with wave.open(temp_wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16 bits = 2 bytes
            wav_file.setframerate(self.config.SAMPLE_RATE)
            
            # Ensure the audio data is 16-bit
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_wav_path
    
    def set_language(self, language: str):
        """Set the recognition language"""
        # whisper.cpp language code mapping
        language_map = {
            "Auto-detect": "auto",
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
        self.language = language_map.get(language, "auto")
    
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
                
                # Save to a temporary wav file (must pass the file path)
                wav_path = self.save_audio_to_temp_wav(audio_data)
                
                try:
                    # Use transcribe method and pass the file path (latest API requirement)
                    result = self.model.transcribe(
                        wav_path,
                        language=self.language
                    )
                    
                    # Process the return result
                    text = ""
                    if isinstance(result, dict):
                        text = result.get("text", "").strip()
                    elif isinstance(result, str):
                        text = result.strip()
                    else:
                        # Try to extract text from possible structures
                        text = str(result).strip()
                    
                    if text:
                        self.text_recognized.emit(text)
                        self.status_update.emit("Recognition complete")
                    else:
                        self.status_update.emit("Waiting for speech...")
                        
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(wav_path)
                    except:
                        pass
                
            except queue.Empty:
                continue  # Queue is empty, keep waiting
            except Exception as e:
                print(f"[{self.session_id[:8]}] ❌ Speech recognition error: {e}")
                self.status_update.emit(f"Recognition error: {str(e)[:30]}")
    
    def stop(self):
        """Stop speech recognition - Explicitly release memory"""
        self.is_running = False
        self.status_update.emit("Speech recognition stopped")
        
        # Explicitly release the model
        if self.model:
            print(f"[{self.session_id[:8]}] 🧹 Releasing whisper.cpp model memory")
            del self.model
            self.model = None
        
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
        
        # Force garbage collection
        gc.collect()
        
        print(f"[{self.session_id[:8]}] ✅ Speech recognition resources released")

class StreamTranslator(QThread):
    """Streaming Translation Thread - llama.cpp version (Minimal Zero-shot mode)"""
    
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
        self.model = None
        
        # V6.0 refactored: Remove all context caching logic
        # Use Zero-shot mode, each translation is independent
    
    def set_languages(self, source_lang: str, target_lang: str):
        """Set the translation language"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        print(f"[{self.session_id[:8]}] ✅ Language settings updated: {source_lang} → {target_lang}")
    
    def initialize_model(self):
        """Initialize the llama.cpp model"""
        try:
            # Check if the model file exists
            model_path = Path(self.config.LLM_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"llama.cpp model file does not exist: {model_path}")
            
            # Initialize llama.cpp
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.LLM_CONTEXT_SIZE,
                n_threads=self.config.LLM_THREADS,
                n_gpu_layers=self.config.LLM_N_GPU_LAYERS,
                verbose=False
            )
            print(f"[{self.session_id[:8]}] ✅ llama.cpp model loaded successfully: {model_path.name}")
        except Exception as e:
            print(f"[{self.session_id[:8]}] ❌ llama.cpp model loading failed: {e}")
            raise
    
    def build_prompt(self, text: str) -> str:
        """Build translation prompt - Minimal Zero-shot mode"""
        # V6.0 refactored: Use minimal Prompt to adapt to 1.8B small model
        # Remove all context caching, implement Zero-shot translation
        
        prompt = f"""You are a machine translation engine. Output ONLY the translation. No explanations, no filler words, no repetitions.

Translate this to {self.target_lang}: {text}

Translation:"""
        
        return prompt
    
    def run(self):
        """Run the translation thread"""
        self.is_running = True
        
        # Initialize model
        try:
            self.initialize_model()
        except Exception as e:
            self.status_update.emit(f"Model error: {str(e)[:30]}")
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
                
                # Build minimal prompt
                prompt = self.build_prompt(text)
                
                # llama.cpp streaming generation
                full_translation = ""
                try:
                    # Call llama.cpp for streaming generation
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
                    
                    # Process the streaming response
                    for output in stream:
                        chunk_text = output['choices'][0]['text']
                        full_translation += chunk_text
                        # Send the translation chunk in real-time
                        self.translation_chunk.emit(chunk_text)
                        
                    # Translation complete
                    if full_translation.strip():
                        # Clean up the translation result
                        cleaned_translation = self.clean_translation(full_translation.strip())
                        
                        # Send the complete translation signal
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
        """Clean up translation result - Remove possible hallucination content"""
        # Remove common hallucination markers
        patterns_to_remove = [
            r'^[\[\{\(].*?[\]\}\)]\s*',  # Opening brackets
            r'\s*[\[\{\(].*?[\]\}\)]$',  # Closing brackets
            r'^Translation[：:]\s*',  # Opening "Translation:" marker
            r'^译文[：:]\s*',  # Opening "译文:" marker
            r'^[A-Za-z]+[：:]\s*',  # Opening English labels
            r'^Assistant:\s*',  # Opening Assistant marker
            r'^assistant:\s*',  # Lowercase assistant marker
            r'^\d+\.\s*',  # Opening numbered list
        ]
        
        cleaned = translation
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra spaces and line breaks
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def stop(self):
        """Stop translation - Explicitly release memory"""
        self.is_running = False
        self.status_update.emit("Translation stopped")
        
        # Explicitly release the model
        if self.model:
            print(f"[{self.session_id[:8]}] 🧹 Releasing llama.cpp model memory")
            del self.model
            self.model = None
        
        # Force garbage collection
        gc.collect()
        
        print(f"[{self.session_id[:8]}] ✅ Translation resources released")

class TranslationPipeline(QThread):
    """Translation Pipeline Manager - Coordinates all threads"""
    
    # Signal Definitions
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
        
        # Language settings
        self.source_lang = "Auto-detect"
        self.target_lang = "Chinese"
        
        # Audio devices
        self.audio_device_id = None
    
    def audio_callback(self, audio_data: np.ndarray):
        """Audio callback function - Put audio data into the queue"""
        if self.is_running and not self.audio_queue.full():
            self.audio_queue.put(audio_data)
    
    def start_pipeline(self, device_id: Optional[int] = None):
        """Start the entire pipeline"""
        # Generate a new session ID
        self.current_session_id = str(uuid.uuid4())
        print(f"[{self.current_session_id[:8]}] 🚀 Starting new translation session")
        print(f"[{self.current_session_id[:8]}] 🔧 Language settings: {self.source_lang} → {self.target_lang}")
        
        self.is_running = True
        
        # 1. Start audio recording thread
        self.audio_recorder = AudioRecorder(self.config, device_id)
        self.audio_recorder.audio_ready.connect(self.audio_callback)
        self.audio_recorder.start()
        
        # 2. Start speech recognition thread
        self.speech_recognizer = SpeechRecognizer(self.config, self.audio_queue, self.current_session_id)
        self.speech_recognizer.text_recognized.connect(self.handle_recognized_text)
        self.speech_recognizer.status_update.connect(self.status_signal)
        self.speech_recognizer.set_language(self.source_lang)
        self.speech_recognizer.start()
        
        # 3. Start translation thread
        self.translator = StreamTranslator(self.config, self.text_queue, self.current_session_id)
        self.translator.translation_chunk.connect(self.handle_translation_chunk)
        self.translator.translation_complete.connect(self.handle_translation_complete)
        self.translator.status_update.connect(self.status_signal)
        self.translator.set_languages(self.source_lang, self.target_lang)
        self.translator.start()
        
        # Initialize buffer time
        self.buffer_start_time = time.time()
        
        self.status_signal.emit("Pipeline started")
    
    def handle_recognized_text(self, text: str):
        """Process the recognized text - Direct send to translation"""
        if not text.strip():
            return
        
        # Send the original text signal
        self.original_text_signal.emit(text)
        
        # Put the text into the translation queue
        timestamp = datetime.now().strftime("%H:%M:%S")
        if not self.text_queue.full():
            self.text_queue.put((text, timestamp))
        else:
            print(f"[{self.current_session_id[:8]}] ⚠️ Text queue is full, dropping text")
    
    def handle_translation_chunk(self, chunk: str):
        """Process the translation chunk"""
        # Session check: ensure the current session is still active
        if not self.is_running:
            return
        self.translation_chunk_signal.emit(chunk)
    
    def handle_translation_complete(self, translation: str):
        """Process the completed translation text"""
        # Session check: make sure the current session is still active
        if not self.is_running:
            return
            
        if translation and translation.strip():
            self.translation_complete_signal.emit(translation)
    
    def set_languages(self, source_lang: str, target_lang: str):
        """Set translation language"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Clear text buffer
        self.text_buffer = ""
        self.buffer_start_time = 0
        
        # Update speech recognition language
        if self.speech_recognizer:
            self.speech_recognizer.set_language(source_lang)
        
        # Update translation language
        if self.translator:
            self.translator.set_languages(source_lang, target_lang)
    
    def set_audio_device(self, device_id: int):
        """Set audio device"""
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
        
        # Send stop signals to the queues
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
        
        # Reset session ID
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

# ============== Part 4: GUI Interface ==============
class FloatingWindow(QWidget):
    """Floating Subtitle Window - Independent top-level window"""
    
    def __init__(self, config: Config):
        # Do not pass a parent, make it an independent top-level window
        super().__init__(None)
        
        self.config = config
        self.init_ui()
        self.setup_window_properties()
        
        # History
        self.original_history = []
        self.translation_history = []
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("🎬 Realtime Interpreter V6.0")
        
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
        title_label = QLabel("🎬 Realtime Interpreter V6.0")
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
        """Set window properties"""
        # Window top-most property
        self.setWindowFlags(
            Qt.Window |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        
        # Window size
        self.resize(self.config.FLOATING_WINDOW_WIDTH, 
                   self.config.FLOATING_WINDOW_HEIGHT)
        
        # Window opacity
        self.setWindowOpacity(self.config.OPACITY)
        
        # Initial position (bottom right of the screen)
        screen_geometry = QApplication.desktop().screenGeometry()
        self.move(
            screen_geometry.width() - self.width() - 20,
            screen_geometry.height() - self.height() - 100
        )
    
    def mousePressEvent(self, event):
        """Mouse press event - Supports dragging"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Mouse move event - Drag the window"""
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
    
    def update_opacity(self, opacity: float):
        """Update window opacity"""
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
            
        # Get current translation text
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
            
        # Get current translation text
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
    """Control Window"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Must create child components first, then initialize the UI
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
        """Initialize UI"""
        self.setWindowTitle("Professional Realtime Interpreter V6.0 - WebRTC VAD Refactored Edition")
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
        
        # V6.0 refactored: Remove scene/domain settings (minimal mode)
        
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
        audio_note.setText("• Listen to system audio: Select BlackHole 2ch\n• Microphone recording: Select Built-in Microphone\n• V6.0 feature: WebRTC VAD intelligent segmentation")
        audio_layout.addWidget(audio_note)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # 3. Model settings group
        model_group = QGroupBox("🤖 Model Settings")
        model_layout = QVBoxLayout()
        
        # whisper.cpp model path
        whisper_layout = QHBoxLayout()
        whisper_layout.addWidget(QLabel("Whisper Model:"))
        self.whisper_model_path = QLineEdit(self.config.WHISPER_MODEL_PATH)
        self.whisper_model_path.setPlaceholderText("ggml-base.bin or ggml-small.bin")
        whisper_layout.addWidget(self.whisper_model_path)
        
        whisper_btn = QPushButton("Browse")
        whisper_btn.clicked.connect(self.browse_whisper_model)
        whisper_layout.addWidget(whisper_btn)
        
        model_layout.addLayout(whisper_layout)
        
        # llama.cpp model path
        llm_layout = QHBoxLayout()
        llm_layout.addWidget(QLabel("Translation Model:"))
        self.llm_model_path = QLineEdit(self.config.LLM_MODEL_PATH)
        self.llm_model_path.setPlaceholderText("HY-MT1.5-1.8B-GGUF-q8_0.gguf")
        llm_layout.addWidget(self.llm_model_path)
        
        llm_btn = QPushButton("Browse")
        llm_btn.clicked.connect(self.browse_llm_model)
        llm_layout.addWidget(llm_btn)
        
        model_layout.addLayout(llm_layout)
        
        # Model notes
        model_note = QLabel("💡 V6.0 Features: Explicit memory management to solve repeated OOM issues\nDownload models: whisper.cpp models from https://huggingface.co/ggerganov/whisper.cpp/tree/main\nTencent's Mixed Translation Model: https://huggingface.co/Tencent/HY-MT1.5-1.8B-GGUF")
        model_note.setStyleSheet("color: #666; font-size: 12px;")
        model_note.setWordWrap(True)
        model_layout.addWidget(model_note)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
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
        
        self.hide_floating_btn = QPushButton("👁️​🗨️ Hide Floating Window")
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
        
        # 4. Control button group
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
        
        # 5. Status display group
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
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Bottom note
        footer_label = QLabel("💡 V6.0 features: WebRTC VAD intelligent segmentation + Explicit memory management + Minimal Zero-shot translation\nTip: Drag the floating window around, right-click the system tray icon to exit the program")
        footer_label.setStyleSheet("color: #888; font-size: 12px; padding: 10px;")
        footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer_label)
        
        central_widget.setLayout(main_layout)
    
    def browse_whisper_model(self):
        """Browse for the whisper.cpp model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select whisper.cpp model file",
            str(Path.home()),
            "Model Files (*.bin);;All Files (*.*)"
        )
        if file_path:
            self.whisper_model_path.setText(file_path)
            self.config.WHISPER_MODEL_PATH = file_path
    
    def browse_llm_model(self):
        """Browse for the llama.cpp model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select llama.cpp model file",
            str(Path.home()),
            "GGUF Models (*.gguf *.bin);;All Files (*.*)"
        )
        if file_path:
            self.llm_model_path.setText(file_path)
            self.config.LLM_MODEL_PATH = file_path
    
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
        
        # Language change signals
        self.source_lang_combo.currentTextChanged.connect(self.on_language_changed)
        self.target_lang_combo.currentTextChanged.connect(self.on_language_changed)
        
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
        """Handle opacity slider changes"""
        self.opacity_label.setText(f"{value}%")
        # Ensure the slider value is converted to a float between 0.0 and 1.0
        opacity_value = value / 100.0
        self.floating_window.update_opacity(opacity_value)
    
    def on_language_changed(self):
        """Handle language change"""
        source_lang = self.source_lang_combo.currentText()
        target_lang = self.target_lang_combo.currentText()
        
        # Update pipeline language settings
        self.translation_pipeline.set_languages(source_lang, target_lang)
        
        # Update status display
        status_text = f"Language settings updated: {source_lang} → {target_lang}"
        self.update_status(status_text)
    
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
        """Update status display"""
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
    
    def start_translation(self):
        """Start translation"""
        try:
            # Get settings
            source_lang = self.source_lang_combo.currentText()
            target_lang = self.target_lang_combo.currentText()
            audio_device_index = self.audio_device_combo.currentData()
            
            # Check model file exists
            whisper_model_path = Path(self.config.WHISPER_MODEL_PATH)
            llm_model_path = Path(self.config.LLM_MODEL_PATH)
            
            if not whisper_model_path.exists():
                QMessageBox.critical(self, "Model File Error", 
                    f"whisper.cpp model file does not exist:\n{whisper_model_path}\n\nDownload from: https://huggingface.co/ggerganov/whisper.cpp/tree/main")
                return
                
            if not llm_model_path.exists():
                QMessageBox.critical(self, "Model File Error", 
                    f"llama.cpp model file does not exist:\n{llm_model_path}\n\nRecommended: HY-MT1.5-1.8B-GGUF-q8_0.gguf\nDownload: https://huggingface.co/Tencent/HY-MT1.5-1.8B-GGUF")
                return
            
            # Clear all caches and history
            self.floating_window.clear_all()
            
            # Set the translation pipeline
            self.translation_pipeline.set_languages(source_lang, target_lang)
            self.translation_pipeline.set_audio_device(audio_device_index)
            
            # Start the pipeline
            self.translation_pipeline.start()
            
            # Update button states
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.save_button.setEnabled(False)
            
            status_text = f"Translation started: {source_lang} → {target_lang}"
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
            'Do you want to save the current translation log?\n\nSelect "No" will clear the record (use at your own risk).',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self.save_subtitles()
        else:
            self.floating_window.clear_all()
            self.update_status("Subtitles cleared (use at your own risk)")
    
    def clear_all(self):
        """Clear all content"""
        self.floating_window.clear_all()
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
            # Get the current window's original and translated text
            original_text = self.floating_window.original_text.toPlainText()
            translation_text = self.floating_window.translation_text.toPlainText()
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Realtime Translation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Source Language: {source_lang} → Target Language: {target_lang}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("Original:\n")
                    f.write(original_text + "\n\n")
                    f.write("Translation:\n")
                    f.write(translation_text + "\n")
                
                QMessageBox.information(self, "Success", f"Subtitles saved to:\n{file_path}")
                self.update_status(f"Subtitles saved: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")
    
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
                'Translation is in progress, do you want to exit?\n\nExiting will stop all translation and close the floating window.',
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
    print("🎬 Professional Realtime Interpreter V6.0 - WebRTC VAD Refactored Edition")
    print("=" * 60)
    print("💡 Usage Instructions:")
    print("1. Install dependencies: pip install sounddevice numpy whisper-cpp-python llama-cpp-python PyQt5 webrtcvad scipy")
    print("2. Download model files:")
    print("   - whisper.cpp model: from https://huggingface.co/ggerganov/whisper.cpp/tree/main")
    print("   - Tencent's Mixed Translation Model: https://huggingface.co/Tencent/HY-MT1.5-1.8B-GGUF")
    print("3. Place the model files in the models/ directory, or select them via the UI")
    print("4. Select the source and target language (supports 29 languages)")
    print("5. Select the audio input source (BlackHole to listen to system audio)")
    print("6. Click Start Translation")
    print("7. V6.0 Features:")
    print("   - WebRTC VAD intelligent segmentation (500ms silence segmentation)")
    print("   - Explicit memory management to solve repeated OOM issues")
    print("   - Minimal Zero-shot translation, adapted to 1.8B small models")
    print("=" * 60)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Professional Realtime Interpreter V6.0")
    app.setApplicationDisplayName("Professional Realtime Interpreter V6.0")
    
    # Create configuration and control window
    config = Config()
    control_window = ControlWindow(config)
    control_window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()