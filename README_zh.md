# 🎬 Realtime Interpreter - 本地AI同传翻译器

> **[📄 View English Version](README.md)**

---
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyQt5-UI-orange.svg" alt="PyQt5">
  <img src="https://img.shields.io/badge/Ollama-LocalAI-green.svg" alt="Ollama">
  <img src="https://img.shields.io/badge/Whisper-ASR-yellow.svg" alt="Whisper">
  <img src="https://img.shields.io/badge/License-GPL3.0-red.svg" alt="License">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/GlitchyBlep/realtime-interpreter/main/screenshots/main_window.png" width="45%" alt="控制台界面">
  <img src="https://raw.githubusercontent.com/GlitchyBlep/realtime-interpreter/main/screenshots/floating_window.png" width="45%" alt="悬浮窗界面">
</p>

## 🐱‍💻 关于作者 / About the Author

**GlitchyBlep** | 一只高糊黑猫🫠，代码小白，全白入门正在黑化……（开玩笑的）

## 📖 项目简介 / Introduction

**Realtime Interpreter** 是一个完全本地部署的 AI 同传翻译工具，能够实时监听系统音频或麦克风输入，通过 Whisper 进行语音识别，再经由本地大模型（如 Qwen）进行流式翻译，最后在独立的置顶悬浮窗中显示双语字幕。

**Key Features:**
- 🏠 **完全本地运行**：无需联网，保护隐私，无数据泄露风险
- ⚡ **低延迟实时翻译**：从语音识别到翻译结果输出通常在 2-4 秒内
- 🪟 **独立置顶悬浮窗**：翻译字幕悬浮在所有窗口之上，不影响正常使用（有点毛病，不过没了再唤起就好了🫠）
- 🌍 **29种语言支持**：从中文到匈牙利语，覆盖全球主流语种
- 🎭 **场景领域优化**：可输入特定领域提示词（如"MIT神经生物学"“韩国爱情剧”“日常闲聊”等…）优化翻译质量
- 🔄 **运行中热重载**：可随时切换语言、场景和音频输入源，无需重启

## 🏗️ 核心架构 / Core Architecture

### 三线程异步解耦架构

```
音频采集线程 → audio_queue → Whisper识别线程 → text_queue → Ollama翻译线程 → UI信号
      ↓               ↓              ↓               ↓              ↓
  独立运行       队列缓冲       独立运行       队列缓冲       独立运行
```

### 关键技术要点

1. **多线程队列解耦**：音频采集、语音识别、大模型翻译分别在独立线程中运行，通过队列传递数据，彻底解决UI阻塞问题
2. **强制 `think=False` 压制幻觉**：针对 Qwen 3.5 的指令遵循问题，强制关闭模型的"思考过程"，使其成为无情的翻译机器
3. **标点语义断句**：基于标点符号（., ?, !, 。, ？, ！）进行智能句子分割，避免机械字数切割导致的语义断裂
4. **防吞字兜底机制**：8秒无标点强制送出缓冲区，防止长段落永久卡死
5. **滑动窗口记忆法**：仅保留最近一句的上下文，防止错误级联和幻觉传染

## 🐛 踩坑与版本演进 / Bug History & Evolution

### Subtitler版本 - 木头字幕员
- **问题**：单线程同步处理，UI卡成PPT，模型问题
- **教训**：qwen3.5的Thinking过程导致翻译过程卡住捕获不到翻译完的字幕

### success qwen2.5版本 - 首次翻译正常版
- **问题**：无法在运行过程中切换UI选项，幽灵线程，本地保存无效，字幕溢出（除了能翻译哪哪都是毛病）
- **教训**：whisper中的自动检测“self.language = None”,”currentTextChanged”信号连接到”on_language_changed”,兼容qwen3.5”think=false”关闭思考，标点整合整句翻译

### V1.0版本 - alpha版本
- **问题**：作为翻译器该行的都行，有点机翻味儿，有点小毛病，不过重启解决一切
- **教训**：添加上下文参考和场景提示词，扩充支持语种（下一版本开始29种），继续修复多线程乱跳问题

### V2.0 - 爆水管版本
- **问题**：翻译启动过程硬件锁死，整句聚合太死板根本不出字了，还有小问题一堆（修漏水水管爆了🫠）
- **修复**：增加防吞字机制（buffer中存留字符限制），增加UI热切换，顺手重构了底层languages字典

### V3.0 - 碎香肠与幻觉传染版本
- **问题1**：120字符强制切断导致语义断裂，翻译质量暴跌
- **问题2**：Qwen 3.5 被上下文带偏，开始"碎碎念"和胡言乱语🫠
- **修复**：
  - 废除机械字数切割，回归标点语义断句
  - 重构 Prompt 为"绝对指令"格式，彻底剥离多余上下文
  - 添加翻译结果清洗函数，移除模型输出的括号和标签

### V4.0 - 透明度滑块隐身Bug
- **问题**：透明度滑块一拉，悬浮窗直接消失
- **修复**：`setWindowOpacity()` 需要 0.0-1.0 的浮点数，而不是 0-100 的整数

### V4.1 - 最终稳定版
- **问题**：初始化顺序错误导致 `AttributeError: 'ControlWindow' object has no attribute 'floating_window'`
- **修复**：确保 `ControlWindow.__init__()` 中先实例化子组件，再初始化UI
- **额外改进**：悬浮窗使用 `Qt.Tool` 标志，防止在 macOS 上被其他应用遮挡（但没啥用🫥）

## 🛠️ 避坑指南 / Geek's Troubleshooting Guide

### Mac 音频通道映射问题
```bash
# 系统音频捕获需要 BlackHole 虚拟声卡
brew install blackhole-2ch

# 系统设置 → 声音 → 输出：选择"多输出设备"
# 系统设置 → 声音 → 输入：选择"BlackHole 2ch"
# 然后在软件中选择对应的音频输入设备
```

### Python GIL 与音频回调阻塞
```python
# ❌ 错误示范：在音频回调中直接进行耗时操作
def audio_callback(indata, frames, time_info, status):
    text = whisper_model.transcribe(indata)  # 阻塞！
    translation = ollama.generate(text)      # 再次阻塞！
    update_ui(translation)                   # UI线程也阻塞！

# ✅ 正确方案：队列缓冲 + 多线程解耦
audio_thread → audio_queue → whisper_thread → text_queue → translation_thread → UI_signal
```

### Qwen 3.5 的指令遵循问题
```python
# ❌ 温和请求（模型容易偷懒）：
"请将以下英文翻译成中文：Hello world"

# ✅ 绝对指令（强制服从）：
"""<|im_start|>system
你是一个专业的翻译机器。请严格遵循以下指令：
1. 将【英文】翻译为【中文】
2. 只输出翻译结果，不要解释
3. 不要输出拼音
4. 不要延续对话
5. 不要输出除了译文之外的任何标点或换行
<|im_end|>
<|im_start|>user
Hello world
<|im_end|>
<|im_start|>assistant
"""
```

## 🚀 快速启动 / Quick Start

### 环境依赖
```bash
# 1. 安装 Python 依赖
pip install sounddevice numpy faster-whisper ollama PyQt5 scipy

# 2. 安装 BlackHole（仅 macOS，用于捕获系统音频）
brew install blackhole-2ch

# 3. 下载模型
ollama pull qwen3.5:9b
# 或者更小的模型
ollama pull qwen2.5:7b
```

### 运行程序
```bash
# 1. 在一个终端启动 Ollama 服务（保持运行）
ollama serve

# 2. 在另一个终端运行翻译器
python realtime_interpreter.py
```

### 配置步骤
1. **选择音频输入**：如果是翻译视频/会议，选择 BlackHole；如果是麦克风输入，选择 Built-in Microphone
2. **选择语言对**：如"英文"→"中文"
3. **（可选）输入场景提示**：如"TED演讲"、"游戏直播"、"医学讲座"
4. **点击"开始翻译"**：悬浮窗会自动出现并显示实时翻译

## 🎯 使用技巧 / Usage Tips

### 最佳实践组合
```yaml
视频翻译:
  音频输入: BlackHole 2ch
  语言: 英文 → 中文
  场景: "YouTube科技评测"

会议记录:
  音频输入: 内置麦克风
  语言: 中文 → 英文
  场景: "商务会议"

游戏直播:
  音频输入: BlackHole 2ch
  语言: 日文 → 中文
  场景: "RPG游戏实况"
```

### 性能调优
```python
# 在 Config 类中调整这些参数：
CHUNK_DURATION = 3.0      # 音频块时长（秒），越小延迟越低，越大识别越准
WHISPER_MODEL_SIZE = "small"  # tiny/base/small/medium，越大越准越慢
TRANSLATION_MODEL = "qwen3.5:9b"  # 7b更快，9b更准
```

## 📄 开源协议 / License

本项目采用 **GPL-3.0** 协议开源：

- ✅ **允许**：自由使用、修改、分发
- ✅ **要求**：修改版本必须开源，且保留原始版权声明
- ❌ **禁止**：私自打包商业化售卖（看可以用可以白嫖也欢迎，卖钱？求求你要点脸谢谢🙏🏻）
- 📢 **倡议**：如果你基于本项目做出了改进，欢迎提交 Pull Request！

## 🤝 贡献指南 / Contributing

1. Fork 本仓库
2. 创建你的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

### 待实现功能
- [ ] 支持更多本地大模型（Llama、DeepSeek等）
- [ ] 导出SRT字幕文件
- [ ] 快捷键控制（显示/隐藏、暂停/继续）
- [ ] 翻译历史回顾与编辑
- [ ] 自定义翻译Prompt模板

## 📞 问题反馈 / Issues

遇到问题？按以下步骤排查：

1. **检查依赖**：`pip list | grep -E "(sounddevice|faster-whisper|ollama|PyQt5)"`
2. **检查模型**：`ollama list` 确认已下载对应模型
3. **检查音频**：系统声音设置是否正确？BlackHole安装了吗？
4. **查看日志**：终端输出的错误信息通常很详细
5. **重启试试**：🫠 "如果故障就重启" - 有时候真的有用

如果还是不行，欢迎在 [Issues](https://github.com/GlitchyBlep/realtime-interpreter/issues) 提交问题，最好包含：
- 操作系统版本
- Python版本
- 完整的终端错误日志
- 你已经尝试过的解决方案

---

<p align="center">
  <i>由一只高糊黑猫和无数杯咖啡共同维护</i><br>
  <i>Maintained by a blurry black cat and countless cups of coffee</i>
</p>
