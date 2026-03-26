## 🐈‍⬛ NanoLingo: 1.19GB Offline AI Translator
Welcome to the bleeding edge of local AI translation.

📖 Looking for the detailed documentation? > If you want to see how this project started and the logic behind it, check out the perfectly documented V4.1 Legacy README here.

## ⚠️ WARNING: ABOUT THE NEW V6.1 SOURCE CODE ⚠️
The code currently in this repo is the raw, unpolished V6.1.

To achieve the 1.19GB RAM limit and zero-latency WebRTC VAD, I had to fight through 14 continuous C++ compilation errors and Metal GPU dependency hell. To be completely honest with you... I don't even fully remember how I fixed all of them. 😵‍💫

But I swear to God, I am actively working hard to fix the plumbing right now! 🪠🛠️ (Clean code is coming in the future, promise!)

Because of this chaotic state, there is no step-by-step installation tutorial for V6.1.

## 🧰 The "Good Luck" Requirements
For the brave souls attempting to run the source code, here is your basic survival kit. You will still need to figure out the Metal/C++ library bindings yourself:

Plaintext
# Basic Dependencies
pip install webrtcvad
pip install pyaudio
pip install numpy
### The Final Boss (You need to compile this with Metal support enabled)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python whisper-cpp-python


*(Note: You might also need to fight with `ffmpeg` and local `whisper` paths. May the compiler be with you. ⚔️)*
## 🚀 For Everyone Else (Highly Recommended)
If you value your weekend and your sanity, I have already compiled, packaged, and optimized the stable V6.2 version into a clean, double-click-to-run macOS .dmg app.

👉 Don't build it. Just run it. Get the ready-to-use .dmg here: > https://liwenchen.gumroad.com/l/nanolingo-beta
