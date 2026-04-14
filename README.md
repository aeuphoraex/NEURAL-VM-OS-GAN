# Neural VM WebOS - GAN-Powered Generative App Engine

A complete **desktop operating system** running in the browser where a **Generative Adversarial Network (GAN)** creates, executes, and evolves **real programs** from neural latent space.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Browser](https://img.shields.io/badge/browser-chrome%20%7C%20firefox%20%7C%20edge-orange.svg)](https://caniuse.com/)

---

## What Is This?

The Neural VM WebOS is a **zero-dependency, browser-native desktop environment** with a GAN-powered Neural Virtual Machine at its core. Instead of running pre-written code, the VM **generates executable programs from random noise** using a trained neural network, decodes them into bytecode, and runs them as real applications inside draggable, resizable windows.

Think of it as: **GAN + Bytecode VM + WebOS Desktop = Apps that create themselves.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Neural VM WebOS                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Program     │  │  Bytecode    │  │  App         │  │
│  │  Generator   │→ │  Executor    │→ │  Renderer    │  │
│  │  (GAN)       │  │  (VM)        │  │  (Canvas)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                  │                  │          │
│  Latent → Program     21 Opcodes        Windows /       │
│  Space    Bytecode      (DRAW, WAVE,     Gallery /      │
│  (128d)   (hex)         FRACTAL, GAME)   Terminal       │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                    WebOS Desktop                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────────┐ │
│  │ Windows │ │Taskbar  │ │ Start   │ │ System Tray   │ │
│  │ Manager │ │         │ │ Menu    │ │ + Clock       │ │
│  └─────────┘ └─────────┘ └─────────┘ └───────────────┘ │
│                                                          │
├─────────────────────────────────────────────────────────┤
│              GGUF Conversion Pipeline                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  GGUF F16    │→ │  GAN VM      │→ │  Bootstrap   │  │
│  │  Parser      │  │  Converter   │  │  Trainer     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 10 Program Types the Neural VM Generates

| Icon | Type | Description |
|------|------|-------------|
| 🎨 | **Visual Art** | Circles, rects, pixels with neural-sampled colors/positions |
| 🔷 | **Pattern Generator** | Mathematical patterns (sine, cosine, XOR, radial) |
| 〰️ | **Wave Engine** | Sine, square, sawtooth, quadratic waves with frequency/phase/amplitude |
| 🌀 | **Fractal Generator** | Mandelbrot fractals with neural-sampled center coords and iterations |
| 🎮 | **Mini Game** | Animated particle/ball games with physics |
| 📊 | **Data Visualization** | Waveform and signal visualizations |
| 🎵 | **Audio Synth** | Wave synthesis with multiple oscillator types |
| 🧱 | **Texture Generator** | Procedural texture patterns |
| ✨ | **Particle System** | Particle-like visual compositions |
| 🔢 | **Matrix Operations** | Matrix-based mathematical visualizations |

---

## Custom Bytecode ISA (Instruction Set Architecture)

The Neural VM executes a **custom 21-opcode bytecode** with drawing, signal processing, and control flow instructions:

| Opcode | Hex | Description |
|--------|-----|-------------|
| `NOP` | `0x00` | No operation |
| `PUSH` | `0x01` | Push 16-bit value to stack |
| `POP` | `0x02` | Pop from stack |
| `ADD` | `0x03` | Add top two stack values |
| `SUB` | `0x04` | Subtract |
| `MUL` | `0x05` | Multiply |
| `DIV` | `0x06` | Divide (safe) |
| `DUP` | `0x08` | Duplicate top of stack |
| `SWAP` | `0x09` | Swap top two values |
| `LOAD` | `0x0A` | Load from memory |
| `STORE` | `0x0B` | Store to memory |
| `JMP` | `0x0C` | Jump to address |
| `JZ` | `0x0D` | Jump if zero |
| `JNZ` | `0x0E` | Jump if not zero |
| `PIXEL` | `0x20` | Draw pixel (x, y, color) |
| `LINE` | `0x21` | Draw line (x1, y1, x2, y2, color) |
| `RECT` | `0x22` | Draw rectangle (x, y, w, h, color) |
| `CIRCLE` | `0x23` | Draw circle (x, y, r, color) |
| `WAVE` | `0x30` | Generate wave (type, freq, amp, phase, color) |
| `FRACTAL` | `0xF0` | Render Mandelbrot (cx, cy, maxIter) |
| `GAME` | `0xF1` | Spawn mini-game (type, speed, color) |
| `CLEAR` | `0x25` | Clear canvas |
| `HALT` | `0xFF` | Stop execution |

---

## Quick Start

### Option 1: Browser (No Installation)

```
1. Open NEURAL_VM_WEBOS.html in Chrome/Firefox/Edge
2. Click ⚡ Neural VM on the desktop
3. Click "Generate Program" - watch it create and execute
4. Click "Train Generator" to evolve better programs
5. Open App Gallery to browse all generated programs
```

### Option 2: GGUF Conversion Pipeline (Python)

Convert any GGUF model into a GAN-powered Neural VM:

```bash
# Convert your GGUF to GAN VM
python gguf_gan_vm.py your_model.gguf

# Check VM status
python gan_vm_runtime.py your_model_GAN_VM_F16.gguf status

# Run benchmark
python gan_vm_runtime.py your_model_GAN_VM_F16.gguf bench 100

# Train the GAN
python gan_vm_runtime.py your_model_GAN_VM_F16.gguf train 1000

# Bootstrap self-play training
python bootstrap_trainer.py your_model_GAN_VM_F16.gguf 50 200
```

Or use the Windows batch file:
```cmd
convert_to_gan_vm.bat
```

---

## File Structure

```
neural-vm-webos/
├── NEURAL_VM_WEBOS.html       # Full WebOS desktop + generative app engine
├── NEURAL_VM_BROWSER.html     # Single-page Neural VM (no desktop)
├── gguf_gan_vm.py             # GGUF → GAN-powered Neural VM converter
├── gan_vm_runtime.py          # Runtime for executing GAN-VM from GGUF
├── bootstrap_trainer.py       # Self-play bootstrap trainer with curriculum
├── convert_to_gan_vm.bat      # Windows batch converter
└── README.md                  # This file
```

---

## How the Generative System Works

### 1. Latent Space Sampling
The `ProgramGenerator` maintains a 128-dimensional latent space with learned weight matrices mapping latent vectors to:
- **Program type probabilities** (10 types)
- **Instruction distributions** (256 opcodes)
- **Parameter value ranges** (colors, positions, frequencies)

### 2. Neural Decoding to Bytecode
For each generated program:
```
latent[128] → type_weights[10] → softmax → selected_type
latent[128] → program_weights → bytecode[variable length]
```
Each program type has its own decoder function that maps latent dimensions to meaningful program parameters.

### 3. Fitness Scoring
Every program receives a fitness score based on:
- **Bytecode length** (longer = more complex)
- **Opcode diversity** (unique opcodes / total)
- **Validity** (has HALT instruction, no crashes)

### 4. Training Loop
The generator trains by:
- Reinforcing weight patterns that produce high-fitness programs
- Gradient-like updates: `weights += lr * latent * fitness`
- Auto-generating and executing programs every 25 epochs

### 5. Mutation & Evolution
Any program can be mutated:
- **Bytecode mutation**: Random bit flips in instruction stream
- **Latent mutation**: Perturbations in latent space
- Mutated programs are executed and scored, creating an evolutionary loop

### 6. Execution & Rendering
The `BytecodeVM` executes programs with:
- 1024-element stack
- 4096-byte memory
- Visual output as draw command list
- Canvas rendering for display

---

## WebOS Desktop Features

- **Window Manager**: Draggable, resizable, minimizable, maximimizable, closable windows
- **Taskbar**: Shows all open windows, click to focus/minimize
- **Start Menu**: Access to all applications
- **System Tray**: FPS counter + live clock
- **Right-click Context Menu**: Quick access to apps
- **Terminal**: CLI with commands for the Neural VM
- **App Gallery**: Browse all generated programs with preview thumbnails
- **System Info**: Full system specifications

---

## Technical Details

### GAN Architecture
- **Type**: WGAN-GP (Wasserstein GAN with Gradient Penalty)
- **Generator**: 256 → 512 → 512 → 512 → 1024 (4 fully-connected layers)
- **Discriminator**: 1024 → 512 → 512 → 1 (3 fully-connected layers)
- **Precision**: F16 (half-precision IEEE 754)
- **Optimizer**: Adam (beta1=0.0, beta2=0.9)

### Bootstrap Training
- **Replay Buffer**: 10,000 generated samples
- **Curriculum Learning**: 10 levels with progressive difficulty
- **Meta-Learning**: Automatic hyperparameter adaptation
- **Self-Play**: Programs train on their own generated outputs

### GGUF Format
- Full binary parser for GGUF v3
- F16 tensor loading with IEEE 754 half-precision
- Support for all GGUF metadata types (string, i32, f32, f16)

---

## Use Cases

1. **Procedural Art Generation** - Generate unique visual art, patterns, textures
2. **Algorithmic Music** - Create waveforms and audio synthesis programs
3. **Fractal Exploration** - Discover interesting regions of the Mandelbrot set
4. **Game Prototyping** - Generate mini-game concepts with random parameters
5. **Education** - Learn about GANs, bytecode VMs, and latent space decoding
6. **Creative Coding** - Evolve programs through mutation to find interesting outputs

---

## Requirements

### Browser (WebOS)
- Modern browser with Canvas API support
- Chrome, Firefox, Edge recommended
- No server or installation needed

### Python Pipeline
- Python 3.7+
- NumPy
- No deep learning frameworks required

---

## License

MIT License

---

## Acknowledgments

- GGUF format by [ggerganov](https://github.com/ggerganov/ggml)
- WGAN-GP from [Gulrajani et al.](https://arxiv.org/abs/1704.00028)
- Inspired by neural network interpretability and program synthesis research
