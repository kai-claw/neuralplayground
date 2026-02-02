# 🧬 NeuralPlayground

**Watch neural networks learn in real-time.** Draw digits, train networks, visualize weights and activations — all in the browser, no ML libraries required.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://kai-claw.github.io/neuralplayground/)
[![Tests](https://img.shields.io/badge/tests-250%20passing-brightgreen)]()
[![TypeScript](https://img.shields.io/badge/TS-0%20errors-blue)]()
[![Bundle](https://img.shields.io/badge/bundle-77KB%20gzip-purple)]()

## ✨ Features

### Core Neural Network
- **Custom from-scratch implementation** — No TensorFlow, no ML libraries. Pure TypeScript neural network with backpropagation
- **Configurable architecture** — 1-5 hidden layers, 8-256 neurons per layer
- **3 activation functions** — ReLU, Sigmoid, Tanh (per layer)
- **Xavier weight initialization** + SGD + Softmax + Cross-entropy

### Interactive Visualization
- **🧠 Network Architecture** — Node graph with color-coded weighted connections and signal flow animation
- **✏️ Drawing Canvas** — Draw digits (0-9) with mouse or touch, get live predictions
- **📈 Training Progress** — Dual-axis loss/accuracy chart with real-time updates
- **⚡ Activation Bars** — Per-layer activation magnitude visualization
- **🔥 Weight Heatmap** — Color-coded weight matrices with layer switching
- **🎯 Prediction Bar** — 0-9 probability distribution with confidence glow

### Advanced Features
- **🔬 Feature Maps** — See what each first-layer neuron has learned to detect (28×28 weight grids)
- **🎭 Adversarial Lab** — Add gaussian, salt-pepper, or targeted noise; watch confidence crumble
- **🔀 Digit Morphing** — Blend between two drawn digits and see prediction shift
- **🎬 Cinematic Demo** — Auto-draw all 10 digits with training sequence
- **✨ Signal Flow** — Animated particles flowing through the network on prediction

### Quality
- **250 tests** across 5 test suites (functional, stress, structural)
- **0 TypeScript errors**, 0 build warnings
- **Full ARIA accessibility** — keyboard navigation, screen reader labels, skip links
- **`prefers-reduced-motion`** respected throughout
- **Error boundary** with retry/reload
- **Responsive** — 3-column → 2-column → 1-column layout

## 🚀 Quick Start

```bash
# Install
npm install

# Development
npm run dev

# Run tests
npm test

# Build for production
npm run build

# Deploy to GitHub Pages
npm run deploy
```

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Train / Pause |
| `R` | Reset network |
| `D` | Cinematic demo |
| `H` | Toggle help |
| `Esc` | Close / Stop demo |

## 🏗️ Architecture

```
src/
├── nn/                          # Neural network engine (pure logic, 0 React imports)
│   ├── NeuralNetwork.ts         # Forward/backward pass, training, prediction
│   └── sampleData.ts            # Procedural digit generation, canvasToInput
├── hooks/                       # React state management
│   ├── useNeuralNetwork.ts      # Training loop, state management
│   ├── useCinematic.ts          # Cinematic demo state machine
│   └── useContainerDims.ts      # Responsive ResizeObserver hook
├── components/                  # UI (13 components, all Canvas-based)
│   ├── NetworkVisualizer.tsx     # Network topology + signal flow animation
│   ├── DrawingCanvas.tsx         # Touch/mouse digit drawing
│   ├── LossChart.tsx             # Dual-axis training progress
│   ├── ActivationVisualizer.tsx  # Per-layer activation bars
│   ├── PredictionBar.tsx         # 0-9 probability display
│   ├── ControlPanel.tsx          # Training controls + architecture config
│   ├── WeightPanel.tsx           # Weight heatmap with layer tabs
│   ├── WeightHeatmap.tsx         # Color-coded weight matrix
│   ├── FeatureMaps.tsx           # First-layer weight visualization
│   ├── AdversarialLab.tsx        # Noise lab (gaussian/salt-pepper/targeted)
│   ├── DigitMorph.tsx            # Digit blending lab
│   ├── CinematicBadge.tsx        # Demo mode status badge
│   └── ErrorBoundary.tsx         # Crash recovery
├── data/
│   └── digitStrokes.ts           # Cinematic auto-draw stroke data
├── App.tsx                       # Root orchestrator
├── constants.ts                  # All magic numbers (103 LOC)
├── types.ts                      # Shared type definitions
└── utils.ts                      # Pure math utilities
```

**Data flow**: `nn/` → `hooks/` → `App.tsx` → `components/` (unidirectional)

## 📊 Metrics

| Metric | Value |
|--------|-------|
| Source files | 21 |
| Source LOC | ~3,600 |
| Test LOC | ~2,560 |
| CSS LOC | ~1,929 |
| Tests | 250 (5 suites) |
| TS errors | 0 |
| Build warnings | 0 |
| Bundle JS | 247 KB (77 KB gzip) |
| Bundle CSS | 30 KB (6 KB gzip) |
| Dependencies | React + React DOM only |

## 🎨 Development Process

Built through 6 structured passes using the **Six Thinking Hats** methodology:

1. ⚪ **White Hat** — Facts & audit baseline
2. ⚫ **Black Hat** — Risk mitigation & stability hardening
3. 🟢 **Green Hat** — Creative features (signal flow, feature maps, adversarial lab, morphing, cinematic)
4. 🟡 **Yellow Hat** — Auto-start, animations, micro-interactions
5. 🔴 **Red Hat** — Feel & intuition polish (glows, shadows, warmth)
6. 🔵 **Blue Hat** — Structural tests, architecture validation, process audit

See [AUDIT.md](./AUDIT.md) for the full journey with quantitative growth tables and qualitative assessments.

## 📄 License

MIT
