# 🧬 NeuralPlayground

**Watch a neural network learn in real-time.** Draw digits, train networks, visualize every weight and activation, attack with adversarial noise, dream with gradient ascent — all in the browser. No TensorFlow, no ML libraries. Just pure TypeScript and math.

[![Live Demo](https://img.shields.io/badge/🚀_demo-live-brightgreen?style=for-the-badge)](https://kai-claw.github.io/neuralplayground/)
[![Tests](https://img.shields.io/badge/tests-331_passing-brightgreen?style=for-the-badge)]()
[![TypeScript](https://img.shields.io/badge/TypeScript-0_errors-blue?style=for-the-badge)]()
[![Bundle](https://img.shields.io/badge/bundle-82.76KB_gzip-purple?style=for-the-badge)]()

<p align="center">
  <strong><a href="https://kai-claw.github.io/neuralplayground/">👉 Try the Live Demo</a></strong>
</p>

---

## ✨ What Can You Do?

### 🎨 Draw & Predict
Draw any digit (0–9) on the canvas. The network predicts in real-time, showing probability distributions across all 10 classes. Watch confidence shift as you draw each stroke.

### 🧠 See the Network Think
A full topology visualization shows every neuron and weighted connection. Color-coded signal flow particles animate through the network when you make a prediction, showing data flowing from input to output.

### 🔬 Feature Maps — What Neurons See
Peer inside the first hidden layer. Each neuron's learned 28×28 weight pattern is rendered as a heatmap, revealing the edge detectors, curve recognizers, and stroke patterns the network has discovered on its own.

### 🎭 Adversarial Noise Lab
Attack the network. Apply gaussian blur, salt-and-pepper static, or targeted adversarial noise to your drawing and watch confidence crumble — or hold. Explore the fragility and resilience of neural networks.

### 🔀 Digit Morphing
Draw two different digits, then blend between them with a slider. Watch the prediction smoothly shift as the input transitions — revealing decision boundaries in real-time.

### 🎬 Cinematic Demo Mode
Sit back. The demo auto-draws all 10 digits with realistic stroke animations, training the network through a full cycle. Perfect for presentations or just watching a network come alive.

### 💭 Network Dreams — Gradient Ascent Visualization
Run the network *backwards*. Starting from random noise, gradient ascent reveals what the network "imagines" each digit looks like — its platonic ideal of a 0, 1, 2, etc. Eerie, beautiful, and deeply informative.

### 🔧 Neuron Surgery
Freeze or kill individual neurons and watch the network compensate — or break. Toggle neurons on and off to understand which ones are critical and which are redundant. Live experimentation with network architecture.

### 🏁 Training Race
Pit two network architectures against each other in a side-by-side training race. Compare shallow vs. deep, narrow vs. wide, ReLU vs. Sigmoid. Choose from presets or build custom configurations. See which converges faster and which generalizes better.

### 📊 Full Training Dashboard
- **Dual-axis loss/accuracy chart** with real-time updates
- **Per-layer activation magnitudes** showing signal strength through the network
- **Weight heatmaps** with layer switching — see the actual learned parameters
- **Prediction probability bars** for all 10 digits with confidence-reactive glow

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Train / Pause training |
| `R` | Reset network weights |
| `D` | Start cinematic demo |
| `H` | Toggle help panel |
| `Esc` | Close panel / Stop demo |

---

## 🏗️ Architecture

```
src/
├── nn/                          # Neural network engine (pure logic, 0 React imports)
│   ├── NeuralNetwork.ts         # Forward/backward pass, training, prediction
│   └── sampleData.ts            # Procedural digit generation, canvasToInput
├── hooks/                       # React state management (4 hooks)
│   ├── useNeuralNetwork.ts      # Training loop, snapshot state
│   ├── useCinematic.ts          # Demo mode state machine
│   ├── useContainerDims.ts      # Responsive ResizeObserver
│   └── useTrainingRace.ts       # Side-by-side race controller
├── components/                  # UI layer (16 components, Canvas-based rendering)
│   ├── NetworkVisualizer.tsx     # Network topology + signal flow particles
│   ├── DrawingCanvas.tsx         # Touch/mouse digit drawing (forwardRef)
│   ├── LossChart.tsx             # Dual-axis training progress chart
│   ├── ActivationVisualizer.tsx  # Per-layer activation bars
│   ├── PredictionBar.tsx         # 0-9 probability distribution
│   ├── ControlPanel.tsx          # Training controls + architecture config
│   ├── WeightPanel.tsx           # Weight heatmap with layer tabs
│   ├── WeightHeatmap.tsx         # Color-coded weight matrix renderer
│   ├── FeatureMaps.tsx           # First-layer weight visualization
│   ├── AdversarialLab.tsx        # Noise lab (gaussian/salt-pepper/targeted)
│   ├── DigitMorph.tsx            # Digit blending lab
│   ├── NetworkDreams.tsx         # Gradient ascent dream visualization
│   ├── NeuronSurgery.tsx         # Freeze/kill neuron interface
│   ├── TrainingRace.tsx          # Side-by-side architecture comparison
│   ├── CinematicBadge.tsx        # Demo mode status badge
│   └── ErrorBoundary.tsx         # Crash recovery with retry
├── data/
│   └── digitStrokes.ts           # Cinematic auto-draw stroke sequences
├── App.tsx                       # Root orchestrator (hooks → components)
├── constants.ts                  # All magic numbers centralized (103 LOC)
├── types.ts                      # Shared TypeScript types
├── utils.ts                      # Pure math (activation, softmax, argmax, RNG)
├── noise.ts                      # Noise generation (gaussian, salt-pepper, targeted)
├── rendering.ts                  # Canvas rendering utilities
└── visualizer.ts                 # Network visualization helpers
```

**Data flow** (unidirectional): `nn/` → `hooks/` → `App.tsx` → `components/`

**Prediction pipeline**: `DrawingCanvas(ImageData)` → `canvasToInput(28×28)` → `NeuralNetwork.predict()` → `{label, probabilities, layers}` → UI components

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | React 19 + TypeScript (strict mode) |
| **Rendering** | HTML5 Canvas (zero DOM-per-neuron) |
| **NN Engine** | Custom from-scratch (forward/backward pass, SGD, cross-entropy, Xavier init) |
| **Build** | Vite 7 |
| **Testing** | Vitest 4 (331 tests across 6 suites) |
| **Deployment** | GitHub Pages via gh-pages |
| **Dependencies** | React + React DOM only. No ML libraries. |

---

## 📊 Metrics

| Metric | Value |
|--------|-------|
| Source files | 31 |
| Source LOC | ~5,100 |
| Test LOC | ~3,500 |
| CSS LOC | ~2,650 |
| Tests | 331 (6 suites) |
| TypeScript errors | 0 |
| Build warnings | 0 |
| Bundle JS | 266 KB (82.76 KB gzip) |
| Bundle CSS | 41 KB (7.78 KB gzip) |
| Components | 16 |
| Hooks | 4 |
| External deps | React + ReactDOM only |

---

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/kai-claw/neuralplayground.git
cd neuralplayground

# Install
npm install

# Development (hot reload)
npm run dev

# Run all 331 tests
npm test

# Production build
npm run build

# Deploy to GitHub Pages
npm run deploy
```

---

## ♿ Accessibility

- Full **ARIA labels** on all interactive elements
- **Keyboard navigation** — all features accessible without mouse
- **`prefers-reduced-motion`** — all animations respect system settings
- **Skip links** and semantic HTML structure
- **Error boundary** with retry and reload options
- **Noscript fallback** for JS-disabled browsers

---

## 🎩 Development Process — Six Thinking Hats (10 Passes)

This project was built through 10 structured passes using Edward de Bono's **Six Thinking Hats** methodology. Each pass applied a different cognitive lens:

| Pass | Hat | Focus | Key Deliverables |
|------|-----|-------|-----------------|
| 1 | ⚪ White | Facts & Audit | CI/CD, SEO, PWA manifest, 60 tests |
| 2 | ⚫ Black | Risks & Problems | NaN guards, error boundary, ARIA, keyboard shortcuts, 39 tests |
| 3 | 🟢 Green | Creative Features | Signal flow, cinematic demo, morphing, feature maps, adversarial lab, 22 tests |
| 4 | 🟡 Yellow | Polish & Delight | Auto-start training, heartbeat, spring animations, slide-ins |
| 5 | 🔴 Red | Feel & Intuition | Confidence glow, warm accents, hover depth, vignette overlays |
| 6 | 🔵 Blue | Process & Summary | 129 structural tests, architecture audit, README/AUDIT docs |
| 7 | 🟢 Green #2 | Creative Features | Network Dreams, Neuron Surgery, Training Race, 51 tests |
| 8 | ⚫ Black #2 | Re-Audit | Memory leak fix, render stability, cleanup verification, 30 tests |
| 9 | 🔴 Red #2 | Final Polish | Ambient gradients, entrance animations, micro-interactions |
| 10 | ⚪ White #2 | Final Verification | Build verification, showcase docs, cleanup, deploy |

> **Test growth**: 0 → 60 → 99 → 121 → 121 → 121 → 250 → 301 → 331 → 331 → 331

See [AUDIT.md](./AUDIT.md) for the complete journey with quantitative metrics and qualitative assessments at each stage.

---

## 📄 License

MIT
