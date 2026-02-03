# 🧬 NeuralPlayground

**Watch a neural network learn in real-time.** Draw digits, train networks, visualize every weight and activation, attack with adversarial noise, dream with gradient ascent, ablate neurons, race architectures — all in the browser. No TensorFlow, no ML libraries. Just pure TypeScript and math.

[![Live Demo](https://img.shields.io/badge/🚀_demo-live-brightgreen?style=for-the-badge)](https://kai-claw.github.io/neuralplayground/)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict_0_errors-blue?style=for-the-badge)]()
[![Tests](https://img.shields.io/badge/tests-680_passing-brightgreen?style=for-the-badge)]()
[![Bundle](https://img.shields.io/badge/bundle-103.9KB_gzip-purple?style=for-the-badge)]()
[![License](https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge)]()

<p align="center">
  <strong><a href="https://kai-claw.github.io/neuralplayground/">👉 Try the Live Demo</a></strong>
</p>

---

## ✨ Features

### Core Neural Network Engine
| Feature | Description |
|---------|-------------|
| **Custom NN from Scratch** | Forward/backward pass, SGD, cross-entropy loss, Xavier init — zero ML libraries |
| **Configurable Architecture** | 1–5 hidden layers, 8–256 neurons each, ReLU/Sigmoid/Tanh activations |
| **Real-time Training** | Watch loss decrease and accuracy climb with live dual-axis charts |
| **Procedural Digit Data** | Stroke-based digit generation (no MNIST download needed) |

### Visualization & Interaction
| Feature | Description |
|---------|-------------|
| **Network Topology** | Full graph of neurons + weighted connections with color-coded signal flow particles |
| **Drawing Canvas** | Touch/mouse digit input with live prediction probability bars |
| **Feature Maps** | First-layer 28×28 weight heatmaps — see edge/curve detectors emerge |
| **Weight Heatmaps** | Per-layer weight matrices with diverging colormap + layer tabs |
| **Activation Bars** | Per-layer activation magnitudes showing signal strength |
| **Saliency Maps** | Input-gradient highlighting — which pixels matter most? |
| **Activation Space** | PCA-projected hidden representations — watch digit clusters form |
| **Confusion Matrix** | NxN prediction error grid — where does the network confuse digits? |
| **Gradient Flow Monitor** | Per-layer gradient magnitudes — detect vanishing/exploding gradients |

### Labs & Experiments
| Feature | Description |
|---------|-------------|
| **Adversarial Noise Lab** | Gaussian, salt-pepper, and targeted noise — watch confidence crumble |
| **Digit Morphing** | Blend between two drawn digits, watch the decision boundary in real-time |
| **Network Dreams** | Gradient ascent from random noise — see what the network "imagines" each digit looks like |
| **Neuron Surgery** | Freeze or kill individual neurons, watch the network compensate or break |
| **Training Race** | Pit two architectures against each other — shallow vs. deep, narrow vs. wide |
| **Chimera Lab** | Gradient ascent toward *two* digits simultaneously — hybrid dream creatures |
| **Ablation Lab** | Systematic per-neuron knockout study — identify critical vs. redundant neurons |
| **Weight Evolution** | Filmstrip of weight snapshots across epochs — watch features crystallize |
| **Epoch Replay** | Training time machine — scrub through weight history and replay learning |
| **Decision Boundary** | 2D PCA projection showing how the network carves up input space |
| **Misfit Gallery** | Curated gallery of the network's worst predictions — learn from failure |

### Experience Modes
| Feature | Description |
|---------|-------------|
| **Cinematic Demo** | Auto-draws all 10 digits with stroke animations + full training cycle |
| **Auto-start Training** | Trains 15 epochs on first load for instant visual impact |
| **Adaptive Performance** | Auto-degrades heavy features at low FPS, auto-recovers when stable |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Train / Pause training |
| `R` | Reset network weights |
| `D` | Start cinematic demo |
| `H` | Toggle help overlay |
| `Esc` | Close panel / Stop demo |

---

## 🏗️ Architecture

```
src/
├── nn/                              # Neural network engine (pure TS, 0 React imports)
│   ├── NeuralNetwork.ts        471  # Forward/backward pass, training, prediction, surgery
│   ├── sampleData.ts           116  # Procedural digit generation, canvasToInput
│   ├── dreams.ts                76  # Gradient ascent dream generation
│   ├── saliency.ts             39   # Input-gradient saliency computation
│   ├── noise.ts                 47  # Gaussian, salt-pepper, targeted noise
│   ├── ablation.ts              74  # Per-neuron knockout study
│   ├── chimera.ts               54  # Multi-digit hybrid dream generation
│   ├── confusion.ts             22  # Confusion matrix computation
│   ├── decisionBoundary.ts    265   # PCA-based decision boundary mapping
│   ├── epochReplay.ts          66   # Weight snapshot recording for replay
│   ├── gradientFlow.ts        213   # Per-layer gradient magnitude tracking
│   ├── misfits.ts               46  # Worst-prediction finder
│   ├── pca.ts                  107  # Principal component analysis
│   ├── weightEvolution.ts       87  # Epoch-by-epoch weight filmstrip recorder
│   └── index.ts                 18  # Barrel export
│
├── hooks/                           # React state management (6 hooks)
│   ├── useNeuralNetwork.ts     176  # Training loop, snapshot state, dream/saliency
│   ├── useCinematic.ts         202  # Demo mode state machine
│   ├── useActivationSpace.ts   138  # PCA-projected activation cloud
│   ├── useTrainingRace.ts      136  # Side-by-side architecture race controller
│   ├── usePerformanceMonitor.ts 78  # FPS tracking + adaptive degradation
│   ├── useContainerDims.ts      43  # Responsive ResizeObserver
│   └── index.ts                  7  # Barrel export
│
├── components/                      # UI layer (29 components, Canvas-based rendering)
│   ├── NetworkVisualizer.tsx   363  # Topology graph + signal flow particles
│   ├── DrawingCanvas.tsx       224  # Touch/mouse digit drawing (forwardRef)
│   ├── LossChart.tsx           192  # Dual-axis loss/accuracy chart
│   ├── ActivationVisualizer.tsx 91  # Per-layer activation bar chart
│   ├── PredictionBar.tsx        53  # 0–9 probability distribution bars
│   ├── ControlPanel.tsx        145  # Training controls + architecture config
│   ├── WeightPanel.tsx         105  # Weight heatmap with layer tabs
│   ├── WeightHeatmap.tsx        95  # Color-coded weight matrix renderer
│   ├── FeatureMaps.tsx         240  # First-layer weight tile grid + magnifier
│   ├── AdversarialLab.tsx      246  # Noise attack lab
│   ├── DigitMorph.tsx          196  # Digit blending with slider
│   ├── NetworkDreams.tsx       237  # Gradient ascent dream viewer
│   ├── NeuronSurgery.tsx       163  # Freeze/kill neuron interface
│   ├── TrainingRace.tsx        173  # Side-by-side architecture race
│   ├── SaliencyMap.tsx         161  # Input-gradient heatmap
│   ├── ActivationSpace.tsx     285  # PCA activation cloud scatter
│   ├── ConfusionMatrix.tsx     165  # NxN prediction error grid
│   ├── GradientFlowMonitor.tsx 158  # Per-layer gradient health bars
│   ├── EpochReplay.tsx         403  # Training time machine + weight scrubber
│   ├── DecisionBoundary.tsx    316  # 2D input space partition map
│   ├── ChimeraLab.tsx          264  # Multi-digit hybrid dream lab
│   ├── MisfitGallery.tsx       296  # Worst-prediction gallery
│   ├── WeightEvolution.tsx     315  # Epoch filmstrip + playback
│   ├── AblationLab.tsx         251  # Per-neuron knockout importance map
│   ├── StatsPanel.tsx           45  # Epoch/loss/accuracy stats
│   ├── HelpOverlay.tsx          49  # Keyboard shortcuts dialog
│   ├── ExperiencePanel.tsx      33  # Cinematic toggle
│   ├── CinematicBadge.tsx       28  # Demo mode status badge
│   ├── ErrorBoundary.tsx        45  # Crash recovery with retry
│   └── index.ts                 30  # Barrel export
│
├── renderers/                       # Canvas rendering utilities (pure functions)
│   ├── pixelRendering.ts       106  # Pixel/weight ImageData generation
│   ├── confusionRenderer.ts    133  # Confusion matrix canvas painting
│   ├── dreamRenderer.ts        165  # Dream/chimera canvas rendering
│   ├── gradientFlowRenderer.ts 228  # Gradient health bar chart rendering
│   ├── raceChart.ts            163  # Race comparison chart rendering
│   ├── surgeryRenderer.ts      224  # Surgery topology canvas rendering
│   └── index.ts                  8  # Barrel export
│
├── visualizers/                     # Layout computation (pure math)
│   ├── networkLayout.ts         65  # Network topology positioning
│   └── index.ts                  2  # Barrel export
│
├── utils/                           # Shared utilities
│   ├── math.ts                  75  # Softmax, argmax, RNG, helpers
│   ├── activations.ts           24  # ReLU, sigmoid, tanh + derivatives
│   ├── colors.ts                15  # Diverging colormap helpers
│   ├── prng.ts                  32  # Seeded mulberry32 PRNG
│   └── index.ts                  6  # Barrel export
│
├── data/                            # Static data
│   ├── digitStrokes.ts         147  # Cinematic auto-draw stroke paths
│   └── racePresets.ts           80  # Architecture presets for Training Race
│
├── App.tsx                     442  # Root orchestrator (hooks → components)
├── App.css                    4486  # All styles (animations, responsive, reduced-motion)
├── constants.ts                188  # Centralized magic numbers + configs
├── types.ts                    127  # Shared TypeScript interfaces
├── main.tsx                      8  # Entry point
└── index.css                     1  # CSS reset
```

**Data flow** (unidirectional): `nn/` → `hooks/` → `App.tsx` → `components/`

**Module discipline**:
- `nn/` and `renderers/` have **zero React imports** — pure computation
- `utils/` files have **no internal cross-dependencies**
- Barrel exports at every directory level for clean imports

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | React 19 + TypeScript 5.9 (strict mode) |
| **Rendering** | HTML5 Canvas (zero DOM-per-neuron) |
| **NN Engine** | Custom from-scratch (forward/backward, SGD, cross-entropy, Xavier init) |
| **Build** | Vite 7 |
| **Testing** | Vitest 4 (680 tests across 14 suites) |
| **Deployment** | GitHub Pages via gh-pages + GitHub Actions CI/CD |
| **Dependencies** | React + React DOM only. **No ML libraries.** |

---

## 🧪 ML Concepts Demonstrated

| Concept | Where |
|---------|-------|
| **Backpropagation** | Custom gradient computation through every layer |
| **Xavier Initialization** | Weight scaling proportional to layer fan-in |
| **Cross-Entropy Loss** | Softmax output with log-likelihood loss |
| **Activation Functions** | ReLU, Sigmoid, Tanh — selectable per layer, with derivative computation |
| **Gradient Ascent** | Network Dreams + Chimera Lab — maximize class probability from noise |
| **Saliency Maps** | Input-gradient highlighting of salient pixels |
| **PCA Projection** | 2D visualization of high-dimensional activation space |
| **Adversarial Examples** | Targeted noise pushing predictions toward a chosen class |
| **Ablation Studies** | Systematic neuron knockout measuring per-neuron importance |
| **Decision Boundaries** | PCA-projected input space partition visualization |
| **Confusion Matrices** | NxN classification error analysis |
| **Feature Visualization** | First-layer weight grids revealing learned edge/curve detectors |

---

## 📊 Metrics

| Metric | Value |
|--------|-------|
| Source files | 75 |
| Source LOC | ~11,300 |
| Test LOC | ~7,600 |
| CSS LOC | ~4,500 |
| Tests | 680 (14 suites) |
| TypeScript errors | 0 |
| `as any` casts | 0 |
| Build warnings | 0 |
| Bundle JS | 335 KB (103.9 KB gzip) |
| Bundle CSS | 69 KB (11.9 KB gzip) |
| Components | 29 |
| Hooks | 6 |
| NN modules | 15 |
| Renderers | 7 |
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

# Run all 680 tests
npm test

# Production build
npm run build

# Deploy to GitHub Pages
npm run deploy
```

---

## ♿ Accessibility

- Full **ARIA labels** on all 29 interactive components
- **Keyboard navigation** — all features accessible without mouse (Space/R/D/H/Esc)
- **`prefers-reduced-motion`** — all animations respect system settings
- **Skip links** and semantic HTML structure
- **Error boundary** with retry and reload options
- **Screen reader announcements** for training state changes
- **Focus-visible** outlines with glow on all interactive elements
- **Noscript fallback** for JS-disabled browsers

---

## 🎩 Development Process — Six Thinking Hats (10 Passes)

This project was built through 10 structured iteration passes using Edward de Bono's **Six Thinking Hats** methodology:

| Pass | Hat | Focus | Key Deliverables |
|------|-----|-------|-----------------|
| 1 | ⚪ White | Facts & Audit | CI/CD, SEO, PWA manifest, 60 baseline tests |
| 2 | ⚫ Black | Risks & Problems | NaN guards, error boundary, ARIA, mobile responsive, 39 edge-case tests |
| 3 | 🟢 Green | Creative Features | Feature maps, adversarial lab, signal flow, cinematic demo, digit morph |
| 4 | 🟡 Yellow | Polish & Delight | Auto-start training, heartbeat indicator, spring animations, slide-ins |
| 5 | 🔴 Red | Feel & Intuition | Vignettes, confidence glow, warm accents, hover depth, stat color-coding |
| 6 | 🔵 Blue | Architecture | Module extraction (3 components + 3 module renames), barrel exports, 48 arch tests |
| 7 | 🟢 Green #2 | Creative Features | Weight evolution filmstrip, ablation lab, +46 tests |
| 8 | ⚫ Black #2 | Stress Test | Integer-keyed masks, snapshot caching, scratch buffers, saliency LUT, adaptive perf, +29 tests |
| 9 | 🟡 Yellow #2 | Final Polish | PWA raster icons, enhanced JSON-LD, 73 portfolio-readiness tests, instructions bar, sitemap |
| 10 | ⚪ White #2 | Final Verification | *Coming next* |

> **Test growth**: 0 → 60 → 99 → 121 → 121 → 121 → 472 → 578 → 607 → 680

---

## 📄 License

MIT — see [LICENSE](./LICENSE)
