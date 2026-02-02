# NeuralPlayground — White Hat Audit

## Baseline (Pass 1)

| Metric | Value |
|--------|-------|
| Source LOC | ~2,255 |
| Source Files | 14 |
| Components | 7 (App, NetworkVisualizer, ControlPanel, LossChart, DrawingCanvas, PredictionBar, ActivationVisualizer, WeightPanel, WeightHeatmap) |
| Hooks | 1 (useNeuralNetwork) |
| Core Logic | 2 files (NeuralNetwork.ts, sampleData.ts) |
| Tests | 60 (2 test files) |
| TS Errors | 0 |
| Lint Errors | 0 |
| Bundle JS | 218KB (68KB gzip) |
| Bundle CSS | 9KB (2.3KB gzip) |

## Architecture

```
src/
├── nn/
│   ├── NeuralNetwork.ts    (267 LOC) — Core NN: forward, backward, softmax, Xavier init
│   └── sampleData.ts       (135 LOC) — Procedural digit generation, canvasToInput
├── hooks/
│   └── useNeuralNetwork.ts (117 LOC) — React hook: train loop, state management
├── components/
│   ├── NetworkVisualizer.tsx (201 LOC) — Canvas: node/connection graph
│   ├── ControlPanel.tsx     (170 LOC) — Training controls, LR slider, layer config
│   ├── LossChart.tsx        (138 LOC) — Canvas: loss + accuracy chart
│   ├── DrawingCanvas.tsx    (120 LOC) — Canvas: handwritten digit input
│   ├── ActivationVisualizer.tsx (104 LOC) — Canvas: per-layer activation bars
│   ├── WeightHeatmap.tsx    (90 LOC)  — Canvas: weight matrix heatmap
│   ├── WeightPanel.tsx      (42 LOC)  — Layer tab selector for heatmap
│   └── PredictionBar.tsx    (50 LOC)  — 0-9 probability bars
├── App.tsx                  (149 LOC) — Root: layout, state wiring
├── App.css                  (661 LOC) — All styles
├── main.tsx                 (9 LOC)   — Entry point
└── index.css                (1 LOC)   — Minimal global
```

## Features

- **Custom Neural Network** — From-scratch JS implementation, no ML libraries
- **Configurable Architecture** — 1-5 hidden layers, 8-256 neurons each
- **3 Activation Functions** — ReLU, Sigmoid, Tanh (per layer)
- **Xavier Initialization** — Proper weight scaling
- **SGD Training** — Stochastic gradient descent with configurable learning rate
- **Softmax + Cross-Entropy** — Output layer + loss function
- **Procedural Digit Data** — 10 digit patterns with jitter/noise (not MNIST)
- **Drawing Canvas** — Handwritten digit input with live prediction
- **Network Visualizer** — Node graph with weighted connections (color = weight sign)
- **Activation Bars** — Per-layer activation magnitude visualization
- **Weight Heatmap** — Color-coded weight matrix per layer
- **Loss/Accuracy Chart** — Dual-axis training progress
- **Prediction Bar** — 0-9 probability distribution
- **Responsive** — 3-column → 2-column → 1-column layout

## Known Issues

1. **No Error Boundary** — React crash = blank screen
2. **No keyboard shortcuts** — No Space/R/H shortcuts
3. **No ARIA accessibility** — Canvas elements have no labels, no roles, no keyboard nav
4. **No prefers-reduced-motion** — No animation considerations
5. **No touch optimization** — Drawing works via touch events but no touch targets for buttons
6. **Monolithic CSS** — 661 lines in single App.css, no modules/variables separation
7. **No CI/CD** — No automated testing or deployment (added in this pass)
8. **Boilerplate README** — Vite template default
9. **No favicon** — Emoji-only inline SVG (added custom in this pass)
10. **No SEO/OG tags** — Minimal meta (added in this pass)
11. **Hardcoded canvas dimensions** — NetworkVisualizer width=620, height=420 passed as props but not responsive
12. **No WebGL/Canvas error handling** — Canvas 2D context failures silently ignored
13. **Training data not seeded** — Different data every run (Math.random in digit gen)
14. **Output layer uses sigmoid label but actually uses softmax** — Confusing naming in forward()
