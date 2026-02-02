# NeuralPlayground — Architecture

## Directory Structure

```
src/
├── App.tsx                     (326 LOC)  Root component — layout, keyboard, state wiring
├── App.css                                Global styles
├── main.tsx                    (12)       React entry point
├── types.ts                    (66)       CANONICAL type definitions (single source of truth)
├── constants.ts                (122)      All magic numbers, timing, display, defaults
├── noise.ts                    (8)        Compat re-export → nn/noise.ts
├── visualizer.ts               (14)       Compat re-export → visualizers/networkLayout.ts
├── rendering.ts                (12)       Compat re-export → renderers/pixelRendering.ts
│
├── nn/                                    Neural network core (no React dependency)
│   ├── index.ts                (22)       Barrel export
│   ├── NeuralNetwork.ts        (324)      Network class: init, forward, backward, surgery
│   ├── dreams.ts               (112)      Gradient ascent / network dream visualization
│   ├── noise.ts                (105)      Adversarial noise generation (gaussian, salt-pepper, targeted)
│   └── sampleData.ts           (135)      Training data generation + canvas→input conversion
│
├── utils/                                 Pure utility functions (no side effects)
│   ├── index.ts                (11)       Barrel export
│   ├── activations.ts          (29)       ReLU, sigmoid, tanh + derivatives
│   ├── math.ts                 (43)       safeMax, argmax, softmax, xavierInit
│   ├── prng.ts                 (23)       Seeded PRNG (mulberry32) + gaussian noise
│   └── colors.ts               (26)       Activation/weight → CSS color mapping
│
├── visualizers/                           Pure computation for visual layouts
│   ├── index.ts                (14)       Barrel export
│   └── networkLayout.ts        (169)      Node positions, signal particles, layer sizes
│
├── renderers/                             Pure canvas rendering (no React)
│   ├── index.ts                (35)       Barrel export
│   ├── pixelRendering.ts       (122)      Weight colormaps, pixel ImageData, lerp
│   ├── dreamRenderer.ts        (136)      Dream image rendering + gallery
│   ├── raceChart.ts            (156)      Training race bar chart
│   └── surgeryRenderer.ts      (224)      Neuron surgery interactive canvas
│
├── data/                                  Static/generated data (no React)
│   ├── digitStrokes.ts         (102)      Cinematic drawing stroke paths
│   └── racePresets.ts          (81)       Training race configuration presets
│
├── hooks/                                 Custom React hooks
│   ├── index.ts                (11)       Barrel export
│   ├── useNeuralNetwork.ts     (158)      Network training lifecycle + state
│   ├── useCinematic.ts         (202)      Cinematic demo orchestration
│   ├── useContainerDims.ts     (66)       ResizeObserver-based sizing
│   └── useTrainingRace.ts      (178)      Head-to-head training race logic
│
├── components/                            React UI components
│   ├── index.ts                (26)       Barrel export
│   ├── NetworkVisualizer.tsx   (347)      3D-style network graph + signal flow
│   ├── AdversarialLab.tsx      (246)      Noise perturbation + confidence tracking
│   ├── NetworkDreams.tsx       (237)      Gradient ascent dream gallery
│   ├── FeatureMaps.tsx         (222)      First-layer weight visualization
│   ├── DrawingCanvas.tsx       (188)      Digit drawing input
│   ├── ControlPanel.tsx        (183)      Training config + architecture editor
│   ├── NeuronSurgery.tsx       (173)      Interactive neuron freeze/kill
│   ├── LossChart.tsx           (165)      Loss + accuracy dual chart
│   ├── TrainingRace.tsx        (156)      Head-to-head config comparison
│   ├── DigitMorph.tsx          (131)      Pixel interpolation between digits
│   ├── ActivationVisualizer.tsx(117)      Per-layer activation heatmaps
│   ├── ErrorBoundary.tsx       (96)       Crash recovery wrapper
│   ├── StatsPanel.tsx          (88)       Epoch/loss/accuracy/prediction display
│   ├── WeightHeatmap.tsx       (90)       Weight matrix visualization
│   ├── PredictionBar.tsx       (54)       10-class probability bars
│   ├── WeightPanel.tsx         (48)       Tab container for weight views
│   ├── HelpOverlay.tsx         (40)       Keyboard shortcuts dialog
│   ├── ExperiencePanel.tsx     (36)       Cinematic demo toggle
│   └── CinematicBadge.tsx      (30)       Floating demo status indicator
│
└── __tests__/                             Test suites (472 tests)
    ├── neuralNetwork.test.ts   (502)      Core NN: construction, training, predict
    ├── sampleData.test.ts      (146)      Data generation + canvas conversion
    ├── blackhat.test.ts        (478)      Edge cases, stability, type system
    ├── blackhat2.test.ts       (481)      Surgery, dreams, race presets, RNG
    ├── greenhat.test.ts        (384)      Feature maps, noise, adversarial
    ├── bluehat.test.ts        (1462)      Architecture, module validation
    ├── bluehat2.test.ts        (539)      Visualization, rendering, dreams
    ├── bluehat3.test.ts        (654)      Constants, race, cinematic, noise
    └── bluehat4.test.ts        (new)      Barrel exports, boundaries, integration
```

## Design Decisions

### Module Organization
- **nn/** contains the pure neural network engine — no React, no DOM.
  Noise generation lives here because it's domain-specific (adversarial perturbation).
- **utils/** contains stateless helper functions with no internal cross-dependencies.
  Each file is independently importable.
- **visualizers/** and **renderers/** are pure computation/canvas modules.
  They depend only on `types.ts` and `constants.ts` — never on React or the DOM.
- **hooks/** orchestrate React state and effects.
  They depend on nn/ for computation and constants for configuration.
- **components/** are React UI components.
  They depend on hooks/, renderers/, and constants/ but never on nn/ directly (except via hooks).

### Barrel Exports
Every directory has an `index.ts` barrel for clean imports:
```ts
import { NeuralNetwork, dream } from './nn';
import { computeNodePositions } from './visualizers';
import { weightsToImageData } from './renderers';
```

### Backward Compatibility
Three root-level re-export files (`noise.ts`, `visualizer.ts`, `rendering.ts`)
maintain backward compatibility with any code importing from the old flat paths.
These are thin (< 15 lines) re-exports — no logic.

### Type Ownership
All types live in `src/types.ts` — the single source of truth.
No circular re-exports. Every module imports types from there.

### Component Extraction (Pass 6)
App.tsx was reduced from 377 → 326 lines by extracting:
- **StatsPanel** — epoch/loss/accuracy/prediction display with color thresholds
- **HelpOverlay** — keyboard shortcuts dialog with backdrop close
- **ExperiencePanel** — cinematic demo toggle button

### Neuron Surgery Architecture
Surgery operates through a mask system on NeuralNetwork:
- `setNeuronStatus(layer, neuron, status)` marks neurons as active/frozen/killed
- Forward pass zeros killed neurons; backward pass skips frozen/killed gradients
- Surgery + dreams combine safely (gradient ascent respects masks)
