import { useRef, useEffect, useState, useCallback } from 'react';

interface AdversarialLabProps {
  /** Current drawing as 784-element pixel array (28×28, values 0-1) */
  currentInput: number[] | null;
  /** Called with the noised input for prediction */
  onPredict: (input: number[]) => void;
  /** Current prediction probabilities (10 classes) */
  probabilities: number[] | null;
  /** Predicted label */
  predictedLabel: number | null;
}

const DISPLAY_SIZE = 160;
const INPUT_DIM = 28;

/** Seeded PRNG for reproducible noise patterns */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller transform for gaussian noise */
function gaussianNoise(rng: () => number): number {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Pre-allocated noise pattern (regenerated on seed change)
const NOISE_PATTERN = new Float32Array(INPUT_DIM * INPUT_DIM);

/** Noise type options */
type NoiseType = 'gaussian' | 'salt-pepper' | 'adversarial';

const NOISE_LABELS: Record<NoiseType, string> = {
  gaussian: '🌊 Gaussian',
  'salt-pepper': '🧂 Salt & Pepper',
  adversarial: '🎯 Targeted',
};

const NOISE_DESCRIPTIONS: Record<NoiseType, string> = {
  gaussian: 'Random bell-curve noise — like TV static',
  'salt-pepper': 'Random black & white pixel flips',
  adversarial: 'Push the prediction toward a target digit',
};

/**
 * Adversarial Noise Lab — gradually corrupt a drawing and watch the
 * neural network's confidence crumble.
 *
 * Three noise modes:
 * - Gaussian: smooth random noise
 * - Salt & Pepper: random pixel flips
 * - Targeted: attempt to push prediction toward a chosen digit
 */
export function AdversarialLab({
  currentInput,
  onPredict,
  probabilities,
  predictedLabel,
}: AdversarialLabProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [noiseLevel, setNoiseLevel] = useState(0);
  const [noiseType, setNoiseType] = useState<NoiseType>('gaussian');
  const [noiseSeed, setNoiseSeed] = useState(42);
  const [targetDigit, setTargetDigit] = useState(3);
  const [originalLabel, setOriginalLabel] = useState<number | null>(null);

  // Track original prediction when input arrives
  useEffect(() => {
    if (predictedLabel !== null && noiseLevel === 0) {
      setOriginalLabel(predictedLabel);
    }
  }, [predictedLabel, noiseLevel]);

  // Generate noise pattern and render noised image
  useEffect(() => {
    if (!currentInput || currentInput.length < INPUT_DIM * INPUT_DIM) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = DISPLAY_SIZE * dpr;
    canvas.height = DISPLAY_SIZE * dpr;
    ctx.scale(dpr, dpr);

    // Generate reproducible noise
    const rng = mulberry32(noiseSeed);
    const len = INPUT_DIM * INPUT_DIM;

    if (noiseType === 'gaussian') {
      for (let i = 0; i < len; i++) {
        NOISE_PATTERN[i] = gaussianNoise(rng);
      }
    } else if (noiseType === 'salt-pepper') {
      for (let i = 0; i < len; i++) {
        const r = rng();
        // Each pixel has a chance of being flipped
        if (r < 0.15) NOISE_PATTERN[i] = 1;       // salt (white)
        else if (r < 0.30) NOISE_PATTERN[i] = -1;  // pepper (black)
        else NOISE_PATTERN[i] = 0;
      }
    } else {
      // "Adversarial" — gradient-like perturbation toward target
      // Simple heuristic: push pixels toward a pattern that activates the target
      for (let i = 0; i < len; i++) {
        // Seeded semi-structured noise with spatial coherence
        const x = i % INPUT_DIM;
        const y = Math.floor(i / INPUT_DIM);
        const cx = INPUT_DIM / 2;
        const cy = INPUT_DIM / 2;
        const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (INPUT_DIM / 2);
        // Target-dependent angular bias
        const angle = Math.atan2(y - cy, x - cx);
        const targetAngle = (targetDigit / 10) * Math.PI * 2;
        const angleBias = Math.cos(angle - targetAngle);
        NOISE_PATTERN[i] = (angleBias * (1 - dist) + gaussianNoise(rng) * 0.3);
      }
    }

    // Apply noise to input
    const noised = new Array<number>(len);
    for (let i = 0; i < len; i++) {
      if (noiseType === 'salt-pepper') {
        // Salt-pepper: at noise level, flip or not
        const flip = Math.abs(NOISE_PATTERN[i]) > 0.5;
        if (flip && rng() < noiseLevel) {
          noised[i] = NOISE_PATTERN[i] > 0 ? 1 : 0;
        } else {
          noised[i] = currentInput[i];
        }
      } else {
        noised[i] = Math.max(0, Math.min(1, currentInput[i] + NOISE_PATTERN[i] * noiseLevel));
      }
    }

    // Render
    const scale = DISPLAY_SIZE / INPUT_DIM;
    for (let y = 0; y < INPUT_DIM; y++) {
      for (let x = 0; x < INPUT_DIM; x++) {
        const v = Math.round(noised[y * INPUT_DIM + x] * 255);
        ctx.fillStyle = `rgb(${v}, ${v}, ${v})`;
        ctx.fillRect(x * scale, y * scale, scale + 0.5, scale + 0.5);
      }
    }

    // Noise level overlay
    if (noiseLevel > 0) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(0, DISPLAY_SIZE - 20, DISPLAY_SIZE, 20);
      ctx.fillStyle = noiseLevel < 0.3 ? '#10b981' : noiseLevel < 0.6 ? '#fbbf24' : '#ff6384';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Noise: ${(noiseLevel * 100).toFixed(0)}%`, DISPLAY_SIZE / 2, DISPLAY_SIZE - 6);
    }

    // Fire prediction with noised input
    onPredict(noised);
  }, [currentInput, noiseLevel, noiseType, noiseSeed, targetDigit, onPredict]);

  const handleNoiseSlider = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setNoiseLevel(parseFloat(e.target.value));
  }, []);

  const handleNewSeed = useCallback(() => {
    setNoiseSeed(prev => prev + 1);
  }, []);

  // Compute confidence delta
  const confidenceDrop = probabilities && originalLabel !== null
    ? (1 - (probabilities[originalLabel] || 0)) * 100
    : 0;

  const flipped = predictedLabel !== null && originalLabel !== null && predictedLabel !== originalLabel;

  if (!currentInput) {
    return (
      <div className="adversarial-lab" role="group" aria-label="Adversarial noise lab">
        <div className="panel-header">
          <span className="panel-icon" aria-hidden="true">🎭</span>
          <span>Adversarial Lab</span>
        </div>
        <div className="adversarial-empty">
          <p>Draw a digit first, then add noise to test the network's robustness.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="adversarial-lab" role="group" aria-label="Adversarial noise lab">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🎭</span>
        <span>Adversarial Lab</span>
      </div>
      <div className="adversarial-content">
        {/* Noised image preview */}
        <div className="adversarial-preview">
          <canvas
            ref={canvasRef}
            style={{ width: DISPLAY_SIZE, height: DISPLAY_SIZE }}
            className="adversarial-canvas"
            role="img"
            aria-label={`Drawing with ${(noiseLevel * 100).toFixed(0)}% ${noiseType} noise applied`}
          />
          {flipped && (
            <div className="adversarial-flip-badge">
              ⚡ FOOLED! {originalLabel} → {predictedLabel}
            </div>
          )}
        </div>

        {/* Noise type selector */}
        <div className="adversarial-type-selector" role="radiogroup" aria-label="Noise type">
          {(Object.keys(NOISE_LABELS) as NoiseType[]).map(type => (
            <button
              key={type}
              className={`adversarial-type-btn ${noiseType === type ? 'active' : ''}`}
              onClick={() => setNoiseType(type)}
              role="radio"
              aria-checked={noiseType === type}
              aria-label={NOISE_DESCRIPTIONS[type]}
            >
              {NOISE_LABELS[type]}
            </button>
          ))}
        </div>

        {/* Target digit (for adversarial mode) */}
        {noiseType === 'adversarial' && (
          <div className="adversarial-target">
            <label className="adversarial-target-label" htmlFor="target-digit">
              Target digit
            </label>
            <div className="adversarial-target-digits" role="radiogroup" aria-label="Target digit">
              {Array.from({ length: 10 }, (_, i) => (
                <button
                  key={i}
                  className={`target-digit-btn ${targetDigit === i ? 'active' : ''}`}
                  onClick={() => setTargetDigit(i)}
                  role="radio"
                  aria-checked={targetDigit === i}
                >
                  {i}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Noise level slider */}
        <div className="adversarial-slider-section">
          <label className="control-label" htmlFor="noise-slider">
            Noise Level
            <span className={`control-value ${noiseLevel > 0.6 ? 'danger' : noiseLevel > 0.3 ? 'warning' : ''}`}>
              {(noiseLevel * 100).toFixed(0)}%
            </span>
          </label>
          <input
            id="noise-slider"
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={noiseLevel}
            onChange={handleNoiseSlider}
            className="slider adversarial-slider"
            aria-valuemin={0}
            aria-valuemax={1}
            aria-valuenow={noiseLevel}
            aria-label="Noise level"
          />
          <div className="adversarial-slider-labels">
            <span>Clean</span>
            <span>Destroyed</span>
          </div>
        </div>

        <button className="btn btn-secondary adversarial-reseed" onClick={handleNewSeed}>
          🎲 New Noise Pattern
        </button>

        {/* Confidence meter */}
        {probabilities && originalLabel !== null && (
          <div className="adversarial-confidence">
            <div className="confidence-header">
              <span className="confidence-label">Original confidence</span>
              <span className={`confidence-value ${flipped ? 'flipped' : ''}`}>
                {((probabilities[originalLabel] || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="confidence-track">
              <div
                className={`confidence-fill ${flipped ? 'flipped' : ''}`}
                style={{ width: `${(probabilities[originalLabel] || 0) * 100}%` }}
              />
            </div>
            {confidenceDrop > 0 && (
              <div className="confidence-drop">
                ↓ {confidenceDrop.toFixed(1)}% confidence drop
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default AdversarialLab;
