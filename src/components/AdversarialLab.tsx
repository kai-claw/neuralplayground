import { useRef, useEffect, useState, useCallback } from 'react';
import { generateNoisePattern, applyNoise } from '../noise';
import { pixelsToImageData } from '../rendering';
import type { NoiseType } from '../types';
import {
  ADVERSARIAL_DISPLAY_SIZE,
  ADVERSARIAL_DEFAULT_SEED,
  ADVERSARIAL_DEFAULT_TARGET,
  INPUT_DIM,
  NOISE_LABELS,
  NOISE_DESCRIPTIONS,
} from '../constants';

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
  const [noiseSeed, setNoiseSeed] = useState(ADVERSARIAL_DEFAULT_SEED);
  const [targetDigit, setTargetDigit] = useState(ADVERSARIAL_DEFAULT_TARGET);
  const [originalLabel, setOriginalLabel] = useState<number | null>(null);

  // Generate noise pattern and render noised image
  useEffect(() => {
    if (!currentInput || currentInput.length < INPUT_DIM * INPUT_DIM) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = ADVERSARIAL_DISPLAY_SIZE * dpr;
    canvas.height = ADVERSARIAL_DISPLAY_SIZE * dpr;
    ctx.scale(dpr, dpr);

    // Generate reproducible noise pattern and apply at current level
    const pattern = generateNoisePattern(noiseType, noiseSeed, targetDigit);
    const noised = applyNoise(currentInput, pattern, noiseLevel, noiseType, noiseSeed);

    // Render noised image via shared rendering helper
    const imgData = pixelsToImageData(noised, ADVERSARIAL_DISPLAY_SIZE);
    // putImageData ignores canvas scale, so use an offscreen canvas
    const offscreen = new OffscreenCanvas(ADVERSARIAL_DISPLAY_SIZE, ADVERSARIAL_DISPLAY_SIZE);
    const offCtx = offscreen.getContext('2d');
    if (offCtx) {
      offCtx.putImageData(imgData, 0, 0);
      ctx.drawImage(offscreen, 0, 0, ADVERSARIAL_DISPLAY_SIZE, ADVERSARIAL_DISPLAY_SIZE);
    }

    // Noise level overlay
    if (noiseLevel > 0) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(0, ADVERSARIAL_DISPLAY_SIZE - 20, ADVERSARIAL_DISPLAY_SIZE, 20);
      ctx.fillStyle = noiseLevel < 0.3 ? '#10b981' : noiseLevel < 0.6 ? '#fbbf24' : '#ff6384';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Noise: ${(noiseLevel * 100).toFixed(0)}%`, ADVERSARIAL_DISPLAY_SIZE / 2, ADVERSARIAL_DISPLAY_SIZE - 6);
    }

    // Fire prediction with noised input
    onPredict(noised);
  }, [currentInput, noiseLevel, noiseType, noiseSeed, targetDigit, onPredict]);

  const handleNoiseSlider = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newLevel = parseFloat(e.target.value);
    // Snapshot original label when noise is first applied
    if (noiseLevel === 0 && newLevel > 0 && predictedLabel !== null) {
      setOriginalLabel(predictedLabel);
    }
    // Reset original label when noise returns to zero
    if (newLevel === 0 && predictedLabel !== null) {
      setOriginalLabel(predictedLabel);
    }
    setNoiseLevel(newLevel);
  }, [noiseLevel, predictedLabel]);

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
            style={{ width: ADVERSARIAL_DISPLAY_SIZE, height: ADVERSARIAL_DISPLAY_SIZE }}
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
