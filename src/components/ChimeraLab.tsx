/**
 * ChimeraLab — Blend digit classes to create hybrid images.
 *
 * "What does 50% '3' + 50% '8' look like to a neural network?"
 *
 * Users adjust sliders for each digit class (0-9), and the network
 * generates a chimera image via multi-class gradient ascent. Preset
 * buttons offer interesting combinations. Confidence bars show how
 * the network interprets each chimera.
 */

import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import type { NeuralNetwork } from '../nn';
import { dreamChimera, CHIMERA_PRESETS } from '../nn';
import type { ChimeraResult } from '../nn';
import {
  CHIMERA_DISPLAY_SIZE,
  CHIMERA_STEPS,
  CHIMERA_LR,
  CHIMERA_ANIMATION_INTERVAL,
  INPUT_DIM,
} from '../constants';

/** Digit class colors — consistent across app */
const DIGIT_COLORS = [
  '#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff',
  '#ff9f40', '#10b981', '#f472b6', '#63deff', '#a78bfa',
] as const;

interface ChimeraLabProps {
  networkRef: React.MutableRefObject<NeuralNetwork | null>;
  hasTrained: boolean;
}

export default function ChimeraLab({ networkRef, hasTrained }: ChimeraLabProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [weights, setWeights] = useState<number[]>(() => new Array(10).fill(0));
  const [result, setResult] = useState<ChimeraResult | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [animStep, setAnimStep] = useState(0);
  const animRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const historyRef = useRef<ChimeraResult | null>(null);

  // Active weight count for display
  const activeCount = useMemo(() => weights.filter((w) => w > 0).length, [weights]);
  const totalWeight = useMemo(() => weights.reduce((a, b) => a + b, 0), [weights]);

  const updateWeight = useCallback((digit: number, value: number) => {
    setWeights((prev) => {
      const next = [...prev];
      next[digit] = value;
      return next;
    });
  }, []);

  const applyPreset = useCallback((presetWeights: number[]) => {
    setWeights([...presetWeights]);
  }, []);

  const generate = useCallback(() => {
    const net = networkRef.current;
    if (!net || totalWeight === 0) return;

    setIsGenerating(true);
    setAnimStep(0);

    // Clear previous animation
    if (animRef.current) clearInterval(animRef.current);

    // Run chimera generation (synchronous, but we animate display after)
    const chimerResult = dreamChimera(net, weights, CHIMERA_STEPS, CHIMERA_LR);
    historyRef.current = chimerResult;

    // Animate the generation step-by-step
    let step = 0;
    animRef.current = setInterval(() => {
      step += 2; // skip every other step for speed
      if (step >= chimerResult.confidenceHistory.length) {
        step = chimerResult.confidenceHistory.length - 1;
        if (animRef.current) {
          clearInterval(animRef.current);
          animRef.current = null;
        }
        setIsGenerating(false);
      }
      setAnimStep(step);
      setResult(chimerResult);
    }, CHIMERA_ANIMATION_INTERVAL);
  }, [networkRef, weights, totalWeight]);

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      if (animRef.current) clearInterval(animRef.current);
    };
  }, []);

  // Render chimera image to canvas
  useEffect(() => {
    if (!result) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const displaySize = CHIMERA_DISPLAY_SIZE;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = displaySize * dpr;
    canvas.height = displaySize * dpr;
    ctx.scale(dpr, dpr);

    const { image } = result;
    const cellSize = displaySize / INPUT_DIM;

    for (let y = 0; y < INPUT_DIM; y++) {
      for (let x = 0; x < INPUT_DIM; x++) {
        const v = Math.round(image[y * INPUT_DIM + x] * 255);
        ctx.fillStyle = `rgb(${v}, ${v}, ${v})`;
        ctx.fillRect(x * cellSize, y * cellSize, cellSize + 0.5, cellSize + 0.5);
      }
    }
  }, [result, animStep]);

  // Current confidence snapshot for bars
  const currentConf = useMemo(() => {
    if (!result || !result.confidenceHistory.length) return null;
    const idx = Math.min(animStep, result.confidenceHistory.length - 1);
    return result.confidenceHistory[idx];
  }, [result, animStep]);

  return (
    <div
      className="chimera-lab"
      role="group"
      aria-label="Digit Chimera Lab — blend digit classes to create hybrid images"
    >
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🧬</span>
        <span>Chimera Lab</span>
        {activeCount > 0 && (
          <span className="chimera-badge">{activeCount} classes</span>
        )}
      </div>

      {!hasTrained ? (
        <p className="chimera-hint">Train the network first to create chimeras</p>
      ) : (
        <>
          {/* Presets */}
          <div className="chimera-presets" role="group" aria-label="Chimera presets">
            {CHIMERA_PRESETS.map((preset) => (
              <button
                key={preset.name}
                className="chimera-preset-btn"
                onClick={() => applyPreset(preset.weights)}
                title={preset.description}
                aria-label={`${preset.name} — ${preset.description}`}
              >
                <span className="preset-emoji" aria-hidden="true">
                  {preset.emoji}
                </span>
                <span className="preset-name">{preset.name}</span>
              </button>
            ))}
          </div>

          {/* Weight sliders */}
          <div className="chimera-sliders" role="group" aria-label="Digit class weights">
            {weights.map((w, i) => (
              <div key={i} className="chimera-slider-row">
                <span
                  className="chimera-digit-label"
                  style={{ color: DIGIT_COLORS[i] }}
                >
                  {i}
                </span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={w}
                  onChange={(e) => updateWeight(i, Number(e.target.value))}
                  className="chimera-slider"
                  aria-label={`Digit ${i} weight`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={w}
                  style={{ '--slider-color': DIGIT_COLORS[i] } as React.CSSProperties}
                />
                <span className="chimera-weight-val" style={{ opacity: w > 0 ? 1 : 0.3 }}>
                  {totalWeight > 0 ? Math.round((w / totalWeight) * 100) : 0}%
                </span>
              </div>
            ))}
          </div>

          {/* Generate button */}
          <button
            className={`chimera-generate-btn ${isGenerating ? 'generating' : ''}`}
            onClick={generate}
            disabled={isGenerating || totalWeight === 0}
            aria-label="Generate chimera digit"
          >
            {isGenerating ? '🧬 Generating…' : '🧬 Generate Chimera'}
          </button>

          {/* Result display */}
          {result && (
            <div className="chimera-result">
              <div className="chimera-canvas-wrap">
                <canvas
                  ref={canvasRef}
                  style={{ width: CHIMERA_DISPLAY_SIZE, height: CHIMERA_DISPLAY_SIZE }}
                  className="chimera-canvas"
                  role="img"
                  aria-label="Generated chimera digit image"
                />
                {isGenerating && (
                  <div className="chimera-progress" aria-hidden="true">
                    Step {animStep}/{CHIMERA_STEPS}
                  </div>
                )}
              </div>

              {/* Mini confidence bars */}
              {currentConf && (
                <div className="chimera-confidence" role="list" aria-label="Chimera class confidence">
                  {currentConf.map((conf, i) => (
                    <div
                      key={i}
                      className="chimera-conf-row"
                      role="listitem"
                      aria-label={`Digit ${i}: ${Math.round(conf * 100)}%`}
                    >
                      <span
                        className="chimera-conf-digit"
                        style={{ color: DIGIT_COLORS[i] }}
                      >
                        {i}
                      </span>
                      <div className="chimera-conf-bar-bg">
                        <div
                          className="chimera-conf-bar-fill"
                          style={{
                            width: `${Math.round(conf * 100)}%`,
                            backgroundColor: DIGIT_COLORS[i],
                          }}
                        />
                      </div>
                      <span className="chimera-conf-pct">
                        {Math.round(conf * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
