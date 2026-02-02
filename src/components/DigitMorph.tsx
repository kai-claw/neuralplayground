import { useRef, useEffect, useCallback, useState } from 'react';

interface DigitMorphProps {
  slotA: number[] | null; // 784-element pixel array (28×28)
  slotB: number[] | null;
  onMorphPredict: (input: number[]) => void;
  onSaveSlot: (slot: 'A' | 'B') => void;
}

/**
 * Digit Morphing — blend between two saved digit drawings with a slider.
 * Shows the interpolated 28×28 image and fires predictions in real-time.
 */
export function DigitMorph({ slotA, slotB, onMorphPredict, onSaveSlot }: DigitMorphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [morphT, setMorphT] = useState(0.5);
  const prevTRef = useRef(morphT);
  const bothReady = slotA !== null && slotB !== null;

  // Render the interpolated image and fire prediction
  const renderMorph = useCallback((t: number) => {
    if (!slotA || !slotB) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = 28;
    const morphed = new Array<number>(size * size);
    for (let i = 0; i < size * size; i++) {
      morphed[i] = slotA[i] * (1 - t) + slotB[i] * t;
    }

    // Draw scaled-up version
    const displaySize = 140;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = displaySize * dpr;
    canvas.height = displaySize * dpr;
    ctx.scale(dpr, dpr);

    const cellSize = displaySize / size;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const v = Math.round(morphed[y * size + x] * 255);
        ctx.fillStyle = `rgb(${v}, ${v}, ${v})`;
        ctx.fillRect(x * cellSize, y * cellSize, cellSize + 0.5, cellSize + 0.5);
      }
    }

    onMorphPredict(morphed);
  }, [slotA, slotB, onMorphPredict]);

  // Render when morph slider changes or slots update
  useEffect(() => {
    if (bothReady) {
      renderMorph(morphT);
    }
  }, [morphT, bothReady, renderMorph]);

  // Also re-render if the previous t changed (avoids stale closure)
  useEffect(() => {
    prevTRef.current = morphT;
  }, [morphT]);

  const handleSlider = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setMorphT(parseFloat(e.target.value));
  }, []);

  return (
    <div className="digit-morph" role="group" aria-label="Digit morphing">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🔀</span>
        <span>Digit Morph</span>
      </div>

      <div className="morph-slots">
        <button
          className={`morph-slot-btn ${slotA ? 'saved' : ''}`}
          onClick={() => onSaveSlot('A')}
          aria-label={slotA ? 'Slot A saved — click to overwrite' : 'Save current drawing as Slot A'}
        >
          {slotA ? '✓ A' : 'Save A'}
        </button>
        <span className="morph-arrow" aria-hidden="true">⟷</span>
        <button
          className={`morph-slot-btn ${slotB ? 'saved' : ''}`}
          onClick={() => onSaveSlot('B')}
          aria-label={slotB ? 'Slot B saved — click to overwrite' : 'Save current drawing as Slot B'}
        >
          {slotB ? '✓ B' : 'Save B'}
        </button>
      </div>

      {bothReady ? (
        <div className="morph-content">
          <canvas
            ref={canvasRef}
            style={{ width: 140, height: 140 }}
            className="morph-canvas"
            role="img"
            aria-label={`Morphed digit — ${Math.round(morphT * 100)}% toward B`}
          />
          <div className="morph-slider-row">
            <span className="morph-label">A</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={morphT}
              onChange={handleSlider}
              className="slider morph-slider"
              aria-label="Morph blend slider"
              aria-valuemin={0}
              aria-valuemax={1}
              aria-valuenow={morphT}
            />
            <span className="morph-label">B</span>
          </div>
          <div className="morph-value" aria-hidden="true">
            {Math.round((1 - morphT) * 100)}% A — {Math.round(morphT * 100)}% B
          </div>
        </div>
      ) : (
        <div className="morph-empty">
          <p>Draw a digit and save it to each slot, then morph between them.</p>
        </div>
      )}
    </div>
  );
}

export default DigitMorph;
