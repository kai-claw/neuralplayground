import { useRef, useEffect, useCallback, useState } from 'react';
import {
  DREAM_DISPLAY_SIZE,
  DREAM_STEPS,
  DREAM_LR,
  DREAM_ANIMATION_INTERVAL,
  INPUT_DIM,
} from '../constants';
import { renderDreamImage, renderDreamGallery, GALLERY_DIMS } from '../renderers/dreamRenderer';

interface NetworkDreamsProps {
  /** Dream function from the neural network */
  onDream: (
    targetClass: number,
    steps?: number,
    lr?: number,
    startImage?: number[],
  ) => { image: number[]; confidenceHistory: number[] } | null;
  /** Whether network has been trained (epoch > 0) */
  hasTrained: boolean;
}

/**
 * Network Dreams — run the network backwards via gradient ascent to visualize
 * what the network "imagines" would produce each digit.
 * 
 * Starting from noise, iteratively optimizes the input image to maximize
 * the network's confidence for a target digit class.
 */
export function NetworkDreams({ onDream, hasTrained }: NetworkDreamsProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const galleryCanvasRef = useRef<HTMLCanvasElement>(null);
  const [targetDigit, setTargetDigit] = useState(0);
  const [isDreaming, setIsDreaming] = useState(false);
  const [dreamImage, setDreamImage] = useState<number[] | null>(null);
  const [dreamConfidence, setDreamConfidence] = useState(0);
  const [dreamStep, setDreamStep] = useState(0);
  const [gallery, setGallery] = useState<(number[] | null)[]>(new Array(10).fill(null));
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // Render current dream image to canvas
  useEffect(() => {
    if (!dreamImage) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = DREAM_DISPLAY_SIZE * dpr;
    canvas.height = DREAM_DISPLAY_SIZE * dpr;
    ctx.scale(dpr, dpr);

    renderDreamImage(ctx, dreamImage, DREAM_DISPLAY_SIZE, dreamStep, dreamConfidence);
  }, [dreamImage, dreamStep, dreamConfidence]);

  // Render gallery of all-digit dreams
  useEffect(() => {
    const canvas = galleryCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = GALLERY_DIMS.width * dpr;
    canvas.height = GALLERY_DIMS.height * dpr;
    ctx.scale(dpr, dpr);

    renderDreamGallery(ctx, gallery);
  }, [gallery]);

  const startDream = useCallback(() => {
    if (!hasTrained || isDreaming) return;
    setIsDreaming(true);
    setDreamStep(0);
    setDreamConfidence(0);
    setDreamImage(null);

    // Animate the dream step by step
    let step = 0;
    const startImage = Array.from(
      { length: INPUT_DIM * INPUT_DIM },
      () => Math.random() * 0.3 + 0.1,
    );
    let currentImage = [...startImage];
    let lr = DREAM_LR;

    const animate = () => {
      if (step >= DREAM_STEPS) {
        setIsDreaming(false);
        setGallery(prev => {
          const next = [...prev];
          next[targetDigit] = currentImage;
          return next;
        });
        return;
      }

      // Run a small batch of steps for each animation frame
      const batchSize = 4;
      for (let i = 0; i < batchSize && step < DREAM_STEPS; i++) {
        const result = onDream(targetDigit, 1, lr, currentImage);
        if (result) {
          currentImage = result.image;
          lr *= 0.998;
        }
        step++;
      }

      // Get current confidence
      const checkResult = onDream(targetDigit, 0, 0, currentImage);
      const confidence = checkResult ? checkResult.confidenceHistory[0] || 0 : 0;

      setDreamImage([...currentImage]);
      setDreamStep(step);
      setDreamConfidence(confidence);

      timerRef.current = setTimeout(animate, DREAM_ANIMATION_INTERVAL);
    };

    animate();
  }, [hasTrained, isDreaming, targetDigit, onDream]);

  const dreamAll = useCallback(() => {
    if (!hasTrained || isDreaming) return;

    // Dream all 10 digits at once (no animation)
    const newGallery: (number[] | null)[] = [];
    for (let d = 0; d < 10; d++) {
      const result = onDream(d, DREAM_STEPS, DREAM_LR);
      newGallery.push(result ? result.image : null);
    }
    setGallery(newGallery);
  }, [hasTrained, isDreaming, onDream]);

  const handleStop = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setIsDreaming(false);
  }, []);

  return (
    <div className="network-dreams" role="group" aria-label="Network dreams visualization">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">💭</span>
        <span>Network Dreams</span>
      </div>

      {!hasTrained ? (
        <div className="dreams-empty">
          <p>Train the network first, then see what it "imagines" for each digit.</p>
        </div>
      ) : (
        <div className="dreams-content">
          <div className="dreams-target-selector" role="radiogroup" aria-label="Target digit for dreaming">
            {Array.from({ length: 10 }, (_, i) => (
              <button
                key={i}
                className={`dreams-digit-btn ${targetDigit === i ? 'active' : ''}`}
                onClick={() => setTargetDigit(i)}
                disabled={isDreaming}
                role="radio"
                aria-checked={targetDigit === i}
              >
                {i}
              </button>
            ))}
          </div>

          <div className="dreams-main">
            <canvas
              ref={canvasRef}
              style={{ width: DREAM_DISPLAY_SIZE, height: DREAM_DISPLAY_SIZE }}
              className="dreams-canvas"
              role="img"
              aria-label={dreamImage
                ? `Network dream for digit ${targetDigit} — ${(dreamConfidence * 100).toFixed(1)}% confidence`
                : 'No dream generated yet'}
            />

            {isDreaming && (
              <div className="dreams-progress">
                <div className="dreams-progress-bar">
                  <div
                    className="dreams-progress-fill"
                    style={{ width: `${(dreamStep / DREAM_STEPS) * 100}%` }}
                  />
                </div>
                <span className="dreams-progress-text">
                  Dreaming... {dreamStep}/{DREAM_STEPS}
                </span>
              </div>
            )}
          </div>

          <div className="dreams-buttons">
            {isDreaming ? (
              <button className="btn btn-danger dreams-btn" onClick={handleStop}>
                ⏹ Stop
              </button>
            ) : (
              <button className="btn btn-primary dreams-btn" onClick={startDream}>
                💭 Dream Digit {targetDigit}
              </button>
            )}
            <button
              className="btn btn-secondary dreams-btn"
              onClick={dreamAll}
              disabled={isDreaming}
            >
              🌈 Dream All 10
            </button>
          </div>

          <div className="dreams-gallery">
            <div className="dreams-gallery-label">Dream Gallery</div>
            <canvas
              ref={galleryCanvasRef}
              style={{ width: GALLERY_DIMS.width, height: GALLERY_DIMS.height }}
              className="dreams-gallery-canvas"
              role="img"
              aria-label={`Gallery of dreamed digits — ${gallery.filter(Boolean).length}/10 generated`}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default NetworkDreams;
