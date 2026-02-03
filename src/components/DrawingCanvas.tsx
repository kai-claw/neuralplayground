import { useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react';

interface DrawingCanvasProps {
  onDraw: (imageData: ImageData) => void;
  size?: number;
}

/** Imperative handle for programmatic drawing (cinematic mode, etc.) */
export interface DrawingCanvasHandle {
  clear: () => void;
  drawStroke: (x1: number, y1: number, x2: number, y2: number) => void;
  drawDot: (x: number, y: number) => void;
  getImageData: () => ImageData | null;
}

/** Minimum ms between prediction callbacks during continuous drawing.
 *  ~33ms = 30fps prediction rate (drawing is still 60fps visually). */
const PREDICT_THROTTLE_MS = 33;

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(
  function DrawingCanvas({ onDraw, size = 280 }, ref) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const isDrawingRef = useRef(false);
    const lastPosRef = useRef<{ x: number; y: number } | null>(null);
    // Throttle prediction callbacks during fast drawing to reduce CPU load
    const lastPredictTimeRef = useRef(0);
    const pendingPredictRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, size, size);
    }, [size]);

    // Expose imperative methods for programmatic drawing
    useImperativeHandle(ref, () => ({
      clear() {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, size, size);
      },
      drawStroke(x1: number, y1: number, x2: number, y2: number) {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        // Fire prediction update
        const imageData = ctx.getImageData(0, 0, size, size);
        onDraw(imageData);
      },
      drawDot(x: number, y: number) {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(x, y, 9, 0, Math.PI * 2);
        ctx.fill();
        const imageData = ctx.getImageData(0, 0, size, size);
        onDraw(imageData);
      },
      getImageData() {
        const canvas = canvasRef.current;
        if (!canvas) return null;
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;
        return ctx.getImageData(0, 0, size, size);
      },
    }), [size, onDraw]);

    const getPos = useCallback((e: React.MouseEvent | React.TouchEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      let clientX: number, clientY: number;
      if ('touches' in e) {
        const touch = e.touches[0] || e.changedTouches?.[0];
        if (!touch) return { x: 0, y: 0 };
        clientX = touch.clientX;
        clientY = touch.clientY;
      } else {
        clientX = e.clientX;
        clientY = e.clientY;
      }
      return {
        x: (clientX - rect.left) * (size / rect.width),
        y: (clientY - rect.top) * (size / rect.height),
      };
    }, [size]);

    /** Fire throttled prediction callback — always fires on stroke end via flush. */
    const firePrediction = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      lastPredictTimeRef.current = performance.now();
      if (pendingPredictRef.current) {
        clearTimeout(pendingPredictRef.current);
        pendingPredictRef.current = null;
      }
      const imageData = ctx.getImageData(0, 0, size, size);
      onDraw(imageData);
    }, [onDraw, size]);

    const draw = useCallback((x: number, y: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.strokeStyle = '#ffffff';
      ctx.fillStyle = '#ffffff';
      ctx.lineWidth = 18;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      if (lastPosRef.current) {
        ctx.beginPath();
        ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y);
        ctx.lineTo(x, y);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(x, y, 9, 0, Math.PI * 2);
        ctx.fill();
      }

      lastPosRef.current = { x, y };

      // Throttle prediction: fire immediately if enough time has passed,
      // otherwise schedule a trailing fire so the last stroke position always predicts
      const now = performance.now();
      if (now - lastPredictTimeRef.current >= PREDICT_THROTTLE_MS) {
        firePrediction();
      } else if (!pendingPredictRef.current) {
        const remaining = PREDICT_THROTTLE_MS - (now - lastPredictTimeRef.current);
        pendingPredictRef.current = setTimeout(firePrediction, remaining);
      }
    }, [firePrediction]);

    const handleStart = useCallback((e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      isDrawingRef.current = true;
      lastPosRef.current = null;
      const pos = getPos(e);
      draw(pos.x, pos.y);
    }, [getPos, draw]);

    const handleMove = useCallback((e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      if (!isDrawingRef.current) return;
      const pos = getPos(e);
      draw(pos.x, pos.y);
    }, [getPos, draw]);

    const handleEnd = useCallback(() => {
      isDrawingRef.current = false;
      lastPosRef.current = null;
      // Flush any pending throttled prediction on stroke end
      if (pendingPredictRef.current) {
        clearTimeout(pendingPredictRef.current);
        pendingPredictRef.current = null;
      }
      firePrediction();
    }, [firePrediction]);

    const clear = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, size, size);
    }, [size]);

    return (
      <div className="drawing-canvas-container" role="group" aria-label="Drawing area">
        <div className="panel-header">
          <span className="panel-icon" aria-hidden="true">✏️</span>
          <span>Draw a Digit</span>
        </div>
        <canvas
          ref={canvasRef}
          width={size}
          height={size}
          className="drawing-canvas"
          role="img"
          aria-label="Drawing canvas for digit input. Use mouse or touch to draw a digit 0-9."
          onMouseDown={handleStart}
          onMouseMove={handleMove}
          onMouseUp={handleEnd}
          onMouseLeave={handleEnd}
          onTouchStart={handleStart}
          onTouchMove={handleMove}
          onTouchEnd={handleEnd}
        />
        <div className="canvas-controls">
          <button className="btn btn-secondary" onClick={clear} aria-label="Clear drawing canvas">
            Clear Canvas
          </button>
        </div>
        <p className="canvas-hint">Draw a digit (0-9) and watch the network predict it</p>
      </div>
    );
  }
);

export default DrawingCanvas;
