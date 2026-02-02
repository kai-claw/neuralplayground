import { useRef, useEffect, useCallback } from 'react';

interface DrawingCanvasProps {
  onDraw: (imageData: ImageData) => void;
  size?: number;
}

export function DrawingCanvas({ onDraw, size = 280 }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, size, size);
  }, [size]);

  const getPos = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    // BUG FIX: touchEnd has no .touches — use changedTouches fallback
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
    const imageData = ctx.getImageData(0, 0, size, size);
    onDraw(imageData);
  }, [onDraw, size]);

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
  }, []);

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

export default DrawingCanvas;
