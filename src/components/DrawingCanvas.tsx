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
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
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
    <div className="drawing-canvas-container">
      <div className="panel-header">
        <span className="panel-icon">✏️</span>
        <span>Draw a Digit</span>
      </div>
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="drawing-canvas"
        onMouseDown={handleStart}
        onMouseMove={handleMove}
        onMouseUp={handleEnd}
        onMouseLeave={handleEnd}
        onTouchStart={handleStart}
        onTouchMove={handleMove}
        onTouchEnd={handleEnd}
      />
      <div className="canvas-controls">
        <button className="btn btn-secondary" onClick={clear}>
          Clear Canvas
        </button>
      </div>
      <p className="canvas-hint">Draw a digit (0-9) and watch the network predict it</p>
    </div>
  );
}

export default DrawingCanvas;
