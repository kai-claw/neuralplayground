import { useRef, useEffect } from 'react';
import type { LayerState } from '../nn/NeuralNetwork';

interface WeightHeatmapProps {
  layers: LayerState[] | null;
  selectedLayer: number;
  width?: number;
  height?: number;
}

export function WeightHeatmap({ layers, selectedLayer, width = 300, height = 200 }: WeightHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (!layers || selectedLayer >= layers.length) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No data yet', width / 2, height / 2);
      return;
    }

    const layer = layers[selectedLayer];
    const weights = layer.weights;
    const rows = weights.length;
    const cols = weights[0]?.length || 0;

    if (rows === 0 || cols === 0) return;

    const padding = 10;
    const cellW = Math.min((width - padding * 2) / cols, 8);
    const cellH = Math.min((height - padding * 2) / rows, 8);

    let maxW = 0;
    for (const row of weights) {
      for (const w of row) {
        maxW = Math.max(maxW, Math.abs(w));
      }
    }
    if (maxW === 0) maxW = 1;

    const offsetX = (width - cols * cellW) / 2;
    const offsetY = (height - rows * cellH) / 2;

    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const val = weights[j][i] / maxW;
        const x = offsetX + i * cellW;
        const y = offsetY + j * cellH;

        if (val > 0) {
          const intensity = Math.floor(val * 255);
          ctx.fillStyle = `rgb(${Math.floor(intensity * 0.39)}, ${Math.floor(intensity * 0.87)}, ${intensity})`;
        } else {
          const intensity = Math.floor(Math.abs(val) * 255);
          ctx.fillStyle = `rgb(${intensity}, ${Math.floor(intensity * 0.39)}, ${Math.floor(intensity * 0.52)})`;
        }

        ctx.fillRect(x, y, cellW - 0.5, cellH - 0.5);
      }
    }

    const legendY = height - 8;
    ctx.font = '9px Inter, sans-serif';
    ctx.fillStyle = '#9ca3af';
    ctx.textAlign = 'center';
    ctx.fillText(`Layer ${selectedLayer + 1} — ${rows}×${cols} weights`, width / 2, legendY);
  }, [layers, selectedLayer, width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className="heatmap-canvas"
    />
  );
}

export default WeightHeatmap;
