import { useRef, useEffect } from 'react';
import type { LayerState } from '../types';

// 256-entry pre-computed color LUTs for weight heatmap (eliminates per-cell string allocation)
const POS_R = new Uint8Array(256);
const POS_G = new Uint8Array(256);
const POS_B = new Uint8Array(256);
const NEG_R = new Uint8Array(256);
const NEG_G = new Uint8Array(256);
const NEG_B = new Uint8Array(256);

for (let i = 0; i < 256; i++) {
  POS_R[i] = (i * 0.39) | 0;
  POS_G[i] = (i * 0.87) | 0;
  POS_B[i] = i;
  NEG_R[i] = i;
  NEG_G[i] = (i * 0.39) | 0;
  NEG_B[i] = (i * 0.52) | 0;
}

interface WeightHeatmapProps {
  layers: LayerState[] | null;
  selectedLayer: number;
  width?: number;
  height?: number;
}

export function WeightHeatmap({ layers, selectedLayer, width = 300, height = 200 }: WeightHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Cached ImageData for direct pixel manipulation (avoids fillRect + CSS strings)
  const imageDataRef = useRef<ImageData | null>(null);

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
    for (let j = 0; j < rows; j++) {
      const wj = weights[j];
      for (let i = 0; i < cols; i++) {
        const a = wj[i] < 0 ? -wj[i] : wj[i];
        if (a > maxW) maxW = a;
      }
    }
    if (maxW === 0) maxW = 1;

    const invMaxW = 1 / maxW;
    const offsetX = (width - cols * cellW) / 2;
    const offsetY = (height - rows * cellH) / 2;

    // Use ImageData for dense heatmaps (>500 cells), fillRect for sparse
    const totalCells = rows * cols;
    if (totalCells > 500 && cellW >= 1 && cellH >= 1) {
      // Direct pixel rendering via ImageData (zero string allocation)
      const imgW = Math.ceil(cols * cellW);
      const imgH = Math.ceil(rows * cellH);
      if (!imageDataRef.current || imageDataRef.current.width !== imgW || imageDataRef.current.height !== imgH) {
        imageDataRef.current = new ImageData(imgW, imgH);
      }
      const data = imageDataRef.current.data;
      data.fill(0); // clear

      for (let j = 0; j < rows; j++) {
        const wj = weights[j];
        const y0 = (j * cellH) | 0;
        const y1 = Math.min(imgH, ((j + 1) * cellH - 0.5) | 0);
        for (let i = 0; i < cols; i++) {
          const val = wj[i] * invMaxW;
          const x0 = (i * cellW) | 0;
          const x1 = Math.min(imgW, ((i + 1) * cellW - 0.5) | 0);
          let r: number, g: number, b: number;
          if (val > 0) {
            const idx = (val * 255) | 0;
            r = POS_R[idx]; g = POS_G[idx]; b = POS_B[idx];
          } else {
            const idx = (-val * 255) | 0;
            r = NEG_R[idx]; g = NEG_G[idx]; b = NEG_B[idx];
          }
          for (let py = y0; py < y1; py++) {
            for (let px = x0; px < x1; px++) {
              const off = (py * imgW + px) << 2;
              data[off] = r; data[off + 1] = g; data[off + 2] = b; data[off + 3] = 255;
            }
          }
        }
      }
      ctx.putImageData(imageDataRef.current, offsetX, offsetY);
    } else {
      // Sparse rendering with fillRect (few cells, string overhead negligible)
      for (let j = 0; j < rows; j++) {
        const wj = weights[j];
        for (let i = 0; i < cols; i++) {
          const val = wj[i] * invMaxW;
          const x = offsetX + i * cellW;
          const y = offsetY + j * cellH;
          if (val > 0) {
            const idx = (val * 255) | 0;
            ctx.fillStyle = `rgb(${POS_R[idx]},${POS_G[idx]},${POS_B[idx]})`;
          } else {
            const idx = (-val * 255) | 0;
            ctx.fillStyle = `rgb(${NEG_R[idx]},${NEG_G[idx]},${NEG_B[idx]})`;
          }
          ctx.fillRect(x, y, cellW - 0.5, cellH - 0.5);
        }
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
