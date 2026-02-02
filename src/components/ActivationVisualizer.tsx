import { useRef, useEffect, useState } from 'react';
import type { LayerState } from '../nn/NeuralNetwork';

interface ActivationVisualizerProps {
  layers: LayerState[] | null;
  width?: number;
  height?: number;
}

export function ActivationVisualizer({ layers, width: propWidth, height: propHeight }: ActivationVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ width: propWidth || 320, height: propHeight || 280 });

  useEffect(() => {
    if (propWidth && propHeight) {
      setDims({ width: propWidth, height: propHeight });
      return;
    }
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0].contentRect.width;
      if (w > 0) setDims({ width: w, height: Math.round(w * 0.875) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [propWidth, propHeight]);

  const { width, height } = dims;

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

    if (!layers || layers.length === 0) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No activations yet', width / 2, height / 2);
      return;
    }

    const padding = 15;
    const layerGap = 12;
    const numLayers = layers.length;
    const layerHeight = (height - padding * 2 - (numLayers - 1) * layerGap) / numLayers;

    for (let l = 0; l < numLayers; l++) {
      const layer = layers[l];
      const activations = layer.activations;
      const numNeurons = activations.length;
      const y = padding + l * (layerHeight + layerGap);
      
      ctx.fillStyle = '#9ca3af';
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'left';
      const isOutput = l === numLayers - 1;
      ctx.fillText(isOutput ? `Output (${numNeurons})` : `Layer ${l + 1} (${numNeurons})`, padding, y + 10);
      
      const barAreaWidth = width - padding * 2;
      const maxBars = Math.min(numNeurons, 64);
      const barWidth = Math.max(2, (barAreaWidth - 60) / maxBars);
      const barMaxHeight = layerHeight - 15;
      const startX = padding + 55;
      
      let maxAct = 0;
      for (const a of activations) {
        maxAct = Math.max(maxAct, Math.abs(a));
      }
      if (maxAct === 0) maxAct = 1;
      
      for (let n = 0; n < maxBars; n++) {
        const val = activations[n];
        const normalized = val / maxAct;
        const barH = Math.abs(normalized) * barMaxHeight;
        const x = startX + n * barWidth;
        const barY = y + 15 + barMaxHeight - barH;
        
        if (val > 0) {
          const intensity = Math.min(1, Math.abs(normalized));
          ctx.fillStyle = `rgba(99, 222, 255, ${0.3 + intensity * 0.7})`;
        } else {
          const intensity = Math.min(1, Math.abs(normalized));
          ctx.fillStyle = `rgba(255, 99, 132, ${0.3 + intensity * 0.7})`;
        }
        
        ctx.fillRect(x, barY, barWidth - 1, barH);
      }
      
      if (numNeurons > maxBars) {
        ctx.fillStyle = '#6b7280';
        ctx.font = '8px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`+${numNeurons - maxBars}`, startX + maxBars * barWidth + 2, y + layerHeight - 2);
      }
    }
  }, [layers, width, height]);

  return (
    <div className="activation-visualizer" ref={containerRef} role="group" aria-label="Layer activation visualization">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">⚡</span>
        <span>Activations</span>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width, height }}
        className="activation-canvas"
        role="img"
        aria-label={layers ? `Activation values for ${layers.length} layers` : 'No activation data yet'}
      />
    </div>
  );
}

export default ActivationVisualizer;
