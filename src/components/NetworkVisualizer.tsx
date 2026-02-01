import { useRef, useEffect } from 'react';
import type { LayerState } from '../nn/NeuralNetwork';

interface NetworkVisualizerProps {
  layers: LayerState[] | null;
  inputSize: number;
  width?: number;
  height?: number;
}

function getColor(value: number, alpha = 1): string {
  if (value > 0) {
    const intensity = Math.min(1, Math.abs(value));
    return `rgba(99, 222, 255, ${intensity * alpha})`;
  } else {
    const intensity = Math.min(1, Math.abs(value));
    return `rgba(255, 99, 132, ${intensity * alpha})`;
  }
}

function getWeightColor(value: number): string {
  const clamped = Math.max(-1, Math.min(1, value));
  if (clamped > 0) {
    return `rgba(99, 222, 255, ${Math.abs(clamped) * 0.6})`;
  } else {
    return `rgba(255, 99, 132, ${Math.abs(clamped) * 0.6})`;
  }
}

export function NetworkVisualizer({ layers, inputSize, width = 600, height = 400 }: NetworkVisualizerProps) {
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

    if (!layers || layers.length === 0) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Start training to visualize the network', width / 2, height / 2);
      return;
    }

    const layerSizes = [Math.min(inputSize, 20), ...layers.map(l => l.activations.length)];
    const numLayers = layerSizes.length;
    const padding = 50;
    const layerSpacing = (width - padding * 2) / (numLayers - 1);

    const nodePositions: { x: number; y: number }[][] = [];
    
    for (let l = 0; l < numLayers; l++) {
      const nodes: { x: number; y: number }[] = [];
      const numNodes = Math.min(layerSizes[l], 16);
      const maxH = height - padding * 2;
      const nodeSpacing = Math.min(maxH / (numNodes + 1), 25);
      const startY = height / 2 - (numNodes - 1) * nodeSpacing / 2;
      
      for (let n = 0; n < numNodes; n++) {
        nodes.push({
          x: padding + l * layerSpacing,
          y: startY + n * nodeSpacing,
        });
      }
      
      if (layerSizes[l] > 16) {
        nodes.push({
          x: padding + l * layerSpacing,
          y: startY + numNodes * nodeSpacing,
        });
      }
      
      nodePositions.push(nodes);
    }

    // Draw connections
    for (let l = 1; l < numLayers; l++) {
      const prevNodes = nodePositions[l - 1];
      const currNodes = nodePositions[l];
      const layer = layers[l - 1];
      
      const maxPrev = Math.min(prevNodes.length, layerSizes[l - 1] > 16 ? 15 : prevNodes.length);
      const maxCurr = Math.min(currNodes.length, layerSizes[l] > 16 ? 15 : currNodes.length);
      
      for (let j = 0; j < maxCurr; j++) {
        for (let i = 0; i < maxPrev; i++) {
          if (j < layer.weights.length && i < (layer.weights[j]?.length || 0)) {
            const weight = layer.weights[j][i];
            ctx.strokeStyle = getWeightColor(weight);
            ctx.lineWidth = Math.min(2, Math.abs(weight) * 1.5);
            ctx.beginPath();
            ctx.moveTo(prevNodes[i].x, prevNodes[i].y);
            ctx.lineTo(currNodes[j].x, currNodes[j].y);
            ctx.stroke();
          }
        }
      }
    }

    // Draw nodes
    for (let l = 0; l < numLayers; l++) {
      const nodes = nodePositions[l];
      const isInput = l === 0;
      const isOutput = l === numLayers - 1;
      
      for (let n = 0; n < nodes.length; n++) {
        const { x, y } = nodes[n];
        const isTruncated = layerSizes[l] > 16 && n === nodes.length - 1;
        
        if (isTruncated) {
          ctx.fillStyle = '#6b7280';
          ctx.font = '12px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(`+${layerSizes[l] - 16}`, x, y + 4);
          continue;
        }
        
        let activation = 0;
        if (!isInput && l - 1 < layers.length) {
          activation = layers[l - 1].activations[n] || 0;
        }
        
        const radius = isOutput ? 10 : 7;
        
        if (Math.abs(activation) > 0.3) {
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 2.5);
          gradient.addColorStop(0, activation > 0 ? 'rgba(99, 222, 255, 0.3)' : 'rgba(255, 99, 132, 0.3)');
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(x, y, radius * 2.5, 0, Math.PI * 2);
          ctx.fill();
        }
        
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isInput ? '#374151' : getColor(activation, 0.8);
        ctx.fill();
        ctx.strokeStyle = '#4b5563';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        if (isOutput) {
          ctx.fillStyle = '#e5e7eb';
          ctx.font = 'bold 9px Inter, sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(String(n), x + radius + 4, y + 3);
          
          const prob = layers[layers.length - 1].activations[n] || 0;
          const barWidth = 40;
          const barHeight = 6;
          const barX = x + radius + 16;
          const barY = y - 3;
          
          ctx.fillStyle = '#1f2937';
          ctx.fillRect(barX, barY, barWidth, barHeight);
          ctx.fillStyle = prob > 0.5 ? '#10b981' : '#63deff';
          ctx.fillRect(barX, barY, barWidth * prob, barHeight);
          
          ctx.fillStyle = '#9ca3af';
          ctx.font = '8px Inter, sans-serif';
          ctx.fillText(`${(prob * 100).toFixed(0)}%`, barX + barWidth + 4, y + 3);
        }
      }
      
      const labelY = height - 15;
      ctx.fillStyle = '#9ca3af';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      const labelX = padding + l * layerSpacing;
      
      if (isInput) ctx.fillText('Input', labelX, labelY);
      else if (isOutput) ctx.fillText('Output', labelX, labelY);
      else ctx.fillText(`Hidden ${l}`, labelX, labelY);
    }
  }, [layers, inputSize, width, height]);

  return (
    <div className="network-visualizer">
      <div className="panel-header">
        <span className="panel-icon">🧠</span>
        <span>Network Architecture</span>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width, height }}
        className="network-canvas"
      />
    </div>
  );
}

export default NetworkVisualizer;
