import { useRef, useEffect } from 'react';
import { useContainerDims } from '../hooks/useContainerDims';
import { LOSS_CHART_DEFAULT, LOSS_CHART_ASPECT, COLOR_CYAN_HEX, COLOR_RED_HEX } from '../constants';

interface LossChartProps {
  lossHistory: number[];
  accuracyHistory: number[];
  width?: number;
  height?: number;
}

export function LossChart({ lossHistory, accuracyHistory, width: propWidth, height: propHeight }: LossChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { containerRef, dims } = useContainerDims({
    propWidth,
    propHeight,
    defaultWidth: LOSS_CHART_DEFAULT.width,
    defaultHeight: LOSS_CHART_DEFAULT.height,
    aspectRatio: LOSS_CHART_ASPECT,
  });

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

    const pad = { top: 25, right: 60, bottom: 30, left: 50 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    // Grid
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    if (lossHistory.length === 0) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Training data will appear here', width / 2, height / 2);
      return;
    }

    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', width / 2, height - 5);

    const drawLine = (data: number[], maxVal: number, color: string, label: string, rightAxis: boolean) => {
      if (data.length < 2) return;

      // Stroke the line
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = pad.left + (i / Math.max(data.length - 1, 1)) * plotW;
        const y = pad.top + plotH - (Math.min(data[i], maxVal) / maxVal) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // BUG FIX: Fill area needs its OWN path — previous code reused stroke path
      // which corrupted the second line's fill by connecting to the first line's endpoint
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = pad.left + (i / Math.max(data.length - 1, 1)) * plotW;
        const y = pad.top + plotH - (Math.min(data[i], maxVal) / maxVal) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      const lastX = pad.left + ((data.length - 1) / Math.max(data.length - 1, 1)) * plotW;
      ctx.lineTo(lastX, pad.top + plotH);
      ctx.lineTo(pad.left, pad.top + plotH);
      ctx.closePath();
      ctx.globalAlpha = 0.08;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.globalAlpha = 1;

      // Y-axis labels
      ctx.fillStyle = color;
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = rightAxis ? 'left' : 'right';
      
      for (let i = 0; i <= 4; i++) {
        const val = (maxVal / 4) * (4 - i);
        const y = pad.top + (plotH / 4) * i;
        ctx.fillText(val.toFixed(2), rightAxis ? pad.left + plotW + 5 : pad.left - 5, y + 3);
      }

      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.fillText(label, rightAxis ? pad.left + plotW + 15 : pad.left - 10, pad.top - 8);
    };

    // Stack-safe max — avoids RangeError with 10K+ epochs
    let maxLoss = 0.1;
    for (let i = 0; i < lossHistory.length; i++) {
      if (lossHistory[i] > maxLoss) maxLoss = lossHistory[i];
    }
    drawLine(lossHistory, maxLoss, COLOR_RED_HEX, 'Loss', false);
    drawLine(accuracyHistory, 1, COLOR_CYAN_HEX, 'Accuracy', true);

    // Current values in header
    if (lossHistory.length > 0) {
      const lastLoss = lossHistory[lossHistory.length - 1];
      const lastAcc = accuracyHistory[accuracyHistory.length - 1] || 0;

      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      
      ctx.fillStyle = COLOR_RED_HEX;
      ctx.fillText(`Loss: ${lastLoss.toFixed(4)}`, width / 3, pad.top - 8);
      ctx.fillStyle = COLOR_CYAN_HEX;
      ctx.fillText(`Acc: ${(lastAcc * 100).toFixed(1)}%`, (2 * width) / 3, pad.top - 8);
    }

    // Epoch ticks
    ctx.fillStyle = '#6b7280';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    const epochs = lossHistory.length;
    const tickInterval = Math.max(1, Math.floor(epochs / 5));
    for (let i = 0; i < epochs; i += tickInterval) {
      const x = pad.left + (i / Math.max(epochs - 1, 1)) * plotW;
      ctx.fillText(String(i + 1), x, pad.top + plotH + 15);
    }
  }, [lossHistory, accuracyHistory, width, height]);

  return (
    <div className="loss-chart" ref={containerRef} role="group" aria-label="Training progress charts">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">📈</span>
        <span>Training Progress</span>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width, height }}
        className="chart-canvas"
        role="img"
        aria-label={lossHistory.length > 0
          ? `Loss chart: ${lossHistory.length} epochs, current loss ${lossHistory[lossHistory.length - 1]?.toFixed(4)}, accuracy ${((accuracyHistory[accuracyHistory.length - 1] || 0) * 100).toFixed(1)}%`
          : 'Training chart — no data yet'}
      />
    </div>
  );
}

export default LossChart;
