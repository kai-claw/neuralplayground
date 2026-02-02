import { useRef, useEffect } from 'react';
import type { TrainingConfig, ActivationFn } from '../nn/NeuralNetwork';
import { useTrainingRace, RACE_PRESETS } from '../hooks/useTrainingRace';
import type { RacerConfig, RaceState } from '../hooks/useTrainingRace';
import { RACE_EPOCHS, RACE_CHART_HEIGHT } from '../constants';

/**
 * Training Race — two networks with different architectures race to learn
 * the same data. Side-by-side loss/accuracy curves with a winner announcement.
 */
export function TrainingRace() {
  const chartRef = useRef<HTMLCanvasElement>(null);

  const {
    racerA,
    racerB,
    raceState,
    startRace,
    stopRace,
    applyPreset,
    updateRacerLayers,
  } = useTrainingRace();

  // Draw the race chart
  useEffect(() => {
    drawRaceChart(chartRef.current, raceState, racerA, racerB);
  }, [raceState, racerA, racerB]);

  const archStr = (config: TrainingConfig) =>
    `784→${config.layers.map(l => l.neurons).join('→')}→10 (lr=${config.learningRate})`;

  return (
    <div className="training-race" role="group" aria-label="Training race comparison">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🏁</span>
        <span>Training Race</span>
      </div>

      <div className="race-content">
        {/* Presets */}
        <div className="race-presets">
          {RACE_PRESETS.map((preset, i) => (
            <button
              key={i}
              className="race-preset-btn"
              onClick={() => applyPreset(preset)}
              disabled={raceState.isRacing}
            >
              {preset.label}
            </button>
          ))}
        </div>

        {/* Racer configs */}
        <div className="race-configs">
          {(['A', 'B'] as const).map((side) => {
            const racer = side === 'A' ? racerA : racerB;
            const firstLayer = racer.config.layers[0];
            return (
              <div key={side} className="race-config-card" style={{ borderColor: racer.color }}>
                <div className="race-config-header" style={{ color: racer.color }}>
                  {racer.name}
                </div>
                <div className="race-config-arch">{archStr(racer.config)}</div>
                <div className="race-config-controls">
                  <select
                    value={firstLayer.neurons}
                    onChange={(e) =>
                      updateRacerLayers(side, parseInt(e.target.value), firstLayer.activation)
                    }
                    disabled={raceState.isRacing}
                    className="select-small"
                    aria-label={`${racer.name} neurons`}
                  >
                    {[8, 16, 32, 64, 128].map(n => (
                      <option key={n} value={n}>{n} neurons</option>
                    ))}
                  </select>
                  <select
                    value={firstLayer.activation}
                    onChange={(e) =>
                      updateRacerLayers(side, firstLayer.neurons, e.target.value as ActivationFn)
                    }
                    disabled={raceState.isRacing}
                    className="select-small"
                    aria-label={`${racer.name} activation`}
                  >
                    <option value="relu">ReLU</option>
                    <option value="sigmoid">Sigmoid</option>
                    <option value="tanh">Tanh</option>
                  </select>
                </div>
              </div>
            );
          })}
        </div>

        {/* Race chart */}
        <canvas
          ref={chartRef}
          style={{ width: 420, height: RACE_CHART_HEIGHT }}
          className="race-chart-canvas"
          role="img"
          aria-label={raceState.epoch > 0
            ? `Race chart: epoch ${raceState.epoch}/${RACE_EPOCHS}`
            : 'Race chart — not started'}
        />

        {/* Controls */}
        <div className="race-controls">
          {raceState.isRacing ? (
            <button className="btn btn-danger race-btn" onClick={stopRace}>
              ⏹ Stop Race
            </button>
          ) : (
            <button className="btn btn-primary race-btn" onClick={startRace}>
              🏁 Race!
            </button>
          )}
        </div>

        {/* Winner */}
        {raceState.winner && (
          <div className={`race-winner race-winner-${raceState.winner.toLowerCase()}`}>
            {raceState.winner === 'tie' ? (
              <span>🤝 It's a tie! Both networks converged similarly.</span>
            ) : (
              <span>
                🏆 <strong>{raceState.winner === 'A' ? racerA.name : racerB.name}</strong> wins
                with {((raceState.winner === 'A' ? raceState.accA : raceState.accB).slice(-1)[0] * 100).toFixed(1)}% accuracy!
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** Pure rendering function for the race chart (no React dependency) */
function drawRaceChart(
  canvas: HTMLCanvasElement | null,
  raceState: RaceState,
  racerA: RacerConfig,
  racerB: RacerConfig,
) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const width = 420;
  const height = RACE_CHART_HEIGHT;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const pad = { top: 20, right: 50, bottom: 24, left: 40 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  // Grid
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
  }

  const { accA, accB, epoch } = raceState;

  if (accA.length === 0 && accB.length === 0) {
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Configure networks and click Race!', width / 2, height / 2);
    return;
  }

  const maxEpochs = Math.max(accA.length, accB.length, RACE_EPOCHS);

  const drawLine = (data: number[], color: string, maxVal: number) => {
    if (data.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = pad.left + (i / Math.max(maxEpochs - 1, 1)) * plotW;
      const y = pad.top + plotH - (Math.min(data[i], maxVal) / maxVal) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Area fill
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = pad.left + (i / Math.max(maxEpochs - 1, 1)) * plotW;
      const y = pad.top + plotH - (Math.min(data[i], maxVal) / maxVal) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    const lastX = pad.left + ((data.length - 1) / Math.max(maxEpochs - 1, 1)) * plotW;
    ctx.lineTo(lastX, pad.top + plotH);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.closePath();
    ctx.globalAlpha = 0.06;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1;
  };

  drawLine(accA, racerA.color, 1);
  drawLine(accB, racerB.color, 1);

  // Y-axis labels
  ctx.fillStyle = '#9ca3af';
  ctx.font = '9px Inter, sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = (1 / 4) * (4 - i);
    const y = pad.top + (plotH / 4) * i;
    ctx.fillText(`${(val * 100).toFixed(0)}%`, pad.left - 4, y + 3);
  }

  // Legend
  ctx.font = 'bold 10px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillStyle = racerA.color;
  ctx.fillText(`A: ${accA.length > 0 ? (accA[accA.length - 1] * 100).toFixed(1) : 0}%`, pad.left + 4, pad.top - 6);
  ctx.fillStyle = racerB.color;
  ctx.fillText(`B: ${accB.length > 0 ? (accB[accB.length - 1] * 100).toFixed(1) : 0}%`, pad.left + plotW / 2, pad.top - 6);

  // Epoch label
  ctx.fillStyle = '#6b7280';
  ctx.font = '9px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`Epoch ${epoch}/${RACE_EPOCHS}`, width / 2, height - 4);

  // Winner
  if (raceState.winner) {
    ctx.fillStyle = raceState.winner === 'A' ? racerA.color : raceState.winner === 'B' ? racerB.color : '#9ca3af';
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.textAlign = 'right';
    const label = raceState.winner === 'tie' ? '🤝 Tie!' :
      `🏆 ${raceState.winner === 'A' ? racerA.name : racerB.name} wins!`;
    ctx.fillText(label, pad.left + plotW, pad.top - 6);
  }
}

export default TrainingRace;
