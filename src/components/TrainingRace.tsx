import { useRef, useEffect } from 'react';
import type { ActivationFn } from '../types';
import { useTrainingRace } from '../hooks/useTrainingRace';
import { RACE_PRESETS } from '../data/racePresets';
import { drawRaceChart, CHART_WIDTH } from '../renderers/raceChart';
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
    const canvas = chartRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = CHART_WIDTH * dpr;
    canvas.height = RACE_CHART_HEIGHT * dpr;
    ctx.scale(dpr, dpr);

    drawRaceChart(ctx, CHART_WIDTH, RACE_CHART_HEIGHT, {
      accA: raceState.accA,
      accB: raceState.accB,
      epoch: raceState.epoch,
      winner: raceState.winner,
    }, racerA, racerB);
  }, [raceState, racerA, racerB]);

  const archStr = (config: { learningRate: number; layers: { neurons: number }[] }) =>
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
          style={{ width: CHART_WIDTH, height: RACE_CHART_HEIGHT }}
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

export default TrainingRace;
