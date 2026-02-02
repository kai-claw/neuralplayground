/**
 * StatsPanel — training statistics display.
 *
 * Shows epoch count, loss, accuracy, and current prediction
 * with color-coded thresholds and animation on update.
 */

interface StatsPanelProps {
  epoch: number;
  isTraining: boolean;
  loss: number | null;
  accuracy: number | null;
  predictedLabel: number | null;
}

export default function StatsPanel({
  epoch,
  isTraining,
  loss,
  accuracy,
  predictedLabel,
}: StatsPanelProps) {
  return (
    <div className="stats-panel" role="region" aria-label="Training statistics">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">📊</span>
        <span>Statistics</span>
      </div>
      <div className="stats-grid">
        <div className="stat-item">
          <span
            className="stat-number"
            key={`epoch-${epoch}`}
            style={
              isTraining
                ? { animation: 'statTick 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)' }
                : undefined
            }
            aria-label={`${epoch} epochs`}
          >
            {epoch}
          </span>
          <span className="stat-desc">Epochs</span>
        </div>
        <div className="stat-item">
          <span
            className="stat-number"
            aria-label={loss !== null ? `Loss ${loss.toFixed(4)}` : 'No loss data'}
            style={loss !== null && loss < 0.5 ? { color: 'var(--accent-green)' } : undefined}
          >
            {loss !== null ? loss.toFixed(4) : '—'}
          </span>
          <span className="stat-desc">Loss</span>
        </div>
        <div className="stat-item">
          <span
            className="stat-number"
            aria-label={
              accuracy !== null
                ? `Accuracy ${(accuracy * 100).toFixed(1)} percent`
                : 'No accuracy data'
            }
            style={accuracy !== null && accuracy > 0.8 ? { color: 'var(--accent-green)' } : undefined}
          >
            {accuracy !== null ? `${(accuracy * 100).toFixed(1)}%` : '—'}
          </span>
          <span className="stat-desc">Accuracy</span>
        </div>
        <div className="stat-item">
          <span
            className="stat-number"
            aria-label={
              predictedLabel !== null ? `Predicted digit ${predictedLabel}` : 'No prediction'
            }
            style={
              predictedLabel !== null
                ? { color: 'var(--accent-green)', fontSize: '24px' }
                : undefined
            }
          >
            {predictedLabel !== null ? predictedLabel : '—'}
          </span>
          <span className="stat-desc">Prediction</span>
        </div>
      </div>
    </div>
  );
}
