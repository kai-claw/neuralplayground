interface PredictionBarProps {
  probabilities: number[] | null;
}

export function PredictionBar({ probabilities }: PredictionBarProps) {
  if (!probabilities) {
    return (
      <div className="prediction-bar">
        <div className="panel-header">
          <span className="panel-icon">🎯</span>
          <span>Predictions</span>
        </div>
        <div className="prediction-empty">Draw a digit to see predictions</div>
      </div>
    );
  }

  const maxProb = Math.max(...probabilities);
  const predictedDigit = probabilities.indexOf(maxProb);

  return (
    <div className="prediction-bar">
      <div className="panel-header">
        <span className="panel-icon">🎯</span>
        <span>Predictions</span>
        <span className="predicted-digit">→ {predictedDigit}</span>
      </div>
      <div className="probability-bars">
        {probabilities.map((prob, i) => (
          <div key={i} className={`prob-row ${i === predictedDigit ? 'active' : ''}`}>
            <span className="prob-label">{i}</span>
            <div className="prob-track">
              <div
                className="prob-fill"
                style={{
                  width: `${prob * 100}%`,
                  backgroundColor: i === predictedDigit ? '#10b981' : '#63deff',
                  opacity: 0.3 + prob * 0.7,
                }}
              />
            </div>
            <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PredictionBar;
