interface PredictionBarProps {
  probabilities: number[] | null;
}

export function PredictionBar({ probabilities }: PredictionBarProps) {
  if (!probabilities) {
    return (
      <div className="prediction-bar" role="group" aria-label="Predictions">
        <div className="panel-header">
          <span className="panel-icon" aria-hidden="true">🎯</span>
          <span>Predictions</span>
        </div>
        <div className="prediction-empty">Draw a digit to see predictions</div>
      </div>
    );
  }

  // Stack-safe max
  let maxProb = 0;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) maxProb = probabilities[i];
  }
  const predictedDigit = probabilities.indexOf(maxProb);

  return (
    <div className="prediction-bar" role="group" aria-label={`Predictions — predicted digit: ${predictedDigit}`}>
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🎯</span>
        <span>Predictions</span>
        <span className="predicted-digit" aria-label={`Predicted digit: ${predictedDigit}`}>→ {predictedDigit}</span>
      </div>
      <div className="probability-bars" role="list" aria-label="Digit probabilities">
        {probabilities.map((prob, i) => (
          <div key={i} className={`prob-row ${i === predictedDigit ? 'active' : ''}`} role="listitem" aria-label={`Digit ${i}: ${(prob * 100).toFixed(1)} percent`}>
            <span className="prob-label" aria-hidden="true">{i}</span>
            <div className="prob-track" role="progressbar" aria-valuenow={Math.round(prob * 100)} aria-valuemin={0} aria-valuemax={100}>
              <div
                className="prob-fill"
                style={{
                  width: `${prob * 100}%`,
                  backgroundColor: i === predictedDigit ? '#10b981' : '#63deff',
                  opacity: 0.3 + prob * 0.7,
                }}
              />
            </div>
            <span className="prob-value" aria-hidden="true">{(prob * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PredictionBar;
