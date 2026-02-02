interface CinematicBadgeProps {
  phase: 'training' | 'drawing' | 'predicting';
  epoch: number;
  maxEpochs: number;
  currentDigit: number;
  progress: number; // 0-1
}

export function CinematicBadge({ phase, epoch, maxEpochs, currentDigit, progress }: CinematicBadgeProps) {
  const phaseLabel = phase === 'training'
    ? `Training… epoch ${epoch}/${maxEpochs}`
    : phase === 'drawing'
      ? `Drawing digit ${currentDigit}`
      : `Predicting digit ${currentDigit}`;

  const phaseEmoji = phase === 'training' ? '🎓' : phase === 'drawing' ? '✏️' : '🎯';

  return (
    <div className="cinematic-badge" aria-live="polite" role="status">
      <span className="cinematic-pulse" aria-hidden="true" />
      <span className="cinematic-emoji" aria-hidden="true">{phaseEmoji}</span>
      <span className="cinematic-label">{phaseLabel}</span>
      <div className="cinematic-progress" aria-hidden="true">
        <div className="cinematic-progress-fill" style={{ width: `${progress * 100}%` }} />
      </div>
    </div>
  );
}

export default CinematicBadge;
