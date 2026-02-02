/**
 * ExperiencePanel — cinematic demo toggle.
 *
 * Provides buttons to start/stop the cinematic autoplay mode
 * and any future experience mode toggles.
 */

interface ExperiencePanelProps {
  cinematicActive: boolean;
  onStartCinematic: () => void;
}

export default function ExperiencePanel({
  cinematicActive,
  onStartCinematic,
}: ExperiencePanelProps) {
  return (
    <div className="experience-panel" role="group" aria-label="Experience modes">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">✨</span>
        <span>Experience</span>
      </div>
      <div className="experience-buttons">
        <button
          className={`btn btn-experience ${cinematicActive ? 'active' : ''}`}
          onClick={onStartCinematic}
          aria-label={cinematicActive ? 'Stop cinematic demo' : 'Start cinematic demo'}
          aria-pressed={cinematicActive}
        >
          <span aria-hidden="true">🎬</span>{' '}
          {cinematicActive ? 'Stop Demo' : 'Cinematic'}
        </button>
      </div>
    </div>
  );
}
