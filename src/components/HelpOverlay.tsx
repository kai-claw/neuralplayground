/**
 * HelpOverlay — keyboard shortcuts dialog.
 *
 * Modal overlay displaying available keyboard shortcuts.
 * Closes on backdrop click or close button.
 */

import { SHORTCUTS } from '../constants';

interface HelpOverlayProps {
  onClose: () => void;
}

export default function HelpOverlay({ onClose }: HelpOverlayProps) {
  return (
    <div
      className="help-overlay"
      role="dialog"
      aria-label="Keyboard shortcuts"
      onClick={onClose}
    >
      <div className="help-panel" onClick={(e) => e.stopPropagation()}>
        <div className="help-header">
          <h2>Keyboard Shortcuts</h2>
          <button className="btn-close" onClick={onClose} aria-label="Close help">
            ✕
          </button>
        </div>
        <div className="help-list">
          {SHORTCUTS.map(({ key, description }) => (
            <div className="help-row" key={key}>
              <kbd>{key}</kbd>
              <span>{description}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
