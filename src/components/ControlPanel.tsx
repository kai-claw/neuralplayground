import { useState } from 'react';
import type { TrainingConfig, LayerConfig, ActivationFn } from '../nn/NeuralNetwork';

interface ControlPanelProps {
  config: TrainingConfig;
  isTraining: boolean;
  epoch: number;
  onUpdateConfig: (updates: Partial<TrainingConfig>) => void;
  onUpdateLayers: (layers: LayerConfig[]) => void;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
}

export function ControlPanel({
  config,
  isTraining,
  epoch,
  onUpdateConfig,
  onUpdateLayers,
  onStart,
  onStop,
  onReset,
}: ControlPanelProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const addLayer = () => {
    const newLayers = [...config.layers, { neurons: 32, activation: 'relu' as ActivationFn }];
    onUpdateLayers(newLayers);
  };

  const removeLayer = (index: number) => {
    if (config.layers.length <= 1) return;
    const newLayers = config.layers.filter((_, i) => i !== index);
    onUpdateLayers(newLayers);
  };

  const updateLayer = (index: number, updates: Partial<LayerConfig>) => {
    const newLayers = config.layers.map((l, i) => (i === index ? { ...l, ...updates } : l));
    onUpdateLayers(newLayers);
  };

  return (
    <div className="control-panel" role="complementary" aria-label="Network controls">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">⚙️</span>
        <span>Controls</span>
      </div>

      <div className="control-section">
        <div className="control-buttons">
          {!isTraining ? (
            <button className="btn btn-primary" onClick={onStart} aria-label="Start training">
              <span className="btn-icon" aria-hidden="true">▶</span> Train
            </button>
          ) : (
            <button className="btn btn-danger" onClick={onStop} aria-label="Pause training">
              <span className="btn-icon" aria-hidden="true">⏸</span> Pause
            </button>
          )}
          <button className="btn btn-secondary" onClick={onReset} disabled={isTraining} aria-label="Reset network">
            <span className="btn-icon" aria-hidden="true">↺</span> Reset
          </button>
        </div>

        <div className="stat-row">
          <span className="stat-label">Epoch</span>
          <span className="stat-value" aria-label={`Current epoch: ${epoch}`}>{epoch}</span>
        </div>
      </div>

      <div className="control-section">
        <label className="control-label" htmlFor="lr-slider">
          Learning Rate
          <span className="control-value">{config.learningRate}</span>
        </label>
        <input
          id="lr-slider"
          type="range"
          min="0.001"
          max="0.1"
          step="0.001"
          value={config.learningRate}
          onChange={(e) => onUpdateConfig({ learningRate: parseFloat(e.target.value) })}
          disabled={isTraining}
          className="slider"
          aria-valuemin={0.001}
          aria-valuemax={0.1}
          aria-valuenow={config.learningRate}
        />
        <div className="slider-labels" aria-hidden="true">
          <span>0.001</span>
          <span>0.1</span>
        </div>
      </div>

      <div className="control-section">
        <button
          className="section-header"
          onClick={() => setShowAdvanced(!showAdvanced)}
          aria-expanded={showAdvanced}
          aria-controls="layers-config"
        >
          <span>Network Architecture</span>
          <span className="toggle-icon" aria-hidden="true">{showAdvanced ? '▾' : '▸'}</span>
        </button>

        {showAdvanced && (
          <div className="layers-config" id="layers-config">
            {config.layers.map((layer, i) => (
              <div key={i} className="layer-config">
                <div className="layer-header">
                  <span className="layer-name">Layer {i + 1}</span>
                  {config.layers.length > 1 && (
                    <button
                      className="btn-icon-small"
                      onClick={() => removeLayer(i)}
                      disabled={isTraining}
                      aria-label={`Remove layer ${i + 1}`}
                    >
                      ✕
                    </button>
                  )}
                </div>

                <div className="layer-controls">
                  <label className="mini-label" htmlFor={`neurons-${i}`}>
                    Neurons
                    <select
                      id={`neurons-${i}`}
                      value={layer.neurons}
                      onChange={(e) => updateLayer(i, { neurons: parseInt(e.target.value) })}
                      disabled={isTraining}
                      className="select-small"
                    >
                      {[8, 16, 32, 64, 128, 256].map((n) => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </label>

                  <label className="mini-label" htmlFor={`activation-${i}`}>
                    Activation
                    <select
                      id={`activation-${i}`}
                      value={layer.activation}
                      onChange={(e) => updateLayer(i, { activation: e.target.value as ActivationFn })}
                      disabled={isTraining}
                      className="select-small"
                    >
                      <option value="relu">ReLU</option>
                      <option value="sigmoid">Sigmoid</option>
                      <option value="tanh">Tanh</option>
                    </select>
                  </label>
                </div>
              </div>
            ))}

            <button
              className="btn btn-ghost btn-add-layer"
              onClick={addLayer}
              disabled={isTraining || config.layers.length >= 5}
              aria-label="Add a new hidden layer"
            >
              + Add Layer
            </button>
          </div>
        )}
      </div>

      <div className="control-section architecture-summary" aria-label={`Architecture: 784 to ${config.layers.map(l => l.neurons).join(' to ')} to 10`}>
        <span className="arch-label">Architecture:</span>
        <span className="arch-value">
          784 → {config.layers.map(l => l.neurons).join(' → ')} → 10
        </span>
      </div>
    </div>
  );
}

export default ControlPanel;
