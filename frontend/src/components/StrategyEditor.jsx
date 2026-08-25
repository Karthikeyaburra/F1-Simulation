import React from 'react';
import { TEAM_COLORS } from './Header';

const COMPOUND_COLORS = {
  SOFT: '#E80020',
  MEDIUM: '#FFD700',
  HARD: '#CCCCCC',
};

export default function StrategyEditor({
  config,
  currentDriver,
  selectedDriver,
  updateDriverConfig,
  updateStintLap,
  updateStintCompound,
  addStint,
  removeStint,
}) {
  if (!currentDriver) return null;
  
  const activeColor = TEAM_COLORS[currentDriver.team] || '#e10600';
  const totalLaps = config.laps || 57;
  
  const formattedCarOffset = Math.round(currentDriver.t_car * 1000) / 1000;
  const formattedDriverOffset = Math.round(currentDriver.t_driver * 1000) / 1000;

  // Calculate stint lengths for the visual timeline
  const stintSegments = currentDriver.strategy.map((stint, idx) => {
    const startLap = stint.lap;
    const endLap = idx < currentDriver.strategy.length - 1
      ? currentDriver.strategy[idx + 1].lap - 1
      : totalLaps;
    return {
      compound: stint.compound,
      startLap,
      endLap,
      length: endLap - startLap + 1,
      pctWidth: ((endLap - startLap + 1) / totalLaps) * 100,
    };
  });

  return (
    <div className="card strategy-card">
      <h3 className="card-title">
        <span className="card-icon">⚙️</span>
        Strategy — <span style={{ color: activeColor }}>{currentDriver.abbreviation}</span>
        <span className="team-label">{currentDriver.team}</span>
      </h3>

      {/* Visual Stint Timeline */}
      <div className="stint-timeline">
        <div className="timeline-bar">
          {stintSegments.map((seg, idx) => (
            <div
              key={idx}
              className="stint-segment"
              style={{
                width: `${seg.pctWidth}%`,
                backgroundColor: COMPOUND_COLORS[seg.compound] || '#555',
                opacity: 0.85,
              }}
            >
              <span className="segment-label">
                {seg.compound[0]} · L{seg.startLap}-{seg.endLap}
              </span>
            </div>
          ))}
        </div>
        <div className="timeline-laps">
          <span>Lap 1</span>
          <span>Lap {totalLaps}</span>
        </div>
      </div>

      {/* Performance Sliders */}
      <div className="editor-grid">
        <div className="slider-control">
          <div className="slider-header">
            <label>Grid Position</label>
            <span className="slider-val">P{currentDriver.grid_pos}</span>
          </div>
          <input
            type="range" min="1" max="20"
            value={currentDriver.grid_pos}
            onChange={(e) => updateDriverConfig(selectedDriver, 'grid_pos', parseInt(e.target.value) || 1)}
            className="premium-slider"
            style={{ '--slider-color': activeColor }}
          />
        </div>

        <div className="slider-control">
          <div className="slider-header">
            <label>Car Offset</label>
            <span className="slider-val">
              {formattedCarOffset > 0 ? `+${formattedCarOffset.toFixed(3)}` : formattedCarOffset.toFixed(3)}s
            </span>
          </div>
          <input
            type="range" min="-0.50" max="2.50" step="0.01"
            value={currentDriver.t_car}
            onChange={(e) => updateDriverConfig(selectedDriver, 't_car', parseFloat(e.target.value) || 0.0)}
            className="premium-slider"
            style={{ '--slider-color': activeColor }}
          />
        </div>

        <div className="slider-control">
          <div className="slider-header">
            <label>Driver Offset</label>
            <span className="slider-val">
              {formattedDriverOffset > 0 ? `+${formattedDriverOffset.toFixed(3)}` : formattedDriverOffset.toFixed(3)}s
            </span>
          </div>
          <input
            type="range" min="-0.50" max="1.50" step="0.01"
            value={currentDriver.t_driver}
            onChange={(e) => updateDriverConfig(selectedDriver, 't_driver', parseFloat(e.target.value) || 0.0)}
            className="premium-slider"
            style={{ '--slider-color': activeColor }}
          />
        </div>
      </div>

      {/* Stint Controls */}
      <div className="stint-controls">
        {currentDriver.strategy.map((stint, idx) => (
          <div key={idx} className="stint-row">
            <div className="stint-badge" style={{ backgroundColor: COMPOUND_COLORS[stint.compound] }}>
              {idx + 1}
            </div>
            <select
              value={stint.compound}
              onChange={(e) => updateStintCompound(selectedDriver, idx, e.target.value)}
              className="stint-select"
            >
              {config.compounds.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            {idx > 0 && (
              <>
                <div className="slider-control compact">
                  <label>Pit Lap</label>
                  <input
                    type="range"
                    min={Math.max(5, (currentDriver.strategy[idx - 1]?.lap || 1) + 3)}
                    max={idx < currentDriver.strategy.length - 1
                      ? (currentDriver.strategy[idx + 1]?.lap || totalLaps) - 3
                      : totalLaps - 3}
                    value={stint.lap}
                    onChange={(e) => updateStintLap(selectedDriver, idx, parseInt(e.target.value))}
                    className="premium-slider compact"
                    style={{ '--slider-color': COMPOUND_COLORS[stint.compound] }}
                  />
                  <span className="mini-val">L{stint.lap}</span>
                </div>
                <button 
                  className="remove-stint-btn"
                  onClick={() => removeStint(selectedDriver, idx)}
                  title="Remove stint"
                >
                  ×
                </button>
              </>
            )}
          </div>
        ))}
        
        {currentDriver.strategy.length < 4 && (
          <button className="add-stint-btn" onClick={() => addStint(selectedDriver)}>
            + Add Pit Stop
          </button>
        )}
      </div>
    </div>
  );
}
