import React from 'react';
import { motion } from 'framer-motion';
import { Play, RefreshCw, Zap } from 'lucide-react';

export default function SimulationControl({
  simulating,
  onRunSimulation,
  nIterations,
  setNIterations,
  progress,
  activeColor,
  onRunRL,
  rlLoading,
}) {
  const progressPct = progress ? (progress.iteration / progress.n_iterations) * 100 : 0;

  return (
    <div className="card sim-control-card">
      <h3 className="card-title">
        <span className="card-icon">🎯</span>
        Simulation Control
      </h3>

      <div className="sim-controls-grid">
        <div className="sim-top-row">
          <div className="iter-selector">
            <label>Monte Carlo Iterations</label>
            <select value={nIterations} onChange={(e) => setNIterations(parseInt(e.target.value))}>
              <option value={100}>100 — Quick Preview</option>
              <option value={500}>500 — Balanced</option>
              <option value={1000}>1,000 — High Fidelity</option>
              <option value={2000}>2,000 — Research Grade</option>
            </select>
          </div>

          <motion.button
            className={`run-btn ${simulating ? 'running' : ''}`}
            disabled={simulating}
            onClick={onRunSimulation}
            style={{ '--btn-color': simulating ? '#333' : activeColor }}
            whileHover={simulating ? {} : { scale: 1.02 }}
            whileTap={simulating ? {} : { scale: 0.98 }}
          >
            {simulating ? (
              <RefreshCw className="spin-icon" size={18} />
            ) : (
              <Play size={18} />
            )}
            <span>{simulating ? 'Simulating...' : 'RUN MONTE CARLO'}</span>
          </motion.button>
        </div>

        <motion.button
          className={`rl-btn ${rlLoading ? 'running' : ''}`}
          disabled={rlLoading || simulating}
          onClick={onRunRL}
          whileHover={rlLoading ? {} : { scale: 1.02 }}
          whileTap={rlLoading ? {} : { scale: 0.98 }}
        >
          {rlLoading ? (
            <RefreshCw className="spin-icon" size={16} />
          ) : (
            <Zap size={16} />
          )}
          <span>{rlLoading ? 'Training RL Agent...' : 'RUN RL STRATEGY OPTIMIZER'}</span>
        </motion.button>
      </div>

      {/* Progress Bar */}
      {simulating && progress && (
        <div className="progress-container">
          <div className="progress-bar-bg">
            <motion.div
              className="progress-bar-fill"
              style={{ backgroundColor: activeColor }}
              initial={{ width: 0 }}
              animate={{ width: `${progressPct}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <div className="progress-label">
            <span>{progress.iteration} / {progress.n_iterations} iterations</span>
            <span>{Math.round(progressPct)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}
