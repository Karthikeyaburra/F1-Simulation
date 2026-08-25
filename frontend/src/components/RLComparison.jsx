import React from 'react';
import { motion } from 'framer-motion';
import { Zap, ArrowUp, ArrowDown, Minus } from 'lucide-react';

const COMPOUND_COLORS = {
  SOFT: '#E80020',
  MEDIUM: '#FFD700',
  HARD: '#CCCCCC',
};

export default function RLComparison({ rlResult, selectedDriver }) {
  if (!rlResult) return null;

  const { rl_strategy, user_strategy, improvement } = rlResult;
  const positionImproved = improvement.position_delta > 0.3;
  const positionWorsened = improvement.position_delta < -0.3;

  return (
    <div className="card rl-card">
      <h3 className="card-title">
        <span className="card-icon"><Zap size={16} /></span>
        RL Strategy Optimizer — <span className="accent">{selectedDriver}</span>
      </h3>

      <div className="rl-comparison">
        {/* User Strategy */}
        <div className="rl-column user-col">
          <h4>Your Strategy</h4>
          <div className="rl-metric">
            <span className="rl-metric-label">Expected Position</span>
            <span className="rl-metric-value">{user_strategy.expected_position}</span>
          </div>
          <div className="rl-metric">
            <span className="rl-metric-label">Expected Points</span>
            <span className="rl-metric-value">{user_strategy.expected_points}</span>
          </div>
        </div>

        {/* Delta Arrow */}
        <div className="rl-delta">
          <motion.div
            className={`delta-badge ${positionImproved ? 'improved' : positionWorsened ? 'worsened' : 'neutral'}`}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            {positionImproved ? <ArrowUp size={20} /> : positionWorsened ? <ArrowDown size={20} /> : <Minus size={20} />}
            <span>{Math.abs(improvement.position_delta).toFixed(1)} pos</span>
          </motion.div>
          <div className={`delta-points ${improvement.points_delta > 0 ? 'positive' : improvement.points_delta < 0 ? 'negative' : ''}`}>
            {improvement.points_delta > 0 ? '+' : ''}{improvement.points_delta.toFixed(1)} pts
          </div>
        </div>

        {/* RL Strategy */}
        <div className="rl-column rl-col">
          <h4>RL Recommended</h4>
          <div className="rl-metric">
            <span className="rl-metric-label">Expected Position</span>
            <span className="rl-metric-value highlight">{rl_strategy.avg_position}</span>
          </div>
          <div className="rl-metric">
            <span className="rl-metric-label">Expected Points</span>
            <span className="rl-metric-value highlight">{rl_strategy.avg_points}</span>
          </div>
          <div className="rl-metric">
            <span className="rl-metric-label">DNF Rate</span>
            <span className="rl-metric-value">{(rl_strategy.retirement_rate * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* RL Pit Stop Strategy */}
      {rl_strategy.pit_stops && rl_strategy.pit_stops.length > 0 && (
        <div className="rl-pit-stops">
          <h4>Recommended Pit Stops</h4>
          <div className="rl-stops-row">
            {rl_strategy.pit_stops.map((stop, idx) => (
              <motion.div
                key={idx}
                className="rl-stop-badge"
                style={{ borderColor: COMPOUND_COLORS[stop.compound] }}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
              >
                <span className="stop-lap">Lap {stop.lap}</span>
                <span
                  className="stop-compound"
                  style={{ color: COMPOUND_COLORS[stop.compound] }}
                >
                  → {stop.compound}
                </span>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      <div className="rl-training-info">
        Trained on {rl_strategy.n_training_episodes} episodes · Evaluated over {rl_strategy.n_evaluation_runs} runs
      </div>
    </div>
  );
}
