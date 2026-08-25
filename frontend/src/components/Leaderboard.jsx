import React from 'react';
import { motion } from 'framer-motion';
import { TEAM_COLORS } from './Header';

export default function Leaderboard({ simResults, drivers, selectedDriver, setSelectedDriver, nIterations }) {
  if (!simResults) return null;

  const leaderboardData = Object.keys(simResults.position_probabilities)
    .map(drv => {
      const probs = simResults.position_probabilities[drv];
      const pWin = probs[1] || 0;
      const pPodium = (probs[1] || 0) + (probs[2] || 0) + (probs[3] || 0);
      const pPoints = Array.from({ length: 10 }, (_, i) => probs[i + 1] || 0).reduce((a, b) => a + b, 0);
      const pRetire = (simResults.retirement_counts[drv] || 0) / nIterations;
      const dObj = drivers.find(d => d.abbreviation === drv);

      let expFinish = 0;
      probs.forEach((p, idx) => { expFinish += idx * p; });

      // Median position
      let cumulative = 0;
      let median = 10;
      for (let i = 1; i < probs.length; i++) {
        cumulative += probs[i] || 0;
        if (cumulative >= 0.5) { median = i; break; }
      }

      return {
        name: drv,
        team: dObj?.team || '',
        grid: dObj?.grid_pos || 0,
        winProb: pWin,
        podiumProb: pPodium,
        pointsProb: pPoints,
        retireProb: pRetire,
        expFinish,
        median,
      };
    })
    .sort((a, b) => a.expFinish - b.expFinish);

  return (
    <div className="card leaderboard-card">
      <h3 className="card-title">
        <span className="card-icon">🏆</span>
        Predicted Finishing Order
        <span className="card-badge">{nIterations} simulations</span>
      </h3>

      <div className="lb-table-wrap">
        <table className="lb-table">
          <thead>
            <tr>
              <th className="pos-col">#</th>
              <th>Driver</th>
              <th>Team</th>
              <th className="num-col">Grid</th>
              <th className="num-col">Exp.</th>
              <th className="num-col">Med.</th>
              <th className="bar-col">P(Win)</th>
              <th className="num-col">P(Podium)</th>
              <th className="num-col">P(Points)</th>
              <th className="num-col retire-col">P(DNF)</th>
            </tr>
          </thead>
          <tbody>
            {leaderboardData.map((d, index) => {
              const isSelected = selectedDriver === d.name;
              const teamColor = TEAM_COLORS[d.team] || '#888';
              const winBarWidth = Math.max(2, d.winProb * 100);

              return (
                <motion.tr
                  key={d.name}
                  className={`lb-row ${isSelected ? 'selected' : ''}`}
                  onClick={() => setSelectedDriver(d.name)}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.03 }}
                >
                  <td className="pos-col">
                    <span className={`pos-badge ${index < 3 ? `top-${index + 1}` : ''}`}>
                      {index + 1}
                    </span>
                  </td>
                  <td className="drv-col">
                    <span className="drv-indicator" style={{ backgroundColor: teamColor }} />
                    <span className="drv-name" style={{ color: isSelected ? teamColor : '' }}>
                      {d.name}
                    </span>
                  </td>
                  <td className="team-col">{d.team}</td>
                  <td className="num-col mono">P{d.grid}</td>
                  <td className="num-col mono highlight-orange">{d.expFinish.toFixed(1)}</td>
                  <td className="num-col mono">P{d.median}</td>
                  <td className="bar-col">
                    <div className="win-bar-container">
                      <div
                        className="win-bar"
                        style={{
                          width: `${winBarWidth}%`,
                          backgroundColor: d.winProb > 0.1 ? '#00cc44' : '#444',
                        }}
                      />
                      <span className="win-bar-val">{(d.winProb * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="num-col mono">{(d.podiumProb * 100).toFixed(1)}%</td>
                  <td className="num-col mono">{(d.pointsProb * 100).toFixed(1)}%</td>
                  <td className="num-col mono retire-col">{(d.retireProb * 100).toFixed(1)}%</td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
