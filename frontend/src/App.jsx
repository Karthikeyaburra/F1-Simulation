import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RefreshCw, TrendingUp } from 'lucide-react';

import Header, { TEAM_COLORS } from './components/Header';
import DriverGrid from './components/DriverGrid';
import StrategyEditor from './components/StrategyEditor';
import SimulationControl from './components/SimulationControl';
import PositionHeatmap from './components/PositionHeatmap';
import Leaderboard from './components/Leaderboard';
import DriverDeepDive from './components/DriverDeepDive';
import TyreDegradation from './components/TyreDegradation';
import RLComparison from './components/RLComparison';
import './App.css';

function App() {
  // State
  const [selectedYear, setSelectedYear] = useState(2024);
  const [selectedGP, setSelectedGP] = useState('Bahrain');
  const [tracks, setTracks] = useState([{ name: 'Bahrain', total_laps: 57 }]);
  const [config, setConfig] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [simResults, setSimResults] = useState(null);
  const [selectedDriver, setSelectedDriver] = useState(null);
  const [loadingConfig, setLoadingConfig] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [nIterations, setNIterations] = useState(500);
  const [progress, setProgress] = useState(null);
  const [rlResult, setRlResult] = useState(null);
  const [rlLoading, setRlLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('heatmap');

  const wsRef = useRef(null);

  // Load available tracks on mount
  useEffect(() => {
    fetch('/api/tracks')
      .then(r => r.json())
      .then(data => {
        if (data.tracks && data.tracks.length > 0) {
          setTracks(data.tracks);
        }
      })
      .catch(() => {}); // Silently fallback to default track list
  }, []);

  // Load config & parameters when year or GP changes
  useEffect(() => {
    setLoadingConfig(true);
    setSimResults(null);
    setRlResult(null);

    Promise.all([
      fetch(`/api/config?year=${selectedYear}&gp_name=${encodeURIComponent(selectedGP)}`).then(r => r.json()),
      fetch(`/api/analytics?year=${selectedYear}`).then(r => r.json()),
    ])
      .then(([configData, analyticsData]) => {
        setConfig(configData);
        setAnalytics(analyticsData);
        if (configData.drivers?.length > 0) {
          if (!configData.drivers.some(d => d.abbreviation === selectedDriver)) {
            setSelectedDriver(configData.drivers[0].abbreviation);
          }
        }
        setLoadingConfig(false);
      })
      .catch(err => {
        console.error('Error loading data:', err);
        setLoadingConfig(false);
      });
  }, [selectedYear, selectedGP]);

  // WebSocket simulation
  const handleRunSimulation = useCallback(() => {
    if (!config) return;
    setSimulating(true);
    setProgress(null);
    setSimResults(null);
    setRlResult(null);

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/simulate`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({
          year: config.year || selectedYear,
          gp_name: selectedGP,
          n_iterations: nIterations,
          drivers: config.drivers,
        }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'progress') {
          setProgress({ iteration: data.iteration, n_iterations: data.n_iterations });
          // Update live results during simulation
          setSimResults({
            position_probabilities: data.position_probabilities,
            retirement_counts: data.retirement_counts,
            n_iterations: data.iteration,
          });
        } else if (data.type === 'complete') {
          setSimResults({
            position_probabilities: data.position_probabilities,
            retirement_counts: data.retirement_counts,
            n_iterations: data.n_iterations,
          });
          setProgress({ iteration: data.n_iterations, n_iterations: data.n_iterations });
          setSimulating(false);
          ws.close();
        } else if (data.type === 'error') {
          console.error('Simulation error:', data.detail);
          setSimulating(false);
          ws.close();
        }
      };

      ws.onerror = () => {
        // Fallback to HTTP POST if WebSocket fails
        console.warn('WebSocket failed, falling back to HTTP...');
        ws.close();
        handleFallbackSimulation();
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    } catch {
      handleFallbackSimulation();
    }
  }, [config, selectedYear, selectedGP, nIterations]);

  // HTTP fallback for simulation
  const handleFallbackSimulation = useCallback(() => {
    fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        year: config?.year || selectedYear,
        gp_name: selectedGP,
        n_iterations: nIterations,
        drivers: config?.drivers || [],
      }),
    })
      .then(r => r.json())
      .then(data => {
        setSimResults(data);
        setSimulating(false);
      })
      .catch(err => {
        console.error('Simulation failed:', err);
        setSimulating(false);
      });
  }, [config, selectedYear, selectedGP, nIterations]);

  // RL optimization
  const handleRunRL = useCallback(() => {
    if (!config || !selectedDriver) return;
    setRlLoading(true);

    fetch('/api/simulate/rl', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        year: config.year || selectedYear,
        gp_name: selectedGP,
        driver_abbreviation: selectedDriver,
        n_training_episodes: 3000,
        drivers: config.drivers,
      }),
    })
      .then(r => r.json())
      .then(data => {
        setRlResult(data);
        setRlLoading(false);
      })
      .catch(err => {
        console.error('RL optimization failed:', err);
        setRlLoading(false);
      });
  }, [config, selectedDriver, selectedYear, selectedGP]);

  // Strategy editing helpers
  const updateDriverConfig = (drvAbbr, field, value) => {
    setConfig(prev => ({
      ...prev,
      drivers: prev.drivers.map(d =>
        d.abbreviation === drvAbbr ? { ...d, [field]: value } : d
      ),
    }));
  };

  const updateStintLap = (drvAbbr, stintIdx, lap) => {
    setConfig(prev => ({
      ...prev,
      drivers: prev.drivers.map(d => {
        if (d.abbreviation !== drvAbbr) return d;
        const updated = [...d.strategy];
        updated[stintIdx] = { ...updated[stintIdx], lap };
        return { ...d, strategy: updated };
      }),
    }));
  };

  const updateStintCompound = (drvAbbr, stintIdx, compound) => {
    setConfig(prev => ({
      ...prev,
      drivers: prev.drivers.map(d => {
        if (d.abbreviation !== drvAbbr) return d;
        const updated = [...d.strategy];
        updated[stintIdx] = { ...updated[stintIdx], compound };
        return { ...d, strategy: updated };
      }),
    }));
  };

  const addStint = (drvAbbr) => {
    setConfig(prev => ({
      ...prev,
      drivers: prev.drivers.map(d => {
        if (d.abbreviation !== drvAbbr) return d;
        const totalLaps = config.laps || 57;
        const lastStint = d.strategy[d.strategy.length - 1];
        const newLap = Math.min(lastStint.lap + 15, totalLaps - 5);
        return {
          ...d,
          strategy: [...d.strategy, { lap: newLap, compound: 'HARD' }],
        };
      }),
    }));
  };

  const removeStint = (drvAbbr, stintIdx) => {
    setConfig(prev => ({
      ...prev,
      drivers: prev.drivers.map(d => {
        if (d.abbreviation !== drvAbbr) return d;
        if (d.strategy.length <= 1) return d;
        return {
          ...d,
          strategy: d.strategy.filter((_, i) => i !== stintIdx),
        };
      }),
    }));
  };

  // Loading screen
  if (loadingConfig || !config) {
    return (
      <div className="loading-screen">
        <motion.div
          initial={{ rotate: 0 }}
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        >
          <RefreshCw size={44} className="loading-icon" />
        </motion.div>
        <p className="loading-text">Initializing F1 Strategy Lab...</p>
        <p className="loading-sub">Loading {selectedYear} season data from FastF1</p>
      </div>
    );
  }

  const currentDriver = config.drivers.find(d => d.abbreviation === selectedDriver) || config.drivers[0];
  const activeColor = TEAM_COLORS[currentDriver?.team] || '#e10600';

  return (
    <div className="app-root">
      <Header
        config={config}
        selectedYear={selectedYear}
        setSelectedYear={setSelectedYear}
        selectedGP={selectedGP}
        setSelectedGP={setSelectedGP}
        tracks={tracks}
      />

      <main className="app-main">
        {/* LEFT COLUMN — Configuration */}
        <section className="config-column">
          <DriverGrid
            drivers={config.drivers}
            selectedDriver={selectedDriver}
            setSelectedDriver={setSelectedDriver}
          />

          <StrategyEditor
            config={config}
            currentDriver={currentDriver}
            selectedDriver={selectedDriver}
            updateDriverConfig={updateDriverConfig}
            updateStintLap={updateStintLap}
            updateStintCompound={updateStintCompound}
            addStint={addStint}
            removeStint={removeStint}
          />

          <SimulationControl
            simulating={simulating}
            onRunSimulation={handleRunSimulation}
            nIterations={nIterations}
            setNIterations={setNIterations}
            progress={progress}
            activeColor={activeColor}
            onRunRL={handleRunRL}
            rlLoading={rlLoading}
          />

          <TyreDegradation analytics={analytics} />
        </section>

        {/* RIGHT COLUMN — Results */}
        <section className="results-column">
          <AnimatePresence mode="wait">
            {simResults ? (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="results-content"
              >
                {/* Tab Navigation */}
                <div className="results-tabs">
                  <button
                    className={`tab-btn ${activeTab === 'heatmap' ? 'active' : ''}`}
                    onClick={() => setActiveTab('heatmap')}
                  >
                    🔥 Heatmap
                  </button>
                  <button
                    className={`tab-btn ${activeTab === 'leaderboard' ? 'active' : ''}`}
                    onClick={() => setActiveTab('leaderboard')}
                  >
                    🏆 Leaderboard
                  </button>
                  <button
                    className={`tab-btn ${activeTab === 'driver' ? 'active' : ''}`}
                    onClick={() => setActiveTab('driver')}
                  >
                    📊 Driver
                  </button>
                  {rlResult && (
                    <button
                      className={`tab-btn ${activeTab === 'rl' ? 'active' : ''}`}
                      onClick={() => setActiveTab('rl')}
                    >
                      ⚡ RL Optimizer
                    </button>
                  )}
                </div>

                {/* Tab Content */}
                {activeTab === 'heatmap' && (
                  <PositionHeatmap
                    simResults={simResults}
                    drivers={config.drivers}
                    selectedDriver={selectedDriver}
                    setSelectedDriver={setSelectedDriver}
                  />
                )}

                {activeTab === 'leaderboard' && (
                  <Leaderboard
                    simResults={simResults}
                    drivers={config.drivers}
                    selectedDriver={selectedDriver}
                    setSelectedDriver={setSelectedDriver}
                    nIterations={simResults.n_iterations || nIterations}
                  />
                )}

                {activeTab === 'driver' && (
                  <DriverDeepDive
                    simResults={simResults}
                    drivers={config.drivers}
                    selectedDriver={selectedDriver}
                    nIterations={simResults.n_iterations || nIterations}
                  />
                )}

                {activeTab === 'rl' && rlResult && (
                  <RLComparison
                    rlResult={rlResult}
                    selectedDriver={selectedDriver}
                  />
                )}
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="empty-state"
              >
                <div className="empty-visual">
                  <TrendingUp size={64} />
                </div>
                <h3>Ready to Simulate</h3>
                <p>
                  Configure driver strategies and performance offsets, then run 
                  the Monte Carlo simulation to generate statistically grounded 
                  finishing position probabilities.
                </p>
                <div className="empty-features">
                  <div className="feature-pill">🎲 500+ Race Simulations</div>
                  <div className="feature-pill">🔥 Position Heatmap</div>
                  <div className="feature-pill">⚡ RL Strategy Optimization</div>
                  <div className="feature-pill">🏎️ Safety Car Modelling</div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </section>
      </main>
    </div>
  );
}

export default App;
