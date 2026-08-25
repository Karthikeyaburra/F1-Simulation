"""
FastAPI Backend Application — Production
=========================================
Full-featured API server for the F1 Monte Carlo Strategy Simulator.

Endpoints:
    GET  /api/config          — Race entry list and driver configurations
    GET  /api/analytics       — Fitted parameter distributions for charts
    GET  /api/tracks          — Available tracks with their configs
    POST /api/simulate        — Run Monte Carlo simulation (batch)
    WS   /ws/simulate         — WebSocket streaming Monte Carlo simulation
    POST /api/simulate/rl     — Run RL-optimized strategy comparison
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
import os
import asyncio

from data_pipeline import SEASONS, starting_performance, aggregate_seasons_data, driver_stints
from pit_stop_model import compute_team_pit_variability
from probability_model import compute_probabilities
from fcy_model import compute_fcy_probabilities
from race_simulator import RaceSimulator, DriverState
from run_simulation import get_race_entry_list, compute_car_driver_offsets, estimate_tyre_degradation, assign_strategies
from track_configs import TRACK_CONFIGS, get_race_config, get_available_tracks

app = FastAPI(
    title="F1 Monte Carlo Strategy Simulator API",
    description="Production-grade API for F1 race strategy optimization using Monte Carlo simulation and Reinforcement Learning",
    version="2.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================
# Pydantic Models
# ====================

class StintSelection(BaseModel):
    lap: int
    compound: str


class DriverStrategyConfig(BaseModel):
    abbreviation: str
    team: str
    grid_pos: int
    t_car: float
    t_driver: float
    strategy: list[StintSelection]


class SimulationRequest(BaseModel):
    year: int
    gp_name: str
    n_iterations: int = 500
    drivers: list[DriverStrategyConfig]


class RLRequest(BaseModel):
    year: int
    gp_name: str
    driver_abbreviation: str
    n_training_episodes: int = 3000
    drivers: list[DriverStrategyConfig]


# ====================
# Global Caches
# ====================

FITTED_PARAMS = {}
PRE_RACE_CONFIG = {}


def migrate_tyre_degradation_params(tyre_deg: dict) -> dict:
    """Helper to convert old/flat skewt_params to new state-space noise params on the fly."""
    if not isinstance(tyre_deg, dict):
        tyre_deg = {}
        
    # If it is in the simple flat format (e.g., {'SOFT': 0.12, ...})
    if 'SOFT' in tyre_deg and 'degradation_rates' not in tyre_deg:
        tyre_deg = {
            'degradation_rates': tyre_deg,
            'sigma_eta': 0.10,
            'sigma_eps': 0.30,
            'skewness': 1.5
        }
        
    if 'skewt_params' in tyre_deg and 'sigma_eta' not in tyre_deg:
        skewt = tyre_deg['skewt_params']
        scales = [v['scale'] for v in skewt.values() if isinstance(v, dict) and 'scale' in v]
        sigma_eps = float(np.mean(scales)) if scales else 0.30
        
        sigma_eta = 0.10
        
        ncs = [v['nc'] for v in skewt.values() if isinstance(v, dict) and 'nc' in v]
        skewness = float(np.mean(ncs)) if ncs else 1.5
        
        sigma_eps = np.clip(sigma_eps, 0.15, 0.60)
        sigma_eta = np.clip(sigma_eta, 0.02, 0.25)
        skewness = np.clip(skewness, 0.5, 3.0)
        
        return {
            'degradation_rates': tyre_deg.get('degradation_rates', {'SOFT': 0.10, 'MEDIUM': 0.06, 'HARD': 0.03}),
            'sigma_eta': float(sigma_eta),
            'sigma_eps': float(sigma_eps),
            'skewness': float(skewness)
        }
    return tyre_deg


# ====================
# Startup Data Loading
# ====================

for yr in SEASONS:
    params_file = f'fitted_parameters_{yr}.json'
    config_file = f'race_config_bahrain_{yr}.json'

    if os.path.exists(params_file):
        print(f"Startup: Loading pre-computed parameters for {yr} from {params_file}...")
        try:
            with open(params_file, 'r') as f:
                data = json.load(f)
            
            needs_save = False

            # Migrate tyre degradation parameters on-the-fly
            if 'tyre_degradation' in data:
                old_deg = data['tyre_degradation']
                new_deg = migrate_tyre_degradation_params(old_deg)
                if old_deg != new_deg:
                    data['tyre_degradation'] = new_deg
                    needs_save = True
                    print(f"Startup: Migrating tyre degradation parameters for {yr}...")

            # Migrate accident & failure probabilities to fix prior dominance on-the-fly
            if not data.get("probabilities_migrated_v2", False):
                print(f"Startup: Recomputing DNF probabilities for {yr} to fix prior-dominance...")
                try:
                    from probability_model import compute_probabilities
                    fitting_gps = [
                        'Bahrain', 'Saudi Arabia', 'Australia', 'Japan',
                        'Miami', 'Spain', 'Great Britain', 'Italy'
                    ]
                    prob_result = compute_probabilities(SEASONS, fitting_gps)
                    if prob_result:
                        data['p_accident'] = prob_result.get('p_accident', {})
                        data['p_failure'] = prob_result.get('p_failure', {})
                        data['probabilities_migrated_v2'] = True
                        needs_save = True
                        print(f"Startup: Recomputed DNF probabilities successfully.")
                except Exception as e_prob:
                    print(f"Startup: Failed to recompute DNF probabilities for {yr}: {e_prob}")

            if needs_save:
                try:
                    with open(params_file, 'w') as f_out:
                        json.dump(data, f_out, indent=2)
                    print(f"Startup: Successfully saved updated parameters to {params_file}.")
                except Exception as e_write:
                    print(f"Startup: Could not write back updated parameters to {params_file}: {e_write}")
            
            FITTED_PARAMS[yr] = data
        except Exception as e:
            print(f"Error loading {params_file} on startup: {e}")
    else:
        print(f"WARNING: {params_file} not found. Run generate_assets.py to pre-compute.")

    if os.path.exists(config_file):
        print(f"Startup: Loading pre-computed race config for {yr} from {config_file}...")
        try:
            with open(config_file, 'r') as f:
                PRE_RACE_CONFIG[yr] = json.load(f)
        except Exception as e:
            print(f"Error loading {config_file} on startup: {e}")
    else:
        print(f"WARNING: {config_file} not found. Run generate_assets.py to pre-compute.")


def load_all_fitted_parameters(year: int):
    """Returns pre-loaded static parameters for the target year or falls back to dynamic fitting."""
    global FITTED_PARAMS
    if year in FITTED_PARAMS:
        return FITTED_PARAMS[year]

    print(f"Asset missing: fitting parameters dynamically for {year} (fallback)...")
    fitting_gps = [
        'Bahrain', 'Saudi Arabia', 'Australia', 'Japan',
        'Miami', 'Spain', 'Great Britain', 'Italy'
    ]

    # Fits
    a_avg_fitted, driver_start = starting_performance([year], fitting_gps)
    master_df = aggregate_seasons_data([year], fitting_gps)
    
    driver_fisk = {}
    track_fisk = {}
    if not master_df.empty:
        stint_results = driver_stints(master_df)
        driver_fisk = stint_results['driver_fisk_params']
        track_fisk = stint_results['track_fisk_params']

    team_pit_fisk = compute_team_pit_variability([year], fitting_gps)
    prob_result = compute_probabilities([year], fitting_gps)
    fcy = compute_fcy_probabilities([year], fitting_gps)
    offsets = compute_car_driver_offsets([year], fitting_gps)
    tyre_deg = estimate_tyre_degradation([year], fitting_gps)

    FITTED_PARAMS[year] = {
        'a_avg_start': a_avg_fitted or 11.2,
        'driver_start_params': driver_start or {},
        'driver_fisk_params': driver_fisk,
        'track_fisk_params': track_fisk,
        'team_pit_fisk_params': team_pit_fisk,
        'p_accident': prob_result.get('p_accident', {}),
        'p_failure': prob_result.get('p_failure', {}),
        'fcy_params': fcy,
        't_car': offsets['t_car'],
        't_driver': offsets['t_driver'],
        'tyre_degradation': tyre_deg
    }
    return FITTED_PARAMS[year]


# ====================
# API Endpoints
# ====================

@app.get("/api/tracks")
async def get_tracks():
    """Return available tracks with their configurations."""
    tracks = []
    for name, config in TRACK_CONFIGS.items():
        tracks.append({
            "name": name,
            "total_laps": config['total_laps'],
            "t_base": config['t_base'],
        })
    return {"tracks": tracks}


@app.get("/api/config")
async def get_config(year: int = 2024, gp_name: str = "Bahrain"):
    """Get the initial entry list and configurations for the race."""
    try:
        track_config = get_race_config(gp_name)
        
        # Serve immediately from memory if available for target year + Bahrain baseline
        if year in PRE_RACE_CONFIG:
            stored = PRE_RACE_CONFIG[year]
            return {
                "year": stored["year"],
                "gp_name": gp_name,
                "laps": track_config['total_laps'],
                "drivers": stored["drivers"],
                "compounds": ["SOFT", "MEDIUM", "HARD"]
            }

        # Fallback to dynamic loading
        print(f"Asset missing: dynamically loading race config for {year}...")
        params = load_all_fitted_parameters(year)
        entries = get_race_entry_list(year, gp_name)
        entries = assign_strategies(entries, track_config['total_laps'])
        for driver in entries:
            driver['t_car'] = params['t_car'].get(driver['team'], 0.5)
            driver['t_driver'] = params['t_driver'].get(driver['abbreviation'], 0.0)

        return {
            "year": year,
            "gp_name": gp_name,
            "laps": track_config['total_laps'],
            "drivers": entries,
            "compounds": ["SOFT", "MEDIUM", "HARD"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics")
async def get_analytics(year: int = 2024):
    """Retrieve the pre-computed parameter distributions for analysis plots."""
    try:
        params = load_all_fitted_parameters(year)
        return {
            "a_avg_start": params['a_avg_start'],
            "tyre_degradation": params['tyre_degradation'],
            "driver_start_params": params['driver_start_params'],
            "driver_fisk_params": params['driver_fisk_params'],
            "team_pit_fisk_params": params['team_pit_fisk_params'],
            "fcy_params": params.get('fcy_params', {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate")
async def simulate(req: SimulationRequest):
    """Execute the Monte Carlo simulation (batch mode)."""
    try:
        params = load_all_fitted_parameters(req.year)
        
        # Map frontend payload back into simulation format
        driver_configs = []
        for d in req.drivers:
            driver_configs.append({
                'abbreviation': d.abbreviation,
                'team': d.team,
                'grid_pos': d.grid_pos,
                't_car': d.t_car,
                't_driver': d.t_driver,
                'strategy': [{'lap': s.lap, 'compound': s.compound} for s in d.strategy]
            })

        # Run simulator with track-specific config
        track_config = get_race_config(req.gp_name)
        track_config['track_name'] = req.gp_name
        simulator = RaceSimulator(track_config, params)
        mc_results = simulator.monte_carlo(driver_configs, n_iterations=req.n_iterations)

        # Convert numpy arrays to standard lists for JSON output
        pos_probs = {}
        for drv, probs in mc_results['position_probabilities'].items():
            pos_probs[drv] = probs.tolist()

        return {
            "position_probabilities": pos_probs,
            "retirement_counts": mc_results['retirement_counts'],
            "n_iterations": req.n_iterations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/simulate")
async def websocket_simulate(websocket: WebSocket):
    """WebSocket endpoint for streaming Monte Carlo simulation progress.
    
    Client sends JSON: {year, gp_name, n_iterations, drivers: [...]}
    Server sends JSON updates every batch of iterations:
        {type: "progress", iteration, n_iterations, position_probabilities, retirement_counts}
        {type: "complete", position_probabilities, retirement_counts, n_iterations}
    """
    await websocket.accept()
    
    try:
        # Receive simulation request
        data = await websocket.receive_json()
        year = data.get('year', 2024)
        gp_name = data.get('gp_name', 'Bahrain')
        n_iterations = data.get('n_iterations', 500)
        drivers_data = data.get('drivers', [])
        
        params = load_all_fitted_parameters(year)
        
        driver_configs = []
        for d in drivers_data:
            driver_configs.append({
                'abbreviation': d['abbreviation'],
                'team': d['team'],
                'grid_pos': d['grid_pos'],
                't_car': d['t_car'],
                't_driver': d['t_driver'],
                'strategy': d['strategy']
            })
        
        track_config = get_race_config(gp_name)
        track_config['track_name'] = gp_name
        simulator = RaceSimulator(track_config, params)
        
        n_drivers = len(driver_configs)
        position_counts = {dc['abbreviation']: np.zeros(n_drivers + 1) for dc in driver_configs}
        retirement_counts = {dc['abbreviation']: 0 for dc in driver_configs}
        
        batch_size = max(10, n_iterations // 20)  # Send ~20 progress updates
        
        for i in range(n_iterations):
            # Create fresh driver states
            drivers = []
            for dc in driver_configs:
                ds = DriverState(
                    abbreviation=dc['abbreviation'],
                    team=dc['team'],
                    grid_pos=dc['grid_pos'],
                    t_car=dc.get('t_car', 0.0),
                    t_driver=dc.get('t_driver', 0.0),
                    fisk_params=params.get('driver_fisk_params', {}).get(dc['abbreviation'], None),
                    start_perf=params.get('driver_start_params', {}).get(dc['abbreviation'], None),
                    strategy=dc.get('strategy', [
                        {'lap': 1, 'compound': 'MEDIUM'},
                        {'lap': 20, 'compound': 'HARD'}
                    ])
                )
                drivers.append(ds)
            
            result = simulator.simulate_race(drivers, seed=42 + i)
            
            for r in result['results']:
                drv = r['driver']
                pos = r['position']
                if pos <= n_drivers:
                    position_counts[drv][pos] += 1
                if r['retired']:
                    retirement_counts[drv] += 1
            
            # Send progress update
            if (i + 1) % batch_size == 0 or i == n_iterations - 1:
                completed = i + 1
                pos_probs = {}
                for drv, counts in position_counts.items():
                    pos_probs[drv] = (counts / completed).tolist()
                
                msg_type = "complete" if completed == n_iterations else "progress"
                
                await websocket.send_json({
                    "type": msg_type,
                    "iteration": completed,
                    "n_iterations": n_iterations,
                    "position_probabilities": pos_probs,
                    "retirement_counts": {k: v for k, v in retirement_counts.items()},
                })
                
                # Yield control to allow WebSocket messages to be sent
                await asyncio.sleep(0)
    
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except:
            pass


@app.post("/api/simulate/rl")
async def simulate_rl(req: RLRequest):
    """Train an RL agent and return its recommended strategy vs the user's strategy."""
    try:
        from rl_environment import F1StrategyEnv, train_agent
        
        params = load_all_fitted_parameters(req.year)
        track_config = get_race_config(req.gp_name)
        track_config['track_name'] = req.gp_name
        
        driver_configs = []
        agent_idx = 0
        for i, d in enumerate(req.drivers):
            dc = {
                'abbreviation': d.abbreviation,
                'team': d.team,
                'grid_pos': d.grid_pos,
                't_car': d.t_car,
                't_driver': d.t_driver,
                'strategy': [{'lap': s.lap, 'compound': s.compound} for s in d.strategy]
            }
            if d.abbreviation == req.driver_abbreviation:
                agent_idx = i
            driver_configs.append(dc)
        
        # Create RL environment
        env = F1StrategyEnv(
            race_config=track_config,
            sim_params=params,
            driver_configs=driver_configs,
            agent_driver_idx=agent_idx,
        )
        
        # Train the agent (limited episodes for API response time)
        n_episodes = min(req.n_training_episodes, 5000)
        trained_agent = train_agent(env, n_episodes=n_episodes, verbose=False)
        
        # Get RL-recommended strategy (average over 20 episodes)
        rl_results = []
        for _ in range(20):
            result = trained_agent.get_recommended_strategy(env)
            rl_results.append(result)
        
        # Aggregate RL results
        avg_position = np.mean([r['final_position'] for r in rl_results])
        avg_points = np.mean([r['points'] for r in rl_results])
        retirement_rate = np.mean([1 if r['retired'] else 0 for r in rl_results])
        
        # Find most common strategy
        from collections import Counter
        strategy_strs = [str(r['strategy']) for r in rl_results]
        most_common_strategy = Counter(strategy_strs).most_common(1)[0]
        best_strategy_idx = strategy_strs.index(most_common_strategy[0])
        recommended_strategy = rl_results[best_strategy_idx]['strategy']
        
        # Also run user's strategy through Monte Carlo for comparison
        simulator = RaceSimulator(track_config, params)
        mc_results = simulator.monte_carlo(driver_configs, n_iterations=100)
        user_probs = mc_results['position_probabilities'].get(req.driver_abbreviation, np.zeros(21))
        user_expected = sum(i * p for i, p in enumerate(user_probs))
        user_points = sum(F1_POINTS.get(i, 0) * p for i, p in enumerate(user_probs))
        
        F1_POINTS_LOCAL = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        user_points = sum(F1_POINTS_LOCAL.get(i, 0) * float(p) for i, p in enumerate(user_probs))
        
        return {
            "rl_strategy": {
                "pit_stops": recommended_strategy,
                "avg_position": round(avg_position, 1),
                "avg_points": round(avg_points, 1),
                "retirement_rate": round(retirement_rate, 3),
                "n_training_episodes": n_episodes,
                "n_evaluation_runs": 20,
            },
            "user_strategy": {
                "expected_position": round(float(user_expected), 1),
                "expected_points": round(float(user_points), 1),
            },
            "improvement": {
                "position_delta": round(float(user_expected) - avg_position, 1),
                "points_delta": round(avg_points - float(user_points), 1),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi.responses import FileResponse
import glob

@app.get("/api/assets/{filename}")
async def get_asset(filename: str):
    """Serve local assets (e.g. generated images) to the frontend."""
    asset_dir = os.path.dirname(os.path.abspath(__file__))
    # Search for files starting with the base name (handles the dynamic timestamp suffix)
    search_pattern = os.path.join(asset_dir, f"{filename}*")
    matched_files = glob.glob(search_pattern)
    if matched_files:
        return FileResponse(matched_files[0])
    raise HTTPException(status_code=404, detail=f"Asset '{filename}' not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
