"""
Master Run Script - F1 Monte Carlo Race Simulation
====================================================
Ties together all modules following the Heilmeier et al. (2020) paper:
    Section 3.1: Starting Performance
    Section 3.2a: Lap Time Variability (Fisk per driver)
    Section 3.2b: Pit Stop Variability (Fisk per team)
    Section 3.3: Accident & Failure Probabilities (Bayesian Beta-Bernoulli)
    Section 3.4: Safety Car / VSC Phase Determination
    Section 3.5: Monte Carlo Race Simulation Loop

Usage:
    python run_simulation.py
"""

import sys
import json
import numpy as np
import pandas as pd
import fastf1

from data_pipeline import (
    SEASONS, fetch_and_clean_laps, aggregate_seasons_data,
    driver_stints, starting_performance, normalize_team_name
)
from pit_stop_model import compute_team_pit_variability
from probability_model import compute_probabilities
from fcy_model import compute_fcy_probabilities
from race_simulator import RaceSimulator

fastf1.Cache.enable_cache('../cache')


# ============================================================
# CONFIGURATION
# ============================================================

# Which GPs to use for parameter fitting (broad coverage across the calendar)
FITTING_GPS = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Japan',
    'Miami', 'Spain', 'Great Britain', 'Italy'
]

# Which race to simulate
SIM_YEAR = 2024
SIM_GP = 'Bahrain'

# Monte Carlo iterations
N_ITERATIONS = 500

# Race configuration for Bahrain 2024
# (These values are track-specific; sourced from the paper's approach)
RACE_CONFIG = {
    'total_laps': 57,
    't_base': 93.0,              # Approximate base lap time (qualifying + race pace gap)
    't_lap_sens_mass': 0.030,    # Lap time sensitivity to fuel mass (s/kg)
    'm_fuel_start': 110.0,       # Starting fuel mass (kg)
    'b_fuel_perlap': 1.93,       # Fuel consumption per lap (kg/lap)
    't_pit_stationary': 2.5,     # Minimum pit stop stationary time (s)
    't_pit_inlap': 3.0,          # Pit entry time loss (s)
    't_pit_outlap': 18.0,        # Pit exit time loss (s)
    'mult_t_lap_sc': 1.6,        # SC lap time multiplier
    't_gap_overtake': 1.5,       # Required time gap to overtake (s)
    't_drseffect': -0.4,         # DRS effect on lap time (s)
    'drs_window': 1.0,           # DRS activation window (s)
}


def get_race_entry_list(year: int, gp_name: str) -> list:
    """Fetch the actual race entry list from FastF1 session results.
    
    Returns a list of dicts with driver info and grid positions.
    """
    session = fastf1.get_session(year, gp_name, 'R')
    session.load(telemetry=False, weather=False)

    results = session.results
    entry_list = []

    for _, row in results.iterrows():
        grid_pos = row.get('GridPosition', 20)
        if pd.isna(grid_pos) or grid_pos == 0:
            grid_pos = 20  # Pit lane start

        entry_list.append({
            'abbreviation': row['Abbreviation'],
            'team': normalize_team_name(row['TeamName']),
            'grid_pos': int(grid_pos),
        })

    return entry_list


def compute_car_driver_offsets(years: list, gps: list) -> dict:
    """Compute t_car (team offset) and t_driver (driver within team offset).
    
    Normalizes team performance within each specific race session (Year + Track)
    to prevent track-to-track base lap time variations from skewing the averages.
    """
    all_laps = aggregate_seasons_data(years, gps)
    if all_laps.empty:
        return {'t_car': {}, 't_driver': {}}

    # Group by session to find relative offsets within each individual race
    session_groups = all_laps.groupby(['Year', 'Track'])
    
    team_offsets_list = []
    driver_offsets_list = []
    
    for session_key, session_data in session_groups:
        # Team medians for this specific session
        team_medians = session_data.groupby('Team')['LapTimeSeconds'].median()
        fastest_team_median = team_medians.min()
        session_team_offsets = team_medians - fastest_team_median
        team_offsets_list.append(session_team_offsets)
        
        # Driver medians for this specific session
        driver_medians = session_data.groupby(['Driver', 'Team'])['LapTimeSeconds'].median().reset_index()
        for _, row in driver_medians.iterrows():
            team_med = team_medians.get(row['Team'], fastest_team_median)
            offset = row['LapTimeSeconds'] - team_med
            driver_offsets_list.append({
                'Driver': row['Driver'],
                'Offset': offset
            })
            
    # Combine and average team offsets across all sessions
    team_offsets_df = pd.concat(team_offsets_list, axis=1).mean(axis=1)
    t_car = team_offsets_df.to_dict()
    
    # Combine and average driver offsets across all sessions
    driver_offsets_df = pd.DataFrame(driver_offsets_list)
    if not driver_offsets_df.empty:
        t_driver = driver_offsets_df.groupby('Driver')['Offset'].mean().to_dict()
    else:
        t_driver = {}
    
    return {'t_car': t_car, 't_driver': t_driver}


def estimate_tyre_degradation(years: list, gps: list) -> dict:
    """Estimate tyre degradation parameters using a state-space model approach.
    
    Fits:
      - nu[compound]: degradation drift rate (seconds per lap)
      - sigma_eta: process noise standard deviation (state transition uncertainty)
      - sigma_eps: observation noise standard deviation (lap-by-lap pace variation)
      - skewness: skewnorm shape parameter for asymmetric driver errors / traffic
      
    Returns:
        dict: {
            'degradation_rates': {compound: float},
            'sigma_eta': float,
            'sigma_eps': float,
            'skewness': float
        }
    """
    all_laps = aggregate_seasons_data(years, gps)
    if all_laps.empty:
        return {
            'degradation_rates': {'SOFT': 0.10, 'MEDIUM': 0.06, 'HARD': 0.03},
            'sigma_eta': 0.10,
            'sigma_eps': 0.30,
            'skewness': 1.5
        }

    from scipy.stats import skewnorm

    degradation_rates = {}
    stint_groups = all_laps.groupby(['Year', 'Track', 'Driver', 'Compound', 'Stint'])

    # Fuel mass coefficient (from config, typical is 0.030 s/kg)
    gamma = 0.030
    # Average fuel consumption per lap (approx 1.93 kg/lap)
    fuel_consumption = 1.93
    m_fuel_start = 110.0

    all_residuals = []
    first_diff_residuals = []

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        rates = []
        
        for stint_keys, stint_data in stint_groups:
            _, _, _, comp, _ = stint_keys
            if comp != compound or len(stint_data) < 5:
                continue
            
            # Stint age (t = 0, 1, 2...)
            x = stint_data['TyreLife'].values - stint_data['TyreLife'].min()
            
            # Correct lap times for fuel effect:
            laps = stint_data['LapNumber'].values
            fuel_remaining = np.clip(m_fuel_start - fuel_consumption * (laps - 1), 0, m_fuel_start)
            y_raw = stint_data['LapTimeSeconds'].values
            y_fuel_corrected = y_raw - gamma * fuel_remaining
            
            # Fit linear model for tyre wear on fuel-corrected lap times
            if len(x) >= 3:
                k1, k0 = np.polyfit(x, y_fuel_corrected, 1)
                
                # Exclude outliers or negative rates
                if 0.001 < k1 < 0.8:
                    rates.append(k1)
                    y_pred = k0 + k1 * x
                    residuals = y_fuel_corrected - y_pred
                    all_residuals.extend(residuals)
                    
                    if len(residuals) >= 2:
                        first_diff_residuals.extend(np.diff(residuals))

        if rates:
            degradation_rates[compound] = float(np.median(rates))
        else:
            degradation_rates[compound] = {'SOFT': 0.10, 'MEDIUM': 0.06, 'HARD': 0.03}[compound]

    # Now estimate state space noise parameters from residuals
    sigma_eps = 0.30
    sigma_eta = 0.10
    skew_param = 1.5

    if len(all_residuals) >= 10:
        try:
            # Use skewnorm to fit the residuals to capture the driver error / traffic skew
            a_fit, loc_fit, scale_fit = skewnorm.fit(all_residuals)
            skew_param = float(a_fit)
            sigma_eps = float(scale_fit)
        except Exception:
            pass

    if len(first_diff_residuals) >= 10:
        try:
            # Var(diff_residuals) = Var(eta) + 2 * Var(eps)
            var_diff = np.var(first_diff_residuals)
            cov_matrix = np.cov(first_diff_residuals[:-1], first_diff_residuals[1:])
            cov_val = cov_matrix[0, 1] if cov_matrix.ndim > 1 else 0
            
            # Estimate sigma_eps^2 from -cov_val
            est_var_eps = max(0.01, -cov_val)
            est_var_eta = max(0.001, var_diff - 2 * est_var_eps)
            
            sigma_eps = float(np.sqrt(est_var_eps))
            sigma_eta = float(np.sqrt(est_var_eta))
        except Exception:
            pass

    # Ensure reasonable boundaries for simulation stability
    sigma_eps = np.clip(sigma_eps, 0.15, 0.60)
    sigma_eta = np.clip(sigma_eta, 0.02, 0.25)
    skew_param = np.clip(skew_param, 0.5, 3.0)

    return {
        'degradation_rates': degradation_rates,
        'sigma_eta': float(sigma_eta),
        'sigma_eps': float(sigma_eps),
        'skewness': float(skew_param)
    }


def assign_strategies(entry_list: list) -> list:
    """Assign default 1-stop strategies to all drivers.
    
    This is a simplification - the paper's full version optimizes strategies.
    Here we use a standard Medium -> Hard strategy with pit around lap 20.
    """
    for driver in entry_list:
        # Default 1-stop: start on MEDIUM, pit around lap 18-22, switch to HARD
        pit_lap = np.random.randint(18, 23)
        driver['strategy'] = [
            {'lap': 1, 'compound': 'MEDIUM'},
            {'lap': pit_lap, 'compound': 'HARD'}
        ]
    return entry_list


def run_full_simulation():
    """Run the complete Monte Carlo race simulation pipeline."""
    import pandas as pd

    print("=" * 70)
    print("F1 Monte Carlo Race Simulation")
    print(f"Following Heilmeier et al. (2020)")
    print(f"Seasons: {SEASONS} | Fitting GPs: {FITTING_GPS}")
    print(f"Simulating: {SIM_GP} {SIM_YEAR} | Iterations: {N_ITERATIONS}")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Section 3.1 - Starting Performance
    # --------------------------------------------------------
    print("\n[1/6] Section 3.1 - Fitting Starting Performance...")
    a_avg_fitted, driver_start_params = starting_performance(SEASONS, FITTING_GPS)
    if driver_start_params:
        print(f"  Fitted a_avg = {a_avg_fitted:.3f} m/s^2")
        print(f"  Parameters for {len(driver_start_params)} drivers")
    else:
        print("  WARNING: No starting performance data. Using defaults.")
        a_avg_fitted = 11.2
        driver_start_params = {}

    # --------------------------------------------------------
    # STEP 2: Section 3.2a - Lap Time Variability
    # --------------------------------------------------------
    print("\n[2/6] Section 3.2a - Fitting Lap Time Variability (Fisk)...")
    master_df = aggregate_seasons_data(SEASONS, FITTING_GPS)
    if not master_df.empty:
        stint_results = driver_stints(master_df)
        driver_fisk_params = stint_results['driver_fisk_params']
        track_fisk_params = stint_results['track_fisk_params']
        print(f"  Fisk parameters for {len(driver_fisk_params)} drivers")
    else:
        driver_fisk_params = {}
        track_fisk_params = {}

    # --------------------------------------------------------
    # STEP 3: Section 3.2b - Pit Stop Variability
    # --------------------------------------------------------
    print("\n[3/6] Section 3.2b - Fitting Pit Stop Variability (Fisk)...")
    team_pit_fisk_params = compute_team_pit_variability(SEASONS, FITTING_GPS)
    print(f"  Fisk parameters for {len(team_pit_fisk_params)} teams")

    # --------------------------------------------------------
    # STEP 4: Section 3.3 - Accident & Failure Probabilities
    # --------------------------------------------------------
    print("\n[4/6] Section 3.3 - Computing Accident & Failure Probabilities...")
    prob_result = compute_probabilities(SEASONS, FITTING_GPS)
    p_accident = prob_result.get('p_accident', {})
    p_failure = prob_result.get('p_failure', {})
    print(f"  P_accident for {len(p_accident)} drivers")
    print(f"  P_failure for {len(p_failure)} teams")

    # --------------------------------------------------------
    # STEP 5: Section 3.4 - FCY Phase Determination
    # --------------------------------------------------------
    print("\n[5/6] Section 3.4 - Computing Safety Car Probabilities...")
    fcy_params = compute_fcy_probabilities(SEASONS, FITTING_GPS)
    if fcy_params:
        print(f"  Analyzed {fcy_params['n_races_analyzed']} races")
        print(f"  P(0 SC) = {fcy_params['p_sc_quant'][0]:.2f}, "
              f"P(1 SC) = {fcy_params['p_sc_quant'][1]:.2f}, "
              f"P(2 SC) = {fcy_params['p_sc_quant'][2]:.2f}")

    # --------------------------------------------------------
    # Compute car/driver offsets and tyre degradation
    # --------------------------------------------------------
    print("\n  Computing car/driver performance offsets...")
    offsets = compute_car_driver_offsets(SEASONS, FITTING_GPS)
    t_car = offsets['t_car']
    t_driver = offsets['t_driver']
    print(f"  t_car offsets for {len(t_car)} teams")
    print(f"  t_driver offsets for {len(t_driver)} drivers")

    print("\n  Estimating tyre degradation rates...")
    tyre_deg_data = estimate_tyre_degradation(SEASONS, FITTING_GPS)
    tyre_deg = tyre_deg_data['degradation_rates']
    for compound, rate in tyre_deg.items():
        print(f"  {compound}: {rate:.4f} s/lap")

    # --------------------------------------------------------
    # STEP 6: Section 3.5 - Monte Carlo Simulation
    # --------------------------------------------------------
    print(f"\n[6/6] Section 3.5 - Running Monte Carlo Simulation ({N_ITERATIONS} iterations)...")

    # Assemble simulation parameters
    sim_params = {
        'driver_fisk_params': driver_fisk_params,
        'track_fisk_params': track_fisk_params,
        'driver_start_params': driver_start_params,
        'team_pit_fisk_params': team_pit_fisk_params,
        'p_accident': p_accident,
        'p_failure': p_failure,
        'fcy_params': fcy_params,
        'a_avg_start': a_avg_fitted,
        'tyre_degradation': tyre_deg_data,
    }

    # Get the actual entry list for the target race
    entry_list = get_race_entry_list(SIM_YEAR, SIM_GP)
    
    # Assign car/driver offsets
    for driver in entry_list:
        driver['t_car'] = t_car.get(driver['team'], 0.5)
        driver['t_driver'] = t_driver.get(driver['abbreviation'], 0.0)

    # Assign strategies
    entry_list = assign_strategies(entry_list)

    # Create simulator and run
    simulator = RaceSimulator(RACE_CONFIG, sim_params)
    mc_results = simulator.monte_carlo(entry_list, n_iterations=N_ITERATIONS)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RESULTS: {SIM_GP} {SIM_YEAR} Monte Carlo Simulation")
    print("=" * 70)

    pos_probs = mc_results['position_probabilities']
    ret_counts = mc_results['retirement_counts']

    # Print top-3 probability for each driver, sorted by most likely winner
    print(f"\n{'Driver':<6} {'Team':<25} {'Grid':<5} {'P(P1)':<8} {'P(P2)':<8} {'P(P3)':<8} {'P(Top3)':<8} {'P(Ret)':<8}")
    print("-" * 76)

    # Sort by P(P1) descending
    driver_p1 = [(d, pos_probs[d][1]) for d in pos_probs]
    driver_p1.sort(key=lambda x: -x[1])

    for drv_abbr, _ in driver_p1:
        probs = pos_probs[drv_abbr]
        p_ret = ret_counts[drv_abbr] / N_ITERATIONS
        
        # Find team and grid for display
        team = ''
        grid = 0
        for d in entry_list:
            if d['abbreviation'] == drv_abbr:
                team = d['team']
                grid = d['grid_pos']
                break

        p1 = probs[1] if len(probs) > 1 else 0
        p2 = probs[2] if len(probs) > 2 else 0
        p3 = probs[3] if len(probs) > 3 else 0
        p_top3 = p1 + p2 + p3

        print(f"{drv_abbr:<6} {team:<25} {grid:<5} {p1:<8.1%} {p2:<8.1%} {p3:<8.1%} {p_top3:<8.1%} {p_ret:<8.1%}")

    # Save results to JSON
    output = {
        'race': f"{SIM_GP} {SIM_YEAR}",
        'n_iterations': N_ITERATIONS,
        'position_probabilities': {
            drv: probs.tolist() for drv, probs in pos_probs.items()
        },
        'retirement_rates': {
            drv: count / N_ITERATIONS for drv, count in ret_counts.items()
        },
        'parameters': {
            'a_avg_start': a_avg_fitted,
            'tyre_degradation': tyre_deg,
            'race_config': RACE_CONFIG
        }
    }

    output_path = f'../config/simulation_results_{SIM_GP}_{SIM_YEAR}.json'
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"\nCould not save results: {e}")

    print("\nSimulation complete!")
    return output


if __name__ == "__main__":
    run_full_simulation()
