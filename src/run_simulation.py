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
    driver_stints, starting_performance
)
from pit_stop_model import compute_team_pit_variability
from probability_model import compute_probabilities
from fcy_model import compute_fcy_probabilities
from race_simulator import RaceSimulator

fastf1.Cache.enable_cache('../cache')


# ============================================================
# CONFIGURATION
# ============================================================

# Which GPs to use for parameter fitting
FITTING_GPS = ['Bahrain']

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
            'team': row['TeamName'],
            'grid_pos': int(grid_pos),
        })

    return entry_list


def compute_car_driver_offsets(years: list, gps: list) -> dict:
    """Compute t_car (team offset) and t_driver (driver within team offset).
    
    Uses median clean lap times to determine relative performance.
    """
    all_laps = aggregate_seasons_data(years, gps)
    if all_laps.empty:
        return {'t_car': {}, 't_driver': {}}

    # Compute median lap time per team
    team_medians = all_laps.groupby('Team')['LapTimeSeconds'].median()
    fastest_team_median = team_medians.min()
    t_car = (team_medians - fastest_team_median).to_dict()

    # Compute driver offset within their team
    driver_medians = all_laps.groupby(['Driver', 'Team'])['LapTimeSeconds'].median().reset_index()
    driver_medians.columns = ['Driver', 'Team', 'MedianLap']

    t_driver = {}
    for _, row in driver_medians.iterrows():
        team_median = team_medians.get(row['Team'], fastest_team_median)
        t_driver[row['Driver']] = row['MedianLap'] - team_median

    return {'t_car': t_car, 't_driver': t_driver}


def estimate_tyre_degradation(years: list, gps: list) -> dict:
    """Estimate tyre degradation rates per compound from stint data.
    
    Uses the k1 coefficient from the quadratic polynomial fit as a 
    proxy for linear degradation rate.
    """
    all_laps = aggregate_seasons_data(years, gps)
    if all_laps.empty:
        return {'SOFT': 0.10, 'MEDIUM': 0.06, 'HARD': 0.03}

    compound_rates = {}
    stint_groups = all_laps.groupby(['Year', 'Track', 'Driver', 'Compound', 'Stint'])

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        rates = []
        for stint_keys, stint_data in stint_groups:
            _, _, _, comp, _ = stint_keys
            if comp != compound or len(stint_data) < 5:
                continue
            x = stint_data['TyreLife'].values
            y = stint_data['LapTimeSeconds'].values
            k2, k1, k0 = np.polyfit(x, y, 2)
            # k1 represents the linear degradation rate (s/lap)
            if 0 < k1 < 1.0:  # Filter unreasonable values
                rates.append(k1)

        if rates:
            compound_rates[compound] = np.median(rates)
        else:
            # Fallback defaults
            compound_rates[compound] = {'SOFT': 0.10, 'MEDIUM': 0.06, 'HARD': 0.03}[compound]

    return compound_rates


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
        driver_fisk_params, _ = driver_stints(master_df)
        print(f"  Fisk parameters for {len(driver_fisk_params)} drivers")
    else:
        driver_fisk_params = {}

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
    tyre_deg = estimate_tyre_degradation(SEASONS, FITTING_GPS)
    for compound, rate in tyre_deg.items():
        print(f"  {compound}: {rate:.4f} s/lap")

    # --------------------------------------------------------
    # STEP 6: Section 3.5 - Monte Carlo Simulation
    # --------------------------------------------------------
    print(f"\n[6/6] Section 3.5 - Running Monte Carlo Simulation ({N_ITERATIONS} iterations)...")

    # Assemble simulation parameters
    sim_params = {
        'driver_fisk_params': driver_fisk_params,
        'driver_start_params': driver_start_params,
        'team_pit_fisk_params': team_pit_fisk_params,
        'p_accident': p_accident,
        'p_failure': p_failure,
        'fcy_params': fcy_params,
        'a_avg_start': a_avg_fitted,
        'tyre_degradation': tyre_deg,
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
