"""
Section 3.5 - Monte Carlo Race Simulation Engine
==================================================
Heilmeier et al. (2020), Section 3.5:

Lap-by-lap race simulation that combines all probabilistic effects:
    - Starting performance (Section 3.1)
    - Lap time variability (Section 3.2a)
    - Pit stop variability (Section 3.2b)
    - Accidents and failures (Section 3.3)
    - Safety Car / VSC phases (Section 3.4)

Core lap time equation (Equation 1):
    t_lap(l) = t_base + t_tire(a, c) + t_fuel(l) + t_car + t_driver
             + t_grid(l) + t_pit(l) + t_lap_var
"""

import numpy as np
from scipy.stats import fisk as fisk_dist
from fcy_model import pre_determine_sc_phases


class DriverState:
    """Tracks the state of a single driver during the race."""
    def __init__(self, abbreviation, team, grid_pos, t_car, t_driver,
                 fisk_params, start_perf, strategy):
        self.abbreviation = abbreviation
        self.team = team
        self.grid_pos = grid_pos
        self.t_car = t_car
        self.t_driver = t_driver
        self.fisk_params = fisk_params  # {'c', 'loc', 'scale'} for lap time variability
        self.start_perf = start_perf    # {'mu', 'sigma'} for starting performance
        self.strategy = strategy        # List of {'lap': int, 'compound': str}

        # Race state
        self.total_time = 0.0
        self.lap_times = []
        self.current_compound = strategy[0]['compound'] if strategy else 'MEDIUM'
        self.tyre_age = 0
        self.next_pit_idx = 1  # Index into strategy for next pit stop
        self.retired = False
        self.retired_lap = None
        self.retired_reason = None
        self.position = grid_pos


class RaceSimulator:
    """Monte Carlo race simulator following Heilmeier et al. (2020)."""

    def __init__(self, race_config: dict, sim_params: dict):
        """
        Args:
            race_config: Dict containing race-specific parameters:
                - total_laps: int
                - t_base: float (base lap time = t_q + t_gap_racepace)
                - t_lap_sens_mass: float (lap time sensitivity to fuel mass, s/kg)
                - m_fuel_start: float (starting fuel mass in kg)
                - b_fuel_perlap: float (fuel consumption per lap in kg)
                - t_pit_stationary: float (minimum pit stop stationary time in s)
                - t_pit_inlap: float (time loss entering pit lane in s)
                - t_pit_outlap: float (time loss exiting pit lane in s)
                - mult_t_lap_sc: float (lap time multiplier under SC, ~1.6)
                - t_gap_overtake: float (required gap to overtake in s)
                - t_drseffect: float (DRS lap time reduction in s, negative)
                - drs_window: float (DRS activation window in s)
                
            sim_params: Dict containing fitted probabilistic parameters:
                - driver_fisk_params: {driver: {'c', 'loc', 'scale'}}
                - driver_start_params: {driver: {'mu', 'sigma'}}
                - team_pit_fisk_params: {team: {'c', 'loc', 'scale', 'median_duration'}}
                - p_accident: {driver: float}
                - p_failure: {team: float}
                - fcy_params: dict from compute_fcy_probabilities()
                - a_avg_start: float (fitted start acceleration)
                - tyre_degradation: {compound: float} (seconds per lap degradation)
        """
        self.race_config = race_config
        self.sim_params = sim_params
        self.total_laps = race_config['total_laps']

    def _compute_lap_time(self, driver: DriverState, lap: int, 
                          is_sc: bool, rng: np.random.Generator) -> float:
        """Compute the lap time for a single driver on a single lap."""
        rc = self.race_config
        sp = self.sim_params

        # Base lap time
        t_lap = rc['t_base']

        # Team and driver performance offsets
        t_lap += driver.t_car
        t_lap += driver.t_driver

        # Fuel effect: car gets lighter each lap (linear decrease)
        fuel_remaining = max(0, rc['m_fuel_start'] - rc['b_fuel_perlap'] * (lap - 1))
        t_fuel = rc['t_lap_sens_mass'] * fuel_remaining
        t_lap += t_fuel

        # Tyre degradation: linear model per compound
        tyre_deg_rate = sp.get('tyre_degradation', {}).get(driver.current_compound, 0.05)
        t_tyre = tyre_deg_rate * driver.tyre_age
        t_lap += t_tyre

        # Safety Car multiplier
        if is_sc:
            t_lap *= rc.get('mult_t_lap_sc', 1.6)
        else:
            # Lap time variability (Fisk distribution) - only under green flag
            if driver.fisk_params:
                fp = driver.fisk_params
                t_var = fisk_dist.rvs(fp['c'], loc=fp['loc'], scale=fp['scale'],
                                      random_state=rng)
                # Clamp extreme outliers
                t_var = np.clip(t_var, -3.0, 10.0)
                t_lap += t_var

        return t_lap

    def _compute_start_time(self, driver: DriverState, 
                            rng: np.random.Generator) -> float:
        """Compute Lap 1 additional time (standing start + start performance)."""
        sp = self.sim_params
        a_avg = sp.get('a_avg_start', 11.2)
        
        # Deterministic component: time to reach start line from grid position
        # t_s = sqrt(16 * (p_g - 0.8) / a_avg) + 0.2
        grid_pos = driver.grid_pos
        t_start = np.sqrt(16.0 * max(0.2, grid_pos - 0.8) / a_avg) + 0.2

        # Stochastic component: driver-specific start performance
        if driver.start_perf:
            t_startperf = rng.normal(driver.start_perf['mu'], 
                                     max(0.01, driver.start_perf['sigma']))
            t_start += t_startperf

        return t_start

    def _check_pit_stop(self, driver: DriverState, lap: int, 
                        rng: np.random.Generator) -> float:
        """Check if driver pits this lap and return pit time loss (0 if no pit)."""
        if driver.next_pit_idx >= len(driver.strategy):
            return 0.0

        planned_pit_lap = driver.strategy[driver.next_pit_idx]['lap']
        if lap != planned_pit_lap:
            return 0.0

        rc = self.race_config
        sp = self.sim_params

        # Base pit time
        t_pit = rc['t_pit_inlap'] + rc['t_pit_stationary'] + rc['t_pit_outlap']

        # Add team-specific pit stop variability (Fisk distribution)
        team_pit_params = sp.get('team_pit_fisk_params', {}).get(driver.team, None)
        if team_pit_params:
            t_pit_var = fisk_dist.rvs(team_pit_params['c'],
                                       loc=team_pit_params['loc'],
                                       scale=team_pit_params['scale'],
                                       random_state=rng)
            t_pit_var = np.clip(t_pit_var, -2.0, 15.0)
            t_pit += t_pit_var

        # Switch tyres
        new_compound = driver.strategy[driver.next_pit_idx]['compound']
        driver.current_compound = new_compound
        driver.tyre_age = 0
        driver.next_pit_idx += 1

        return t_pit

    def _check_retirement(self, driver: DriverState, lap: int,
                          rng: np.random.Generator) -> bool:
        """Check if driver retires this lap due to accident or failure."""
        sp = self.sim_params

        # Convert per-race probability to per-lap probability:
        # P_per_lap = 1 - (1 - P_per_race)^(1/total_laps)
        p_accident_race = sp.get('p_accident', {}).get(driver.abbreviation, 0.05)
        p_failure_race = sp.get('p_failure', {}).get(driver.team, 0.05)

        p_accident_lap = 1 - (1 - p_accident_race) ** (1 / self.total_laps)
        p_failure_lap = 1 - (1 - p_failure_race) ** (1 / self.total_laps)

        if rng.random() < p_accident_lap:
            driver.retired = True
            driver.retired_lap = lap
            driver.retired_reason = 'accident'
            return True

        if rng.random() < p_failure_lap:
            driver.retired = True
            driver.retired_lap = lap
            driver.retired_reason = 'failure'
            return True

        return False

    def _update_positions(self, drivers: list):
        """Update race positions based on total elapsed time."""
        # Separate active and retired drivers
        active = [d for d in drivers if not d.retired]
        retired = [d for d in drivers if d.retired]

        # Sort active drivers by total time (lowest = P1)
        active.sort(key=lambda d: d.total_time)
        for i, d in enumerate(active):
            d.position = i + 1

        # Retired drivers get positions after active ones
        retired.sort(key=lambda d: d.retired_lap or 0, reverse=True)
        for i, d in enumerate(retired):
            d.position = len(active) + i + 1

    def simulate_race(self, drivers: list, seed: int = None) -> dict:
        """Simulate a single race.
        
        Args:
            drivers: List of DriverState objects
            seed: Random seed for reproducibility
            
        Returns:
            dict with race results
        """
        rng = np.random.default_rng(seed)
        sp = self.sim_params

        # Pre-determine SC phases (Section 3.4)
        fcy_params = sp.get('fcy_params', None)
        sc_phases = []
        if fcy_params:
            sc_phases = pre_determine_sc_phases(fcy_params, self.total_laps, rng)

        # Build a set of laps under SC for quick lookup
        sc_laps = set()
        for phase in sc_phases:
            for lap in range(phase['start_lap'], 
                           min(phase['start_lap'] + phase['duration'], self.total_laps + 1)):
                sc_laps.add(lap)

        # Lap-by-lap simulation
        for lap in range(1, self.total_laps + 1):
            is_sc = lap in sc_laps

            for driver in drivers:
                if driver.retired:
                    continue

                # Check for retirement
                if self._check_retirement(driver, lap, rng):
                    continue

                # Compute lap time
                t_lap = self._compute_lap_time(driver, lap, is_sc, rng)

                # Lap 1: add standing start time
                if lap == 1:
                    t_start = self._compute_start_time(driver, rng)
                    t_lap += t_start

                # Pit stop
                t_pit = self._check_pit_stop(driver, lap, rng)
                t_lap += t_pit

                # Increment tyre age
                driver.tyre_age += 1

                # Accumulate
                driver.total_time += t_lap
                driver.lap_times.append(t_lap)

            # Update positions after each lap
            self._update_positions(drivers)

        # Compile results
        results = []
        for d in sorted(drivers, key=lambda x: x.position):
            results.append({
                'position': d.position,
                'driver': d.abbreviation,
                'team': d.team,
                'total_time': d.total_time,
                'retired': d.retired,
                'retired_lap': d.retired_lap,
                'retired_reason': d.retired_reason,
                'grid_pos': d.grid_pos
            })

        return {
            'results': results,
            'sc_phases': sc_phases,
            'total_laps': self.total_laps
        }

    def monte_carlo(self, driver_configs: list, n_iterations: int = 1000,
                    base_seed: int = 42) -> dict:
        """Run Monte Carlo simulation over multiple iterations.
        
        Args:
            driver_configs: List of dicts with driver configuration:
                {'abbreviation', 'team', 'grid_pos', 't_car', 't_driver',
                 'strategy': [{'lap': int, 'compound': str}, ...]}
            n_iterations: Number of Monte Carlo iterations
            base_seed: Base random seed
            
        Returns:
            dict with aggregated results
        """
        # Position count matrix: position_counts[driver][position] = count
        position_counts = {dc['abbreviation']: np.zeros(len(driver_configs) + 1)
                          for dc in driver_configs}
        retirement_counts = {dc['abbreviation']: 0 for dc in driver_configs}

        for i in range(n_iterations):
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i + 1}/{n_iterations}")

            # Create fresh driver states for each iteration
            drivers = []
            for dc in driver_configs:
                ds = DriverState(
                    abbreviation=dc['abbreviation'],
                    team=dc['team'],
                    grid_pos=dc['grid_pos'],
                    t_car=dc.get('t_car', 0.0),
                    t_driver=dc.get('t_driver', 0.0),
                    fisk_params=self.sim_params.get('driver_fisk_params', {}).get(
                        dc['abbreviation'], None),
                    start_perf=self.sim_params.get('driver_start_params', {}).get(
                        dc['abbreviation'], None),
                    strategy=dc.get('strategy', [
                        {'lap': 1, 'compound': 'MEDIUM'},
                        {'lap': 20, 'compound': 'HARD'}
                    ])
                )
                drivers.append(ds)

            # Simulate race
            result = self.simulate_race(drivers, seed=base_seed + i)

            # Accumulate position counts
            for r in result['results']:
                drv = r['driver']
                pos = r['position']
                if pos <= len(driver_configs):
                    position_counts[drv][pos] += 1
                if r['retired']:
                    retirement_counts[drv] += 1

        # Convert to probabilities
        position_probs = {}
        for drv, counts in position_counts.items():
            position_probs[drv] = counts / n_iterations

        return {
            'position_probabilities': position_probs,
            'retirement_counts': retirement_counts,
            'n_iterations': n_iterations
        }
