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
from scipy.stats import skewnorm
from fcy_model import pre_determine_sc_phases
from data_pipeline import get_track_type


class DriverState:
    """Tracks the state of a single driver during the race."""
    def __init__(self, abbreviation, team, grid_pos, t_car, t_driver,
                 fisk_params, start_perf, strategy):
        self.abbreviation = abbreviation
        self.team = team
        self.grid_pos = grid_pos
        self.t_car = t_car
        self.t_driver = t_driver
        self.fisk_params = fisk_params  # Driver-specific Fisk dictionary (per track)
        self.start_perf = start_perf    # {'mu', 'sigma'} for starting performance
        self.strategy = strategy        # List of {'lap': int, 'compound': str}

        # Race state
        self.total_time = 0.0
        self.lap_times = []
        self.current_compound = strategy[0]['compound'] if strategy else 'MEDIUM'
        self.tyre_age = 0
        self.alpha = 0.0  # Latent tyre wear state (alpha_t in the state-space model)
        self.next_pit_idx = 1  # Index into strategy for next pit stop
        self.retired = False
        self.retired_lap = None
        self.retired_reason = None
        self.position = grid_pos


class RaceSimulator:
    """Monte Carlo race simulator incorporating advanced track-specific parameters."""

    def __init__(self, race_config: dict, sim_params: dict):
        self.race_config = race_config
        self.sim_params = sim_params
        self.total_laps = race_config['total_laps']
        self.track_name = race_config.get('track_name', 'Bahrain')

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

        # Fuel effect: delta from mid-race fuel load as reference
        fuel_remaining = max(0, rc['m_fuel_start'] - rc['b_fuel_perlap'] * (lap - 1))
        fuel_at_midrace = rc['m_fuel_start'] - rc['b_fuel_perlap'] * (self.total_laps / 2)
        t_fuel = rc['t_lap_sens_mass'] * (fuel_remaining - fuel_at_midrace)
        t_lap += t_fuel

        # Tyre degradation: state-space model base wear + process noise
        tyre_degradation = sp.get('tyre_degradation', {})
        degradation_rates = tyre_degradation.get('degradation_rates', {})
        
        if driver.tyre_age == 0:
            driver.alpha = 0.0
        else:
            sigma_eta = tyre_degradation.get('sigma_eta', 0.10)
            eta = rng.normal(0, sigma_eta)
            nu = degradation_rates.get(driver.current_compound, 0.05)
            driver.alpha += nu + eta

        t_tyre = driver.alpha
        t_lap += t_tyre

        # Safety Car multiplier
        if is_sc:
            t_lap *= rc.get('mult_t_lap_sc', 1.6)
        else:
            # Lap time variability (asymmetric observation noise from paper)
            skewness = tyre_degradation.get('skewness', 1.5)
            sigma_eps = tyre_degradation.get('sigma_eps', 0.30)
            
            t_var = skewnorm.rvs(skewness, loc=0, scale=sigma_eps, random_state=rng)
            t_var = np.clip(t_var, -1.0, 5.0)
            t_lap += t_var

        return t_lap

    def _compute_start_time(self, driver: DriverState, 
                            rng: np.random.Generator) -> float:
        """Compute Lap 1 additional time (standing start + start performance)."""
        sp = self.sim_params
        a_avg = sp.get('a_avg_start', 11.2)
        
        grid_pos = driver.grid_pos
        t_start = np.sqrt(16.0 * max(0.2, grid_pos - 0.8) / a_avg) + 0.2

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

        # Base pit time from track-specific team statistics or config fallback
        team_pit_params = sp.get('team_pit_fisk_params', {}).get(driver.team, None)
        if team_pit_params:
            t_pit = team_pit_params['medians'].get(self.track_name, team_pit_params['global_median'])
            
            # Add crew variability
            t_pit_var = fisk_dist.rvs(team_pit_params['c'],
                                       loc=team_pit_params['loc'],
                                       scale=team_pit_params['scale'],
                                       random_state=rng)
            t_pit_var = np.clip(t_pit_var, -2.0, 15.0)
            t_pit += t_pit_var
        else:
            t_pit = rc['t_pit_inlap'] + rc['t_pit_stationary'] + rc['t_pit_outlap']

        # Switch tyres
        new_compound = driver.strategy[driver.next_pit_idx]['compound']
        driver.current_compound = new_compound
        driver.tyre_age = 0
        driver.alpha = 0.0  # Reset tyre wear state space state
        driver.next_pit_idx += 1

        return t_pit

    def _check_retirement(self, driver: DriverState, lap: int,
                          rng: np.random.Generator) -> bool:
        """Check if driver retires this lap due to accident or failure."""
        sp = self.sim_params
        track_type = get_track_type(self.track_name)

        # Retrieve track-type specific probabilities or global fallback
        driver_acc_dict = sp.get('p_accident', {}).get(driver.abbreviation, {})
        p_accident_race = driver_acc_dict.get(track_type, driver_acc_dict.get('All', 0.05))

        team_fail_dict = sp.get('p_failure', {}).get(driver.team, {})
        p_failure_race = team_fail_dict.get(track_type, team_fail_dict.get('All', 0.05))

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
