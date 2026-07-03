"""
Section 3.3 - Determination of Accident and Failure Probabilities
==================================================================
Heilmeier et al. (2020), Section 3.3:

Accidents are attributed to the DRIVER, technical failures to the TEAM.
Uses a Bayesian Beta-Bernoulli model:

    Prior:    Beta(alpha_hat, beta_hat)
    Posterior: Beta(alpha_hat + z, beta_hat + N - z)
    
    where z = number of events, N = total races

    alpha_hat = ((1 - mu_hat) / sigma_hat^2 - 1/mu_hat) * mu_hat^2
    beta_hat  = alpha_hat * (1/mu_hat - 1)

The mean of the posterior is used as the final probability:
    P_event = (alpha_hat + z) / (alpha_hat + beta_hat + N)
"""

import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache('../cache')

# Keywords to classify retirement status from FastF1
ACCIDENT_KEYWORDS = [
    'accident', 'collision', 'spun off', 'crash', 'hit',
    'contact', 'barrier', 'off track', 'damage'
]

FAILURE_KEYWORDS = [
    'engine', 'gearbox', 'hydraulics', 'power unit', 'electrical',
    'brakes', 'overheating', 'oil pressure', 'water pressure',
    'fuel system', 'exhaust', 'transmission', 'suspension',
    'mechanical', 'technical', 'pneumatic', 'water leak',
    'oil leak', 'fuel pressure', 'turbo', 'ers', 'mgu'
]

FINISHED_KEYWORDS = [
    'finished', '+1 lap', '+2 laps', '+3 laps', '+4 laps',
    '+5 laps', '+6 laps'
]


def classify_retirement(status: str) -> str:
    """Classify a FastF1 result status into 'accident', 'failure', or 'finished'."""
    if pd.isna(status):
        return 'finished'

    status_lower = status.lower().strip()

    # Check if the driver finished the race
    for kw in FINISHED_KEYWORDS:
        if kw in status_lower:
            return 'finished'

    # Check for accident (driver-attributed)
    for kw in ACCIDENT_KEYWORDS:
        if kw in status_lower:
            return 'accident'

    # Check for technical failure (team-attributed)
    for kw in FAILURE_KEYWORDS:
        if kw in status_lower:
            return 'failure'

    # If status doesn't match known patterns, treat as failure (conservative)
    # This catches things like "Retired", "Did not finish", etc.
    return 'failure'


def fetch_race_results(year: int, gp_name: str) -> pd.DataFrame:
    """Fetch race results with classification of retirement type.
    
    Returns DataFrame with columns: Driver, Team, Status, Classification
    """
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load(telemetry=False, weather=False)

        results = session.results[['Abbreviation', 'TeamName', 'Status']].copy()
        results = results.rename(columns={'Abbreviation': 'Driver', 'TeamName': 'Team'})
        results['Classification'] = results['Status'].apply(classify_retirement)
        results['Year'] = year
        results['GP'] = gp_name

        return results

    except Exception as e:
        print(f"Error fetching results for {gp_name} {year}: {e}")
        return pd.DataFrame()


def build_historical_database(years: list, gps: list) -> dict:
    """Build a historical database of accidents and failures across seasons.
    
    Returns:
        dict with keys:
            'driver_accidents': {driver: {'races': int, 'accidents': int}}
            'team_failures': {team: {'races': int, 'failures': int}}
            'all_results': pd.DataFrame  (raw results for inspection)
    """
    all_results = []

    for year in years:
        for gp in gps:
            df = fetch_race_results(year, gp)
            if not df.empty:
                all_results.append(df)

    if not all_results:
        print("No race results found!")
        return {}

    combined = pd.concat(all_results, ignore_index=True)

    # Per-driver accident stats
    driver_accidents = {}
    for driver, driver_data in combined.groupby('Driver'):
        total_races = len(driver_data)
        accidents = len(driver_data[driver_data['Classification'] == 'accident'])
        driver_accidents[driver] = {'races': total_races, 'accidents': accidents}

    # Per-team failure stats (each driver entry counts as one race for the team)
    team_failures = {}
    for team, team_data in combined.groupby('Team'):
        # Count race entries (each driver in each race = 1 entry)
        total_entries = len(team_data)
        failures = len(team_data[team_data['Classification'] == 'failure'])
        team_failures[team] = {'races': total_entries, 'failures': failures}

    return {
        'driver_accidents': driver_accidents,
        'team_failures': team_failures,
        'all_results': combined
    }


def compute_beta_prior(entity_stats: dict, event_key: str) -> tuple:
    """Compute the Beta prior parameters (alpha_hat, beta_hat) from historical fractions.
    
    Args:
        entity_stats: dict of {entity: {'races': int, event_key: int}}
        event_key: 'accidents' or 'failures'
    
    Returns:
        (alpha_hat, beta_hat) tuple
    """
    fractions = []
    for entity, stats in entity_stats.items():
        if stats['races'] >= 3:  # Minimum races threshold (relaxed from 30 for our smaller dataset)
            fraction = stats[event_key] / stats['races']
            fractions.append(fraction)

    if len(fractions) < 2:
        print(f"Not enough data to compute prior for {event_key}")
        return (1.0, 1.0)  # Uninformative prior

    fractions = np.array(fractions)
    # Replace exact zeros with a small value to avoid division issues
    fractions = np.maximum(fractions, 0.001)

    mu_hat = np.mean(fractions)
    sigma_hat = np.std(fractions)

    if sigma_hat == 0 or mu_hat == 0:
        return (1.0, 1.0)

    # Beta distribution method of moments (Paper's formula)
    alpha_hat = ((1 - mu_hat) / sigma_hat**2 - 1/mu_hat) * mu_hat**2
    beta_hat = alpha_hat * (1/mu_hat - 1)

    # Ensure positive parameters
    alpha_hat = max(alpha_hat, 0.1)
    beta_hat = max(beta_hat, 0.1)

    return (alpha_hat, beta_hat)


def compute_probabilities(years: list, gps: list) -> dict:
    """Compute accident and failure probabilities using Bayesian Beta-Bernoulli update.
    
    Returns:
        dict with keys:
            'p_accident': {driver: float}  (per-race probability)
            'p_failure': {team: float}     (per-race probability)
            'alpha_beta_accident': (alpha_hat, beta_hat)
            'alpha_beta_failure': (alpha_hat, beta_hat)
    """
    db = build_historical_database(years, gps)
    if not db:
        return {}

    driver_accidents = db['driver_accidents']
    team_failures = db['team_failures']

    # Compute priors
    alpha_a, beta_a = compute_beta_prior(driver_accidents, 'accidents')
    alpha_f, beta_f = compute_beta_prior(team_failures, 'failures')

    # Compute posterior means (per-race probabilities)
    p_accident = {}
    for driver, stats in driver_accidents.items():
        z = stats['accidents']
        N = stats['races']
        p_accident[driver] = (alpha_a + z) / (alpha_a + beta_a + N)

    p_failure = {}
    for team, stats in team_failures.items():
        z = stats['failures']
        N = stats['races']
        p_failure[team] = (alpha_f + z) / (alpha_f + beta_f + N)

    return {
        'p_accident': p_accident,
        'p_failure': p_failure,
        'alpha_beta_accident': (alpha_a, beta_a),
        'alpha_beta_failure': (alpha_f, beta_f),
        'database': db
    }


if __name__ == "__main__":
    from data_pipeline import SEASONS

    test_gps = ['Bahrain']
    result = compute_probabilities(SEASONS, test_gps)

    if result:
        print("\n--- Accident Probabilities per Driver (per race) ---")
        alpha_a, beta_a = result['alpha_beta_accident']
        print(f"  Prior: alpha={alpha_a:.3f}, beta={beta_a:.3f}")
        for driver, p in sorted(result['p_accident'].items(), key=lambda x: -x[1]):
            print(f"  {driver}: P_accident = {p:.4f}")

        print("\n--- Failure Probabilities per Team (per race) ---")
        alpha_f, beta_f = result['alpha_beta_failure']
        print(f"  Prior: alpha={alpha_f:.3f}, beta={beta_f:.3f}")
        for team, p in sorted(result['p_failure'].items(), key=lambda x: -x[1]):
            print(f"  {team}: P_failure = {p:.4f}")

        # Show the raw classification data
        print("\n--- Retirement Classifications Sample ---")
        all_results = result['database']['all_results']
        retirements = all_results[all_results['Classification'] != 'finished']
        if not retirements.empty:
            print(retirements[['Year', 'GP', 'Driver', 'Team', 'Status', 'Classification']].to_string(index=False))
