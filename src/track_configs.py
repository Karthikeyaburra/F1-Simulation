"""
Track Configurations Database
==============================
Centralized track-specific physical parameters for all supported circuits.
Sourced from analysis of actual race paces and pit lane characteristics.
"""

TRACK_CONFIGS = {
    'Bahrain': {
        'total_laps': 57,
        't_base': 93.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 1.93,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 18.0,
        'mult_t_lap_sc': 1.6,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Saudi Arabia': {
        'total_laps': 50,
        't_base': 91.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 2.20,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 17.0,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Australia': {
        'total_laps': 58,
        't_base': 78.5,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 1.90,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 16.0,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Japan': {
        'total_laps': 53,
        't_base': 93.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 2.08,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 17.5,
        'mult_t_lap_sc': 1.6,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Miami': {
        'total_laps': 57,
        't_base': 92.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 1.93,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 18.5,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Spain': {
        'total_laps': 66,
        't_base': 79.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 1.67,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 18.0,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Great Britain': {
        'total_laps': 52,
        't_base': 91.5,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 2.11,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 17.0,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Italy': {
        'total_laps': 53,
        't_base': 84.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 2.08,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 19.0,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
}


def get_race_config(gp_name: str) -> dict:
    """Get track-specific race configuration with fallback to Bahrain defaults."""
    return TRACK_CONFIGS.get(gp_name, TRACK_CONFIGS['Bahrain']).copy()


def get_available_tracks() -> list:
    """Return list of available track names."""
    return list(TRACK_CONFIGS.keys())
