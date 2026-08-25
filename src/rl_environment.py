"""
Reinforcement Learning Environment for F1 Race Strategy Optimization
=====================================================================
Custom Gymnasium-compatible environment that wraps the Monte Carlo race
simulator to train RL agents on pit-stop timing and tyre compound selection.

State Space (Box, 12-dimensional continuous):
    [0] current_lap / total_laps          — Race progress (normalized 0-1)
    [1] tyre_age / total_laps             — Current tyre age (normalized 0-1)
    [2] compound_soft                     — One-hot: is current compound SOFT
    [3] compound_medium                   — One-hot: is current compound MEDIUM
    [4] compound_hard                     — One-hot: is current compound HARD
    [5] fuel_remaining / fuel_start       — Fuel level (normalized 0-1)
    [6] position / n_drivers              — Current position (normalized 0-1)
    [7] gap_to_leader / 120.0             — Time gap to P1 (normalized, clipped)
    [8] is_safety_car                     — Binary: is current lap under SC
    [9] tyre_wear_state / 5.0             — Latent tyre wear alpha (normalized)
    [10] has_pitted                       — Binary: has completed at least 1 pit stop
    [11] n_pit_stops / 3.0                — Number of pit stops completed (normalized)

Action Space (Discrete, 4):
    0 = Continue on current tyres (no pit)
    1 = Pit stop → switch to SOFT
    2 = Pit stop → switch to MEDIUM
    3 = Pit stop → switch to HARD

Reward Model:
    - Terminal reward: F1 points mapping (P1=25, P2=18, ..., P10=1, else 0)
    - Intermediate shaping: small reward/penalty for position changes each lap
    - Penalty for pitting on the same compound already mounted
    - Penalty for pitting before lap 5 of a stint (too early)
    - Bonus for meeting minimum compound diversity rule (must use ≥2 compounds)
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

from race_simulator import RaceSimulator, DriverState
from fcy_model import pre_determine_sc_phases
from data_pipeline import get_track_type


# F1 Points system (2024 rules)
F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

COMPOUND_MAP = {0: 'SOFT', 1: 'MEDIUM', 2: 'HARD'}
COMPOUND_INDEX = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}


class F1StrategyEnv:
    """
    Gymnasium-compatible RL environment for F1 pit-stop strategy optimization.
    
    The agent controls a SINGLE driver's pit strategy. All other drivers follow
    fixed baseline strategies. Each step advances the race by one lap.
    
    Compatible with Gymnasium API: reset(), step(), observation_space, action_space.
    """
    
    # Class-level spaces for Gymnasium compatibility
    metadata = {'render_modes': []}
    
    def __init__(self, race_config: dict = None, sim_params: dict = None,
                 driver_configs: list = None, agent_driver_idx: int = 0,
                 seed: int = None):
        """
        Args:
            race_config: Track-specific race configuration dict
            sim_params: Fitted simulation parameters dict
            driver_configs: List of driver config dicts (abbreviation, team, grid_pos, etc.)
            agent_driver_idx: Index of the agent-controlled driver in driver_configs
            seed: Random seed for reproducibility
        """
        # Default Bahrain 2024 config if none provided
        self.race_config = race_config or {
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
        }
        self.sim_params = sim_params or {}
        self.driver_configs = driver_configs or []
        self.agent_driver_idx = agent_driver_idx
        self.total_laps = self.race_config['total_laps']
        self.n_drivers = max(len(self.driver_configs), 1)
        
        # Define observation and action spaces
        self.observation_space_shape = (12,)
        self.action_space_n = 4
        
        if HAS_GYMNASIUM:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(12,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(4)
        
        # Internal state
        self._rng = np.random.default_rng(seed)
        self._simulator = RaceSimulator(self.race_config, self.sim_params)
        self._current_lap = 0
        self._drivers = []
        self._sc_laps = set()
        self._agent_compounds_used = set()
        self._agent_n_pit_stops = 0
        self._done = False
        self._prev_position = 0
        
    def _build_observation(self) -> np.ndarray:
        """Construct the 12-dimensional observation vector for the agent's driver."""
        agent = self._drivers[self.agent_driver_idx]
        
        # Compute gap to leader
        leader_time = min(
            (d.total_time for d in self._drivers if not d.retired),
            default=agent.total_time
        )
        gap_to_leader = agent.total_time - leader_time if not agent.retired else 120.0
        
        obs = np.array([
            self._current_lap / self.total_laps,                        # [0] race progress
            agent.tyre_age / self.total_laps,                           # [1] tyre age
            1.0 if agent.current_compound == 'SOFT' else 0.0,           # [2] compound one-hot
            1.0 if agent.current_compound == 'MEDIUM' else 0.0,         # [3]
            1.0 if agent.current_compound == 'HARD' else 0.0,           # [4]
            max(0, self.race_config['m_fuel_start'] -                   # [5] fuel remaining
                self.race_config['b_fuel_perlap'] * (self._current_lap - 1)
            ) / self.race_config['m_fuel_start'],
            agent.position / self.n_drivers,                            # [6] position
            min(gap_to_leader / 120.0, 1.0),                           # [7] gap to leader
            1.0 if self._current_lap in self._sc_laps else 0.0,        # [8] safety car
            min(agent.alpha / 5.0, 1.0),                                # [9] tyre wear
            1.0 if self._agent_n_pit_stops > 0 else 0.0,               # [10] has pitted
            min(self._agent_n_pit_stops / 3.0, 1.0),                   # [11] pit count
        ], dtype=np.float32)
        
        return obs
    
    def _discretize_state(self, obs: np.ndarray) -> tuple:
        """Discretize continuous state into bins for tabular Q-learning.
        
        Bins each dimension into a small number of buckets for tractable Q-table.
        """
        bins = [
            5,   # race progress (5 quintiles)
            4,   # tyre age
            2, 2, 2,  # compound one-hot (binary)
            3,   # fuel remaining
            4,   # position
            3,   # gap to leader
            2,   # safety car
            3,   # tyre wear
            2,   # has pitted
            3,   # pit count
        ]
        
        state_tuple = []
        for i, n_bins in enumerate(bins):
            val = np.clip(obs[i], 0.0, 0.999)
            bucket = int(val * n_bins)
            state_tuple.append(bucket)
        
        return tuple(state_tuple)
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode.
        
        Returns:
            observation: Initial state observation
            info: Additional information dict
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        self._current_lap = 1
        self._done = False
        self._agent_n_pit_stops = 0
        self._agent_compounds_used = set()
        
        # Create fresh driver states
        self._drivers = []
        for i, dc in enumerate(self.driver_configs):
            # For non-agent drivers, use their configured strategy
            strategy = dc.get('strategy', [
                {'lap': 1, 'compound': 'MEDIUM'},
                {'lap': 20, 'compound': 'HARD'}
            ])
            
            if i == self.agent_driver_idx:
                # Agent starts with a dummy strategy — agent will decide via actions
                strategy = [{'lap': 1, 'compound': dc.get('start_compound', 'MEDIUM')}]
            
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
                strategy=strategy
            )
            self._drivers.append(ds)
        
        # Record starting compound for the agent
        agent = self._drivers[self.agent_driver_idx]
        self._agent_compounds_used.add(agent.current_compound)
        self._prev_position = agent.grid_pos
        
        # Pre-determine SC phases
        fcy_params = self.sim_params.get('fcy_params', None)
        self._sc_laps = set()
        if fcy_params:
            sc_phases = pre_determine_sc_phases(fcy_params, self.total_laps, self._rng)
            for phase in sc_phases:
                for lap in range(phase['start_lap'],
                               min(phase['start_lap'] + phase['duration'], self.total_laps + 1)):
                    self._sc_laps.add(lap)
        
        obs = self._build_observation()
        info = {
            'lap': self._current_lap,
            'driver': agent.abbreviation,
            'position': agent.position,
            'compound': agent.current_compound,
        }
        
        return obs, info
    
    def step(self, action: int):
        """Advance the race by one lap with the given action.
        
        Args:
            action: 0=continue, 1=pit-SOFT, 2=pit-MEDIUM, 3=pit-HARD
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self._done:
            obs = self._build_observation()
            return obs, 0.0, True, False, {}
        
        agent = self._drivers[self.agent_driver_idx]
        lap = self._current_lap
        is_sc = lap in self._sc_laps
        reward = 0.0
        
        # Process agent action (pit stop decision)
        pit_time = 0.0
        if action in [1, 2, 3] and not agent.retired:
            new_compound = COMPOUND_MAP[action - 1]
            
            # Penalty: pitting to same compound
            if new_compound == agent.current_compound:
                reward -= 2.0
            
            # Penalty: pitting too early in a stint (less than 5 laps)
            if agent.tyre_age < 5 and self._agent_n_pit_stops > 0:
                reward -= 1.5
            
            # Execute pit stop
            rc = self.race_config
            team_pit_params = self.sim_params.get('team_pit_fisk_params', {}).get(agent.team, None)
            if team_pit_params:
                from scipy.stats import fisk as fisk_dist
                pit_time = team_pit_params['medians'].get(
                    rc.get('track_name', 'Bahrain'),
                    team_pit_params['global_median']
                )
                t_pit_var = fisk_dist.rvs(
                    team_pit_params['c'],
                    loc=team_pit_params['loc'],
                    scale=team_pit_params['scale'],
                    random_state=self._rng
                )
                pit_time += np.clip(t_pit_var, -2.0, 15.0)
            else:
                pit_time = rc['t_pit_inlap'] + rc['t_pit_stationary'] + rc['t_pit_outlap']
            
            agent.current_compound = new_compound
            agent.tyre_age = 0
            agent.alpha = 0.0
            self._agent_n_pit_stops += 1
            self._agent_compounds_used.add(new_compound)
        
        # Simulate this lap for ALL drivers
        for i, driver in enumerate(self._drivers):
            if driver.retired:
                continue
            
            # Check retirement
            if self._simulator._check_retirement(driver, lap, self._rng):
                continue
            
            # Compute lap time
            t_lap = self._simulator._compute_lap_time(driver, lap, is_sc, self._rng)
            
            # Lap 1: add standing start
            if lap == 1:
                t_start = self._simulator._compute_start_time(driver, self._rng)
                t_lap += t_start
            
            # Pit stop for non-agent drivers (use their fixed strategy)
            if i != self.agent_driver_idx:
                t_pit = self._simulator._check_pit_stop(driver, lap, self._rng)
                t_lap += t_pit
            else:
                # Agent pit time computed above
                t_lap += pit_time
            
            driver.tyre_age += 1
            driver.total_time += t_lap
            driver.lap_times.append(t_lap)
        
        # Update positions
        self._simulator._update_positions(self._drivers)
        
        # Intermediate reward: position change shaping
        new_position = agent.position
        position_change = self._prev_position - new_position  # positive = gained positions
        reward += position_change * 0.3  # Small shaping reward
        self._prev_position = new_position
        
        # Check termination
        self._current_lap += 1
        terminated = (self._current_lap > self.total_laps) or agent.retired
        truncated = False
        
        if terminated:
            self._done = True
            
            if agent.retired:
                reward -= 10.0  # Penalty for retirement (though agent can't control this)
            else:
                # Terminal reward: F1 points
                final_position = agent.position
                points = F1_POINTS.get(final_position, 0)
                reward += points
                
                # Bonus for compound diversity (F1 rule: must use ≥2 compounds)
                if len(self._agent_compounds_used) >= 2:
                    reward += 3.0
                else:
                    reward -= 5.0  # Penalty for not meeting minimum compound rule
                
                # Small bonus for efficiency (fewer unnecessary stops)
                if self._agent_n_pit_stops <= 2:
                    reward += 1.0
        
        obs = self._build_observation()
        info = {
            'lap': self._current_lap - 1,
            'position': agent.position,
            'compound': agent.current_compound,
            'tyre_age': agent.tyre_age,
            'n_pit_stops': self._agent_n_pit_stops,
            'compounds_used': list(self._agent_compounds_used),
            'retired': agent.retired,
            'total_time': agent.total_time,
        }
        
        return obs, reward, terminated, truncated, info
    
    def get_strategy_summary(self) -> dict:
        """Return a summary of the agent's strategy decisions during the episode."""
        agent = self._drivers[self.agent_driver_idx]
        return {
            'driver': agent.abbreviation,
            'team': agent.team,
            'final_position': agent.position,
            'total_time': agent.total_time,
            'n_pit_stops': self._agent_n_pit_stops,
            'compounds_used': list(self._agent_compounds_used),
            'retired': agent.retired,
            'points': F1_POINTS.get(agent.position, 0) if not agent.retired else 0,
        }


class QLearningAgent:
    """
    Tabular Q-learning agent for F1 pit-stop strategy optimization.
    
    Uses discretized state space with epsilon-greedy exploration.
    Learns a policy mapping (discretized_state → action) that optimizes
    cumulative F1 points over a race.
    """
    
    def __init__(self, n_actions: int = 4, learning_rate: float = 0.1,
                 discount_factor: float = 0.99, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.05):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: maps state_tuple → array of action values
        self.q_table = {}
        
        # Training metrics
        self.episode_rewards = []
        self.episode_positions = []
    
    def _get_q_values(self, state: tuple) -> np.ndarray:
        """Get or initialize Q-values for a state."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]
    
    def select_action(self, state: tuple, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))
    
    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, done: bool):
        """Update Q-value using temporal difference learning."""
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(next_q_values)
        
        q_values[action] += self.lr * (target - q_values[action])
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_policy(self) -> dict:
        """Extract the learned policy as a serializable dict."""
        policy = {}
        for state, q_values in self.q_table.items():
            policy[str(state)] = {
                'best_action': int(np.argmax(q_values)),
                'q_values': q_values.tolist(),
                'action_name': ['CONTINUE', 'PIT_SOFT', 'PIT_MEDIUM', 'PIT_HARD'][int(np.argmax(q_values))]
            }
        return policy
    
    def get_recommended_strategy(self, env: F1StrategyEnv) -> dict:
        """Run one episode with the learned policy (no exploration) to get strategy.
        
        Returns the sequence of pit stop decisions and final outcome.
        """
        obs, info = env.reset()
        state = env._discretize_state(obs)
        
        total_reward = 0
        pit_stops = []
        lap_actions = []
        
        done = False
        while not done:
            action = self.select_action(state, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            lap_actions.append({
                'lap': info.get('lap', 0),
                'action': ['CONTINUE', 'PIT_SOFT', 'PIT_MEDIUM', 'PIT_HARD'][action],
                'position': info.get('position', 0),
                'compound': info.get('compound', ''),
                'tyre_age': info.get('tyre_age', 0),
            })
            
            if action > 0:
                pit_stops.append({
                    'lap': info.get('lap', 0),
                    'compound': COMPOUND_MAP[action - 1],
                })
            
            state = env._discretize_state(obs)
        
        summary = env.get_strategy_summary()
        
        return {
            'strategy': pit_stops,
            'final_position': summary['final_position'],
            'total_reward': total_reward,
            'points': summary['points'],
            'compounds_used': summary['compounds_used'],
            'n_pit_stops': summary['n_pit_stops'],
            'retired': summary['retired'],
        }


def train_agent(env: F1StrategyEnv, n_episodes: int = 5000,
                verbose: bool = True) -> QLearningAgent:
    """Train a Q-learning agent on the F1 strategy environment.
    
    Args:
        env: F1StrategyEnv instance
        n_episodes: Number of training episodes
        verbose: Print progress every 500 episodes
        
    Returns:
        Trained QLearningAgent
    """
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.15,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,
        epsilon_min=0.05
    )
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        state = env._discretize_state(obs)
        
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = env._discretize_state(next_obs)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        agent.episode_rewards.append(total_reward)
        agent.episode_positions.append(info.get('position', 20))
        agent.decay_epsilon()
        
        if verbose and (episode + 1) % 500 == 0:
            recent_rewards = agent.episode_rewards[-100:]
            recent_positions = agent.episode_positions[-100:]
            avg_reward = np.mean(recent_rewards)
            avg_pos = np.mean(recent_positions)
            print(f"  Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Avg Position: {avg_pos:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Q-table size: {len(agent.q_table)}")
    
    return agent


if __name__ == "__main__":
    import json
    
    # Load pre-computed parameters
    params_path = 'fitted_parameters_2024.json'
    config_path = 'race_config_bahrain_2024.json'
    
    try:
        with open(params_path, 'r') as f:
            sim_params = json.load(f)
        with open(config_path, 'r') as f:
            race_config_data = json.load(f)
    except FileNotFoundError:
        print("Pre-computed parameter files not found. Run generate_assets.py first.")
        exit(1)
    
    race_config = {
        'total_laps': 57, 't_base': 93.0, 't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0, 'b_fuel_perlap': 1.93, 't_pit_stationary': 2.5,
        't_pit_inlap': 3.0, 't_pit_outlap': 18.0, 'mult_t_lap_sc': 1.6,
        't_gap_overtake': 1.5, 't_drseffect': -0.4, 'drs_window': 1.0,
        'track_name': 'Bahrain',
    }
    
    driver_configs = race_config_data.get('drivers', [])
    
    if not driver_configs:
        print("No driver configs found.")
        exit(1)
    
    # Create environment with VER as the agent (index 0 = pole position)
    env = F1StrategyEnv(
        race_config=race_config,
        sim_params=sim_params,
        driver_configs=driver_configs,
        agent_driver_idx=0,
    )
    
    print("=" * 60)
    print("F1 Strategy RL Training")
    print(f"Agent: {driver_configs[0]['abbreviation']} ({driver_configs[0]['team']})")
    print(f"Track: Bahrain | Laps: 57")
    print("=" * 60)
    
    # Train
    trained_agent = train_agent(env, n_episodes=5000, verbose=True)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating Learned Policy (10 episodes)...")
    positions = []
    for i in range(10):
        result = trained_agent.get_recommended_strategy(env)
        positions.append(result['final_position'])
        if i < 3:
            print(f"  Run {i+1}: P{result['final_position']} | "
                  f"Stops: {result['n_pit_stops']} | "
                  f"Compounds: {result['compounds_used']} | "
                  f"Pts: {result['points']}")
    
    print(f"\nAvg Final Position: {np.mean(positions):.1f}")
    print(f"Best: P{min(positions)} | Worst: P{max(positions)}")
    
    # Save policy
    policy = trained_agent.get_policy()
    policy_path = 'rl_policy_bahrain_2024.json'
    with open(policy_path, 'w') as f:
        json.dump(policy, f, indent=2)
    print(f"\nPolicy saved to {policy_path} ({len(policy)} state entries)")
