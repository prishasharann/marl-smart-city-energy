import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gymnasium import spaces
from pettingzoo import ParallelEnv

# Loading the processed BDG2 data

PROJ_DIR = Path(r"C:\Users\prish\building-data-genome-project-2")
PROCESSED_DIR = PROJ_DIR / "data" / "processed"
DATA_FILE = PROCESSED_DIR / "bdg2_clean.csv"

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["timestamp"],
    index_col="timestamp"
)

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Normalizing the energy variables

energy_columns = ["building_a", "building_b", "solar"]

for column in energy_columns:
    max_value = df[column].max()
    if max_value > 0:
        df[column] /= max_value

print(df[energy_columns].describe())


def get_electricity_price(hour):
    """
    Return the time-of-use electricity price for an hour.
    """
    if 0 <= hour < 6:
        return 0.08
    elif 6 <= hour < 17:
        return 0.12
    else:
        return 0.20


def get_carbon_intensity(solar):
    """
    Estimate normalized grid carbon intensity from solar availability.
    Higher solar availability corresponds to lower carbon intensity.
    """
    carbon = 1.0 - solar
    return np.clip(carbon, 0.2, 1.0)


AGENTS = [
    "building_a",
    "building_b",
    "battery",
    "grid"
]


# Environment Class - Constructor

class MicrogridEnv(ParallelEnv):
    metadata = {"name": "bdg2_microgrid_v0"}

    # Grid agent's action -> fraction of unmet demand actually imported.
    GRID_IMPORT_FRACTIONS = {
        0: 0.5,   # idle    -> partial import (moderate curtailment)
        1: 1.0,   # import  -> full import (cover all unmet demand)
        2: 0.25,  # curtail -> deep curtailment (save cost/carbon, hurt stability)
    }

    def __init__(self, data, episode_length=24,
                 cost_norm=0.4, emissions_norm=2.0, deviation_norm=None):

        self.data = data.reset_index(drop=False)
        self.episode_length = episode_length
        self.possible_agents = AGENTS.copy()
        self.agents = AGENTS.copy()
        self.episode_start = 0

        # Reward normalization constants 
        self.cost_norm = cost_norm
        self.emissions_norm = emissions_norm
        self.deviation_norm = deviation_norm

        # Battery Parameters
        self.battery_capacity = 1.0
        self.battery_soc = 0.5
        self.max_charge = 0.25
        self.max_discharge = 0.25
        self.charge_efficiency = 0.90
        self.discharge_efficiency = 0.90

        # Current system values
        self.grid_import = 0.0
        self.grid_export = 0.0

        # Action Spaces
        self.action_spaces = {
            # 0 = reduce flexible demand
            # 1 = normal demand
            # 2 = increase flexible demand
            "building_a": spaces.Discrete(3),
            "building_b": spaces.Discrete(3),

            # 0 = idle
            # 1 = charge
            # 2 = discharge
            "battery": spaces.Discrete(3),

            # See GRID_IMPORT_FRACTIONS above:
            # 0 = idle (50% import), 1 = import (100%), 2 = curtail (25%)
            "grid": spaces.Discrete(3)
        }

        # Operator Modes
        self.operator_modes = {
            "economy": 0,
            "sustainability": 1,
            "reliability": 2
        }

        self.operator_mode = 0

        # Current reward weights
        self.weights = np.array(
            [1 / 3, 1 / 3, 1 / 3],
            dtype=np.float32
        )

        LOCAL_OBS_SIZE = 11

        self.observation_spaces = {
            agent: spaces.Box(
                low=np.zeros(LOCAL_OBS_SIZE, dtype=np.float32),
                high=np.ones(LOCAL_OBS_SIZE, dtype=np.float32),
                dtype=np.float32
            )
            for agent in AGENTS
        }

        GLOBAL_STATE_SIZE = 13

        self.global_state_space = spaces.Box(
            low=np.zeros(GLOBAL_STATE_SIZE, dtype=np.float32),
            high=np.ones(GLOBAL_STATE_SIZE, dtype=np.float32),
            dtype=np.float32
        )

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # Building Flexibility

    def _building_demand(self, base_demand, action):
        if action == 0:
            return base_demand * 0.90
        elif action == 1:
            return base_demand
        else:
            return base_demand * 1.10

    # Battery Dynamics

    def _battery_energy(self, action, excess_solar, remaining_demand):
        charge = 0.0
        discharge = 0.0

        if action == 1:
            charge = min(
                self.max_charge, excess_solar,
                (self.battery_capacity - self.battery_soc) / self.charge_efficiency
            )
            self.battery_soc += charge * self.charge_efficiency
        elif action == 2:
            discharge = min(
                self.max_discharge,
                self.battery_soc * self.discharge_efficiency,
                remaining_demand
            )
            self.battery_soc -= discharge / self.discharge_efficiency

        self.battery_soc = np.clip(self.battery_soc, 0.0, self.battery_capacity)

        return charge, discharge

    def _get_observations(self):
        row = self.data.iloc[self.current_step]
        hour = pd.Timestamp(row["timestamp"]).hour
        price = get_electricity_price(hour)
        carbon = get_carbon_intensity(row["solar"])
        grid_load = min(self.grid_import, 1.0)

        mode = np.zeros(3, dtype=np.float32)
        mode[self.operator_mode] = 1.0

        common_state = np.array([
            grid_load,
            price / 0.20,
            row["solar"],
            carbon,
            self.weights[0],
            self.weights[1],
            self.weights[2],
            mode[0],
            mode[1],
            mode[2]
        ], dtype=np.float32)

        observations = {
            "building_a": np.concatenate([[row["building_a"]], common_state]),
            "building_b": np.concatenate([[row["building_b"]], common_state]),
            "battery": np.concatenate([[self.battery_soc], common_state]),
            "grid": np.concatenate([[grid_load], common_state])
        }
        return observations

    def get_global_state(self):
        row = self.data.iloc[self.current_step]
        hour = pd.Timestamp(row["timestamp"]).hour
        price = get_electricity_price(hour)
        carbon = get_carbon_intensity(row["solar"])

        mode = np.zeros(3, dtype=np.float32)
        mode[self.operator_mode] = 1.0

        global_state = np.array([
            row["building_a"],
            row["building_b"],
            self.battery_soc,
            row["solar"],
            min(self.grid_import, 1.0),
            price / 0.20,
            carbon,
            self.weights[0],
            self.weights[1],
            self.weights[2],
            mode[0],
            mode[1],
            mode[2]
        ], dtype=np.float32)

        return global_state

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        max_start = len(self.data) - self.episode_length
        self.episode_start = np.random.randint(0, max_start + 1)
        self.current_step = self.episode_start

        self.agents = self.possible_agents.copy()
        self.battery_soc = 0.5
        self.grid_import = 0.0
        self.grid_export = 0.0
        self.weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        self.operator_mode = 0

        observations = self._get_observations()
        return observations, {}

    def _cost_reward(self, grid_import, price):
        raw_cost = grid_import * price
        normalized_cost = raw_cost / self.cost_norm
        return -np.clip(normalized_cost, 0.0, 1.0)

    def _carbon_reward(self, grid_import, carbon):
        raw_emissions = grid_import * carbon
        normalized_emissions = raw_emissions / self.emissions_norm
        return -np.clip(normalized_emissions, 0.0, 1.0)

    def _get_target_grid_load(self):
        start = self.episode_start
        end = min(self.episode_start + self.episode_length, len(self.data))

        demand = (
            self.data.iloc[start:end]["building_a"]
            + self.data.iloc[start:end]["building_b"]
        )
        return demand.mean()

    def _stability_reward(self, grid_load, target_grid_load):
        deviation = abs(grid_load - target_grid_load)
        denom = self.deviation_norm if self.deviation_norm is not None else max(target_grid_load, 1e-6)
        normalized_deviation = deviation / denom
        return -np.clip(normalized_deviation, 0.0, 1.0)

    def _calculate_reward(self, grid_import, price, carbon, grid_load, target_grid_load):
        r_cost = self._cost_reward(grid_import, price)
        r_carbon = self._carbon_reward(grid_import, carbon)
        r_stability = self._stability_reward(grid_load, target_grid_load)

        reward = (
            self.weights[0] * r_cost
            + self.weights[1] * r_carbon
            + self.weights[2] * r_stability
        )

        components = {
            "cost_reward": r_cost,
            "carbon_reward": r_carbon,
            "stability_reward": r_stability,
            "total_reward": reward
        }
        return reward, components

    def step(self, actions):
        row = self.data.iloc[self.current_step]
        hour = pd.Timestamp(row["timestamp"]).hour
        price = get_electricity_price(hour)
        carbon = get_carbon_intensity(row["solar"])

        # 1. Building actions
        demand_a = self._building_demand(row["building_a"], actions["building_a"])
        demand_b = self._building_demand(row["building_b"], actions["building_b"])
        total_demand = demand_a + demand_b

        # 2. Solar generation
        solar = row["solar"]
        solar_to_buildings = min(solar, total_demand)
        remaining_demand = total_demand - solar_to_buildings
        excess_solar = solar - solar_to_buildings

        # 3. Battery action
        battery_charge, battery_discharge = self._battery_energy(
            actions["battery"], excess_solar, remaining_demand
        )
        excess_solar -= battery_charge
        remaining_demand -= battery_discharge

        # 4. Grid interaction — grid agent controls WHAT FRACTION of unmet
        #    demand gets imported (graduated curtailment lever), rather than
        #    an on/off gate. Export always happens automatically when there's
        #    excess solar, since grid_export never affects reward.
        grid_action = actions["grid"]
        import_fraction = self.GRID_IMPORT_FRACTIONS[grid_action]

        grid_import = remaining_demand * import_fraction if remaining_demand > 0 else 0.0
        grid_export = excess_solar if excess_solar > 0 else 0.0

        # 5. Aggregate grid load
        self.grid_import = grid_import
        self.grid_export = grid_export
        grid_load = grid_import

        # 6. Target flat grid load
        target_grid_load = self._get_target_grid_load()

        # 7. Multi-objective reward
        reward, reward_components = self._calculate_reward(
            grid_import, price, carbon, grid_load, target_grid_load
        )

        # 8. Advance environment
        self.current_step += 1
        terminated = (self.current_step - self.episode_start) >= self.episode_length
        if terminated:
            self.agents = []
            observations = {}
        else:
            observations = self._get_observations()

        # 9. Cooperative rewards
        rewards = {agent: reward for agent in self.possible_agents}
        terminations = {agent: terminated for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}

        # 10. Diagnostic information
        infos = {
            agent: {
                "building_a_demand": demand_a,
                "building_b_demand": demand_b,
                "total_demand": total_demand,
                "solar": solar,
                "battery_soc": self.battery_soc,
                "battery_charge": battery_charge,
                "battery_discharge": battery_discharge,
                "grid_import": grid_import,
                "grid_export": grid_export,
                "grid_load": grid_load,
                "target_grid_load": target_grid_load,
                "price": price,
                "carbon_intensity": carbon,
                **reward_components
            }
            for agent in self.possible_agents
        }

        return observations, rewards, terminations, truncations, infos


def calibrate_reward_norms(env_factory, n_episodes=200):
    """
    Run random-action rollouts and report percentiles of raw_cost,
    raw_emissions, and grid-load deviation, so cost_norm / emissions_norm /
    deviation_norm can be set from real data instead of guessed constants.

    IMPORTANT: raw_cost and raw_emissions are computed from grid_import
    (what the reward function actually uses), not total_demand — re-run
    this any time the grid-interaction logic changes, since that changes
    what grid_import equals for a given state/action.

    Usage:
        stats = calibrate_reward_norms(lambda: MicrogridEnv(data=df, episode_length=24))
        env = MicrogridEnv(data=df, episode_length=24,
                            cost_norm=stats["raw_cost"]["p90"],
                            emissions_norm=stats["raw_emissions"]["p90"])
    """
    raw_costs, raw_emissions, deviations = [], [], []

    for _ in range(n_episodes):
        env = env_factory()
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {a: env.action_space(a).sample() for a in env.possible_agents}
            obs, rewards, term, trunc, infos = env.step(actions)
            info = infos[env.possible_agents[0]]
            raw_costs.append(info["grid_import"] * info["price"])
            raw_emissions.append(info["grid_import"] * info["carbon_intensity"])
            deviations.append(abs(info["grid_load"] - info["target_grid_load"]))
            done = term[env.possible_agents[0]] or trunc[env.possible_agents[0]]

    def summarize(values):
        arr = np.array(values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
        }

    stats = {
        "raw_cost": summarize(raw_costs),
        "raw_emissions": summarize(raw_emissions),
        "deviation": summarize(deviations),
    }

    for name, s in stats.items():
        print(f"{name}: p50={s['p50']:.4f}  p90={s['p90']:.4f}  p99={s['p99']:.4f}  max={s['max']:.4f}")

    return stats