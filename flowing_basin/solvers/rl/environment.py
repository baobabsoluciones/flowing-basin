from flowing_basin.core import Instance, Configuration
from flowing_basin.tools import RiverBasin
from cornflow_client.core.tools import load_json
import numpy as np
import gym
import pandas as pd
from datetime import datetime
import pickle
from random import randint
from dataclasses import dataclass


@dataclass(kw_only=True)
class RLConfiguration(Configuration):

    num_prices: int
    num_incoming_flows: int
    num_unreg_flows: int
    length_episodes: int

    # RiverBasin simulator options
    flow_smoothing: int = 0
    mode: str = "nonlinear"

    def __post_init__(self):
        valid_modes = {"linear", "nonlinear"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid value for 'mode': {self.mode}. Allowed values are {valid_modes}")


class Environment(gym.Env):

    """
    Class representing the environment for the RL agent
    The class acts as a wrapper of the river basin:
     - It filters and normalizes the river basin's states, turning them into observations
     - It computes the rewards for the agent (proportional to the generated energy and its price)
    """

    def __init__(
        self,
        instance: Instance,
        config: RLConfiguration,
        path_constants: str,
        path_training_data: str,
        paths_power_models: dict[str, str] = None,
    ):

        super(Environment, self).__init__()

        self.config = config

        # Files required to create randeom instances
        self.constants = load_json(path_constants)
        self.training_data = pd.read_pickle(path_training_data)

        # Instance and river basin
        self.instance = instance
        self.river_basin = RiverBasin(
            instance=self.instance, mode=self.config.mode, paths_power_models=paths_power_models
        )

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                self.config.num_prices + self.config.num_incoming_flows + sum(
                    1 + self.instance.get_relevant_lags_of_dam(dam_id)[-1] + self.config.num_unreg_flows
                    for dam_id in self.instance.get_ids_of_dams()
                ),
            ),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.instance.get_num_dams(), ),
            dtype=np.float32
        )

        # Observation limits (for state normalization)
        self.obs_low = self.get_obs_lower_limits()
        self.obs_high = self.get_obs_upper_limits()
        self.max_flows = np.array([
            self.instance.get_max_flow_of_channel(dam_id) for dam_id in self.instance.get_ids_of_dams()
        ])
        self.old_flows = np.array([
            self.instance.get_initial_lags_of_channel(dam_id)[0] for dam_id in self.instance.get_ids_of_dams()
        ])

    def reset(self, instance: Instance = None) -> np.array:

        if instance is None:
            self.instance = self.create_instance(
                length_episodes=self.config.length_episodes,
                constants=self.constants,
                training_data=self.training_data,
                initial_row=None,
            )
        else:
            self.instance = instance

        self.river_basin.reset(self.instance)
        self.obs_low = self.get_obs_lower_limits()
        self.obs_high = self.get_obs_upper_limits()
        self.max_flows = np.array([
            self.instance.get_max_flow_of_channel(dam_id) for dam_id in self.instance.get_ids_of_dams()
        ])
        self.old_flows = np.array([
            self.instance.get_initial_lags_of_channel(dam_id)[0] for dam_id in self.instance.get_ids_of_dams()
        ])

        return self.get_observation()

    def get_obs_lower_limits(self) -> np.ndarray:

        """
        Returns the lower limits of the values of the agent's observations
        All attributes are 0, except the volume (for which the minimum volume is given)
        """

        return np.concatenate([
            np.repeat(0, self.config.num_prices),
            np.repeat(0, self.config.num_incoming_flows),
            *[
                np.concatenate([
                    np.array([self.instance.get_min_vol_of_dam(dam_id)]),
                    np.repeat(0, self.instance.get_relevant_lags_of_dam(dam_id)[-1]),
                    np.repeat(0, self.config.num_unreg_flows)
                ])
                for dam_id in self.instance.get_ids_of_dams()
            ]
        ])

    def get_obs_upper_limits(self) -> np.ndarray:

        """
        Returns the upper limits of the values of the agent's observations
        """

        return np.concatenate([
            np.repeat(self.instance.get_largest_price(), self.config.num_prices),
            np.repeat(self.instance.get_max_incoming_flow(), self.config.num_incoming_flows),
            *[
                np.concatenate([
                    np.array([self.instance.get_max_vol_of_dam(dam_id)]),
                    np.repeat(self.instance.get_max_flow_of_channel(dam_id),
                              self.instance.get_relevant_lags_of_dam(dam_id)[-1]),
                    np.repeat(self.instance.get_max_unregulated_flow_of_dam(dam_id), self.config.num_unreg_flows)
                ])
                for dam_id in self.instance.get_ids_of_dams()
            ]
        ])

    def get_observation(self, normalize: bool = True) -> np.array:

        """
        Returns the observation of the agent for the current state of the river basin
        """

        obs = np.concatenate([
            self.instance.get_price(self.river_basin.time + 1, num_steps=self.config.num_prices),
            self.instance.get_incoming_flow(self.river_basin.time + 1, num_steps=self.config.num_incoming_flows),
            *[
                np.concatenate([
                    dam.volume,
                    dam.channel.past_flows.squeeze(),
                    self.instance.get_unregulated_flow_of_dam(
                        self.river_basin.time + 1, dam.idx, num_steps=self.config.num_unreg_flows
                    )
                ])
                for dam in self.river_basin.dams
            ]
        ])

        if normalize:
            obs = (obs - self.obs_low) / (self.obs_high - self.obs_low)

        return obs.astype(np.float32)

    def step(self, action: np.array, normalize_obs: bool = True) -> tuple[np.array, float, bool, dict]:

        """
        Updates the river basin with the given action

        :param action: An array of size num_dams of values between -1 and 1
        indicating how much the flow of each channel should change
        (-1 => reduce it by 100% of the channel's max flow; 1 => increase it by 100% of the channel's max flow)
        :param normalize_obs: Whether to normalize the returned observation or not
        :return: The next observation, the reward obtained, and whether the episode is finished or not
        """

        new_flows = self.old_flows + action * self.max_flows
        self.river_basin.update(new_flows.reshape(-1, 1))
        self.old_flows = self.river_basin.get_clipped_flows().reshape(-1)

        # Reward - we divide the income and penalties by the maximum price in the episode
        # to avoid inconsistencies throughout episodes (in which energy prices are normalized differently)
        # Note we do not take into account the final volumes here; this is something the agent should tackle on its own
        income = self.river_basin.get_income().item()
        startups_penalty = self.river_basin.get_num_startups().item() * self.config.startups_penalty
        limit_zones_penalty = self.river_basin.get_num_times_in_limit().item() * self.config.limit_zones_penalty
        reward = (income - startups_penalty - limit_zones_penalty) / self.instance.get_largest_price()
        # print(self.river_basin.get_income().item(), startups_penalty, limit_zones_penalty, reward)

        next_obs = self.get_observation(normalize=normalize_obs)
        done = self.river_basin.time >= self.instance.get_largest_impact_horizon()

        return next_obs, reward, done, dict()

    @staticmethod
    def create_instance(
            length_episodes: int,
            constants: dict,
            training_data: pd.DataFrame,
            initial_row: int | datetime = None,
    ) -> Instance:

        # Incomplete instance (we create a deepcopy of constants to avoid modifying it)
        data = pickle.loads(pickle.dumps(constants, -1))
        instance_constants = Instance.from_dict(data)

        # Get necessary constants
        dam_ids = instance_constants.get_ids_of_dams()
        channel_last_lags = {
            dam_id: instance_constants.get_relevant_lags_of_dam(dam_id)[-1]
            for dam_id in dam_ids
        }

        # Required rows from data frame
        total_rows = len(training_data.index)
        min_row = max(channel_last_lags.values())
        max_row = total_rows - length_episodes
        if isinstance(initial_row, datetime):
            initial_row = training_data.index[
                training_data["datetime"] == initial_row
                ].tolist()[0]
        if initial_row is None:
            initial_row = randint(min_row, max_row)
        assert initial_row in range(
            min_row, max_row + 1
        ), f"{initial_row=} should be between {min_row=} and {max_row=}"
        last_row = initial_row + length_episodes - 1
        last_row_decisions = last_row - max(
            [
                instance_constants.get_relevant_lags_of_dam(dam_id)[0]
                for dam_id in instance_constants.get_ids_of_dams()
            ]
        )

        # Add time-dependent values to the data

        data["datetime"]["start"] = training_data.loc[
            initial_row, "datetime"
        ].strftime("%Y-%m-%d %H:%M")
        data["datetime"]["end_decisions"] = training_data.loc[
            last_row_decisions, "datetime"
        ].strftime("%Y-%m-%d %H:%M")

        data["incoming_flows"] = training_data.loc[
                                 initial_row:last_row, "incoming_flow"
                                 ].values.tolist()
        data["energy_prices"] = training_data.loc[
                                initial_row:last_row, "price"
                                ].values.tolist()

        for order, dam_id in enumerate(dam_ids):
            # Initial volume
            # Not to be confused with the volume at the end of the first time step
            data["dams"][order]["initial_vol"] = training_data.loc[
                initial_row, dam_id + "_vol"
            ]

            initial_lags = training_data.loc[
                           initial_row - channel_last_lags[dam_id]: initial_row - 1,
                           dam_id + "_flow",
                           ].values.tolist()
            initial_lags.reverse()
            data["dams"][order]["initial_lags"] = initial_lags

            data["dams"][order]["unregulated_flows"] = training_data.loc[
                                                       initial_row:last_row, dam_id + "_unreg_flow"
                                                       ].values.tolist()

        # Complete instance
        instance = Instance.from_dict(data)

        return instance
