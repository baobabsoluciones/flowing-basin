from flowing_basin.core import Instance, Configuration
from flowing_basin.tools import RiverBasin
from cornflow_client.core.tools import load_json
import numpy as np
import gymnasium as gym
import pandas as pd
from datetime import datetime
import pickle
from random import randint
from dataclasses import dataclass


@dataclass(kw_only=True)
class RLConfiguration(Configuration):  # noqa

    # RL environment options
    num_prices: int
    num_incoming_flows: int
    num_unreg_flows: int
    length_episodes: int

    # RL training configuration
    log_ep_freq: int = 5
    eval_ep_freq: int = 5
    eval_num_episodes: int = 5

    # RiverBasin simulator options
    flow_smoothing: int = 0
    mode: str = "nonlinear"
    fast_mode: bool = True

    def __post_init__(self):
        valid_modes = {"linear", "nonlinear"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid value for 'mode': {self.mode}. Allowed values are {valid_modes}")


class RLEnvironment(gym.Env):

    """
    Class representing the environment for the RL agent
    The class acts as a wrapper of the river basin:
     - It filters and normalizes the river basin's states, turning them into observations
     - It computes the rewards for the agent (proportional to the generated energy and its price)
    """

    def __init__(
        self,
        config: RLConfiguration,
        path_constants: str = None,
        path_historical_data: str = None,
        instance: Instance = None,
        initial_row: int | datetime = None,
        paths_power_models: dict[str, str] = None,
    ):

        super(RLEnvironment, self).__init__()

        self.config = config

        # Create instance
        self.constants = None
        if path_constants is not None:
            self.constants = load_json(path_constants)
        self.historical_data = None
        if path_historical_data is not None:
            self.historical_data = pd.read_pickle(path_historical_data)
        self._reset_instance(instance, initial_row)

        # Simulator (the core of the environment)
        self.river_basin = RiverBasin(
            instance=self.instance,
            mode=self.config.mode,
            flow_smoothing=self.config.flow_smoothing,
            paths_power_models=paths_power_models
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

        # Variables that depend on the instance
        self.obs_low = None
        self.obs_high = None
        self.max_flows = None
        self.old_flows = None

        # Initialize these variables
        self._reset_variables()

    def reset(
            self, instance: Instance = None, initial_row: int | datetime = None, seed=None, options=None
    ) -> tuple[np.array, dict]:

        super().reset(seed=seed)

        self._reset_instance(instance, initial_row)
        self.river_basin.reset(self.instance)
        self._reset_variables()

        # Print to debug...
        # new_start, _ = self.instance.get_start_end_datetimes()
        # print(f"Environment reset. New episode's starting datetime: {new_start.strftime('%Y-%m-%d %H:%M')}")
        # info_horizon = self.instance.get_information_horizon()
        # print(f"New episode's information horizon: {info_horizon}")

        return self.get_observation(), dict()

    def _reset_instance(self, instance: Instance = None, initial_row: int | datetime = None):

        """
        Reset the current instance
        """

        if instance is None:

            assert self.constants is not None and self.historical_data is not None, (
                "If you reset the environment without giving an instance,"
                "you need to give the path to the constants JSON and history dataframe in the class constructor,"
                "in order to be able to create a random instance."
            )

            self.instance = self.create_instance(
                length_episodes=self.config.length_episodes,
                constants=self.constants,
                historical_data=self.historical_data,
                config=self.config,
                initial_row=initial_row,
            )

        else:

            required_info_buffer = max(self.config.num_prices, self.config.num_incoming_flows, self.config.num_unreg_flows)
            actual_info_buffer = instance.get_information_horizon() - instance.get_largest_impact_horizon()
            assert actual_info_buffer >= required_info_buffer, (
                "Because of how much the RL agent looks ahead, the information horizon should be "
                f"{required_info_buffer} time steps ahead of the impact horizon, "
                f"but in the given instance it is only {actual_info_buffer} time steps ahead."
            )

            self.instance = instance

    def _reset_variables(self):

        """
        Reset variables according to the current instance
        """

        self.obs_low = self.get_obs_lower_limits()
        self.obs_high = self.get_obs_upper_limits()
        self.max_flows = np.array([
            self.instance.get_max_flow_of_channel(dam_id) for dam_id in self.instance.get_ids_of_dams()
        ])
        self.old_flows = np.array([
            self.instance.get_initial_lags_of_channel(dam_id)[0] for dam_id in self.instance.get_ids_of_dams()
        ])

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
        Returns the observation of the agent for the current state of the river basin, formed by:
            - the next N energy prices
            - the next N incoming flows to the basin
            - for each dam:
                - the current volume
                - the relevant past flows
                - the next N unregulated flows
        """

        # Print to debug...
        # print(f"time {self.river_basin.time}")
        # prices = self.instance.get_price(self.river_basin.time + 1, num_steps=self.config.num_prices)
        # print(f"\tprices ({len(prices)}): {prices}")
        # incoming_flows = self.instance.get_incoming_flow(self.river_basin.time + 1, num_steps=self.config.num_incoming_flows)
        # print(f"\tincoming flows ({len(incoming_flows)}): {incoming_flows}")
        # for dam in self.river_basin.dams:
        #     print(f"\t{dam.idx} volume (1): {dam.volume}")
        #     past_flows = dam.channel.past_flows.squeeze()
        #     print(f"\t{dam.idx} past flows({len(past_flows)}): {past_flows}")
        #     unreg_flows = self.instance.get_unregulated_flow_of_dam(self.river_basin.time + 1, dam.idx, num_steps=self.config.num_unreg_flows)
        #     print(f"\t{dam.idx} unregulated flows ({len(unreg_flows)}): {unreg_flows}")

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

    def get_reward(self) -> float:

        """
        Calculate the reward obtained with the current state of the river basin

        We divide the income and penalties by the maximum price in the episode
        to avoid inconsistencies throughout episodes (in which energy prices are normalized differently)
        Note we do not take into account the final volumes here; this is something the agent should tackle on its own
        """

        income = self.river_basin.get_income().item()
        startups_penalty = self.river_basin.get_num_startups().item() * self.config.startups_penalty
        limit_zones_penalty = self.river_basin.get_num_times_in_limit().item() * self.config.limit_zones_penalty
        reward = (income - startups_penalty - limit_zones_penalty) / self.instance.get_largest_price()
        # print(self.river_basin.get_income().item(), startups_penalty, limit_zones_penalty, reward)

        return reward

    def step(self, action: np.array, normalize_obs: bool = True) -> tuple[np.array, float, bool, bool, dict]:

        """
        Updates the river basin with the given action

        :param action: An array of size num_dams of values between -1 and 1
        indicating how much the flow of each channel should change
        (-1 => reduce it by 100% of the channel's max flow; 1 => increase it by 100% of the channel's max flow)
        :param normalize_obs: Whether to normalize the returned observation or not
        :return: The next observation, the reward obtained, and whether the episode is finished or not
        """

        new_flows = self.old_flows + action * self.max_flows  # noqa
        self.river_basin.update(new_flows.reshape(-1, 1), fast_mode=self.config.fast_mode)
        self.old_flows = self.river_basin.get_clipped_flows().reshape(-1)

        next_obs = self.get_observation(normalize=normalize_obs)
        reward = self.get_reward()
        done = self.river_basin.time >= self.instance.get_largest_impact_horizon() - 1

        return next_obs, reward, done, False, dict()

    @staticmethod
    def create_instance(
            length_episodes: int,
            constants: dict,
            historical_data: pd.DataFrame,
            config: RLConfiguration = None,
            initial_row: int | datetime = None,
    ) -> Instance:

        """
        Create an instance from the data frame of historical data.

        :param length_episodes: Number of time steps of the episodes (including impact buffer)
        :param constants: Dictionary with the constants (e.g. constant physical characteristics of the dams)
        :param historical_data: Data frame with the time-dependent values (e.g. volume of the dams at a particular time)
        :param config: If given, calculates a greater information horizon
            according to the number of time steps the RL agent looks ahead
        :param initial_row: If given, starts the episode in this row or datetime
        """

        # Incomplete instance (we create a deepcopy of constants to avoid modifying it)
        data = pickle.loads(pickle.dumps(constants, -1))
        instance_constants = Instance.from_dict(data)

        # Get necessary constants
        dam_ids = instance_constants.get_ids_of_dams()
        channel_last_lags = {
            dam_id: instance_constants.get_relevant_lags_of_dam(dam_id)[-1]
            for dam_id in dam_ids
        }

        # Impact buffer (included in the length of the episode) and information buffer (added on top of it)
        impact_buffer = max(
            [
                instance_constants.get_relevant_lags_of_dam(dam_id)[0]
                for dam_id in instance_constants.get_ids_of_dams()
            ]
        )
        info_buffer = 0
        if config is not None:
            info_buffer = max(config.num_prices, config.num_incoming_flows, config.num_unreg_flows)

        # Required rows from data frame
        total_rows = len(historical_data.index)
        min_row = max(channel_last_lags.values())
        max_row = total_rows - length_episodes - info_buffer

        # Initial row
        if isinstance(initial_row, datetime):
            initial_row = historical_data.index[
                historical_data["datetime"] == initial_row
                ].tolist()[0]
        if initial_row is None:
            initial_row = randint(min_row, max_row)
        assert initial_row in range(
            min_row, max_row + 1
        ), f"{initial_row=} should be between {min_row=} and {max_row=}"

        # Last rows
        last_row_impact = initial_row + length_episodes - 1
        last_row_decisions = last_row_impact - impact_buffer
        last_row_info = last_row_impact + info_buffer

        # Add time-dependent values to the data

        data["datetime"]["start"] = historical_data.loc[
            initial_row, "datetime"
        ].strftime("%Y-%m-%d %H:%M")
        data["datetime"]["end_decisions"] = historical_data.loc[
            last_row_decisions, "datetime"
        ].strftime("%Y-%m-%d %H:%M")
        data["datetime"]["end_information"] = historical_data.loc[
            last_row_info, "datetime"
        ].strftime("%Y-%m-%d %H:%M")

        data["incoming_flows"] = historical_data.loc[
            initial_row: last_row_info, "incoming_flow"
                                 ].values.tolist()
        data["energy_prices"] = historical_data.loc[
            initial_row: last_row_info, "price"
                                ].values.tolist()

        for order, dam_id in enumerate(dam_ids):
            # Initial volume
            # Not to be confused with the volume at the end of the first time step
            data["dams"][order]["initial_vol"] = historical_data.loc[
                initial_row, dam_id + "_vol"
            ]

            initial_lags = historical_data.loc[
               initial_row - channel_last_lags[dam_id]: initial_row - 1,
               dam_id + "_flow",
                           ].values.tolist()
            initial_lags.reverse()
            data["dams"][order]["initial_lags"] = initial_lags

            data["dams"][order]["unregulated_flows"] = historical_data.loc[
                initial_row: last_row_info, dam_id + "_unreg_flow"
                                                       ].values.tolist()

        # Complete instance
        instance = Instance.from_dict(data)

        return instance
