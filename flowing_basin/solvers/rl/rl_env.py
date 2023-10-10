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
from typing import Callable


@dataclass(kw_only=True)
class RLConfiguration(Configuration):  # noqa

    # Penalty for not fulfilling the flow smoothing parameter
    flow_smoothing_penalty: int  # Penalty for not fulfilling the flow smoothing parameter
    flow_smoothing_clip: bool  # Whether to clip the actions that do not comply with flow smoothing or not

    # RL environment's observation options
    features: list[str]
    num_steps_sight: int
    length_episodes: int

    # RL environment's action options
    action_type: str

    # RL training configuration
    log_ep_freq: int = 5
    eval_ep_freq: int = 5
    eval_num_episodes: int = 5

    # RiverBasin simulator options
    flow_smoothing: int = 0
    mode: str = "nonlinear"
    do_history_updates: bool = True

    def __post_init__(self):

        valid_modes = {"linear", "nonlinear"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid value for 'mode': {self.mode}. Allowed values are {valid_modes}")

        valid_features = {
            "past_vols", "past_flows", "past_prices", "future_prices", "past_inflows", "future_inflows",
            "past_groups", "past_powers", "past_clipped", "past_periods"
        }
        for feature in self.features:
            if feature not in valid_features:
                raise ValueError(f"Invalid feature: {feature}. Allowed features are {valid_features}")

        valid_actions = {"exiting_flows", "exiting_relvars"}
        if self.action_type not in valid_actions:
            raise ValueError(f"Invalid value for 'action_type': {self.action_type}. Allowed values are {valid_actions}")


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
            flow_smoothing=self.config.flow_smoothing if self.config.flow_smoothing_clip else 0,
            paths_power_models=paths_power_models,
            do_history_updates=self.config.do_history_updates
        )

        # Observation is an array of shape num_dams x num_features x num_steps
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.instance.get_num_dams(), len(self.config.features), self.config.num_steps_sight),
            dtype=np.float32
        )

        # Action is an array of shape num_dams
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.instance.get_num_dams(), ),
            dtype=np.float32
        )
        # The action space should be bounded between -1 and 1
        # even if a lower bound of 0 would make more sense
        # Source: https://github.com/hill-a/stable-baselines/issues/473

        # Functions to calculate features
        self.features_functions = self.get_features_functions()

        # Variables that depend on the instance
        self.features_min_values = None
        self.features_max_values = None
        self.max_flows = None

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

        return self.get_observation_normalized(), dict()

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
                info_buffer_start=self.config.num_steps_sight,
                info_buffer_end=self.config.num_steps_sight,
                initial_row_decisions=initial_row,
            )

        else:

            required_info_buffer = self.config.num_steps_sight

            actual_info_buffer_end = instance.get_information_horizon() - instance.get_largest_impact_horizon()
            assert actual_info_buffer_end >= required_info_buffer, (
                "Because of how much the RL agent looks ahead, the information horizon should be "
                f"{required_info_buffer} time steps ahead of the impact horizon, "
                f"but in the given instance it is only {actual_info_buffer_end} time steps ahead."
            )

            actual_info_buffer_start = instance.get_start_information_offset()
            assert actual_info_buffer_start >= required_info_buffer, (
                "Because of how much the RL agent looks back, the information time steps should start "
                f"{actual_info_buffer_start} before the decision time steps, "
                f"but in the given instance they only start {actual_info_buffer_start} time steps back."
            )

            self.instance = instance

    def _reset_variables(self):

        """
        Reset variables according to the current instance
        """

        # Min and max values used to normalize observation
        self.features_min_values = self.get_features_min_values()
        self.features_max_values = self.get_features_max_values()

        self.max_flows = np.array([
            self.instance.get_max_flow_of_channel(dam_id) for dam_id in self.instance.get_ids_of_dams()
        ])

    def get_features_min_values(self) -> dict[str, dict[str, float | int]]:

        """
        Get the minimum values of the features in the observations
        according to the current instance
        """

        min_values = {
            dam_id: {
                "past_vols": self.instance.get_min_vol_of_dam(dam_id),
                "past_flows": 0.,
                "past_prices": 0.,
                "future_prices": 0.,
                "past_inflows": 0.,
                "future_inflows": 0.,
                "past_groups": 0,
                "past_powers": 0.,
                "past_clipped": - self.instance.get_max_flow_of_channel(dam_id),
                "past_periods": - self.config.num_steps_sight
            }
            for dam_id in self.instance.get_ids_of_dams()
        }

        return min_values

    def get_features_max_values(self) -> dict[str, dict[str, float | int]]:

        """
        Get the maximum values of the features in the observations
        according to the current instance
        """

        max_values = {
            dam_id: {
                "past_vols": self.instance.get_max_vol_of_dam(dam_id),
                "past_flows": self.instance.get_max_flow_of_channel(dam_id),
                "past_prices": self.instance.get_largest_price(),
                "future_prices": self.instance.get_largest_price(),
                "past_inflows": (
                    self.instance.get_max_unregulated_flow_of_dam(dam_id)
                    if self.instance.get_order_of_dam(dam_id) > 1
                    else self.instance.get_max_unregulated_flow_of_dam(dam_id) + self.instance.get_max_incoming_flow()
                ),
                "future_inflows": (
                    self.instance.get_max_unregulated_flow_of_dam(dam_id)
                    if self.instance.get_order_of_dam(dam_id) > 1
                    else self.instance.get_max_unregulated_flow_of_dam(dam_id) + self.instance.get_max_incoming_flow()
                ),
                "past_groups": self.instance.get_max_num_power_groups(dam_id),
                "past_powers": self.instance.get_max_power_of_power_group(dam_id),
                "past_clipped": self.instance.get_max_flow_of_channel(dam_id),
                "past_periods": self.instance.get_largest_impact_horizon()
            }
            for dam_id in self.instance.get_ids_of_dams()
        }

        return max_values

    def get_features_functions(self) -> dict[str, Callable[[str], np.array]]:

        """
        Map each feature to a function that returns its value for each dam
        """

        features_functions = {
            "past_vols": lambda dam_id: np.flip(
                self.river_basin.all_past_volumes[dam_id].squeeze()[
                    -self.config.num_steps_sight:
                ]
            ),
            "past_flows": lambda dam_id: np.flip(
                self.river_basin.all_past_clipped_flows.squeeze()[
                    -self.config.num_steps_sight:, self.instance.get_order_of_dam(dam_id) - 1
                ]
            ),
            "past_prices": lambda dam_id: np.flip(
                self.instance.get_all_prices()[
                    self.river_basin.time + 1 + self.instance.get_start_information_offset() - self.config.num_steps_sight:
                        self.river_basin.time + 1 + self.instance.get_start_information_offset()
                ]
            ),
            "future_prices": lambda dam_id: np.array(
                self.instance.get_all_prices()[
                    self.river_basin.time + 1 + self.instance.get_start_information_offset():
                        self.river_basin.time + 1 + self.instance.get_start_information_offset() + self.config.num_steps_sight
                ]
            ),
            "past_inflows": lambda dam_id: np.flip(
                self.instance.get_all_unregulated_flows_of_dam(dam_id)[
                    self.river_basin.time + 1 + self.instance.get_start_information_offset() - self.config.num_steps_sight:
                        self.river_basin.time + 1 + self.instance.get_start_information_offset()
                ]
            ) + (
                np.flip(
                    self.instance.get_all_incoming_flows()[
                        self.river_basin.time + 1 + self.instance.get_start_information_offset() - self.config.num_steps_sight:
                            self.river_basin.time + 1 + self.instance.get_start_information_offset()
                    ]
                ) if self.instance.get_order_of_dam(dam_id) == 1
                else 0.
            ),
            "future_inflows": lambda dam_id: np.array(
                self.instance.get_all_unregulated_flows_of_dam(dam_id)[
                    self.river_basin.time + 1 + self.instance.get_start_information_offset():
                        self.river_basin.time + 1 + self.instance.get_start_information_offset() + self.config.num_steps_sight
                ]
            ) + (
                np.array(
                    self.instance.get_all_incoming_flows()[
                        self.river_basin.time + 1 + self.instance.get_start_information_offset():
                            self.river_basin.time + 1 + self.instance.get_start_information_offset() + self.config.num_steps_sight
                    ]
                ) if self.instance.get_order_of_dam(dam_id) == 1
                else 0.
            ),
            "past_groups": lambda dam_id: np.flip(
                self.river_basin.all_past_groups[dam_id].squeeze()[
                    -self.config.num_steps_sight:
                ]
            ),
            "past_powers": lambda dam_id: np.flip(
                self.river_basin.all_past_powers[dam_id].squeeze()[
                    -self.config.num_steps_sight:
                ]
            ),
            "past_clipped": lambda dam_id: np.flip(
                self.river_basin.all_past_flows.squeeze()[
                    -self.config.num_steps_sight:, self.instance.get_order_of_dam(dam_id) - 1
                ]
            ) - np.flip(
                self.river_basin.all_past_clipped_flows.squeeze()[
                    -self.config.num_steps_sight:, self.instance.get_order_of_dam(dam_id) - 1
                ]
            ),
            "past_periods": lambda dam_id: np.array(
                [self.river_basin.time - i for i in range(self.config.num_steps_sight)]
            )
        }

        return features_functions

    def get_observation(self) -> np.array:

        """
        Returns the observation of the agent for the current state of the river basin

        :return: Array of shape num_dams x num_features x num_steps
        """

        obs = np.array([
            [
                self.features_functions[feature](dam_id)
                for feature in self.config.features
            ]
            for dam_id in self.instance.get_ids_of_dams()
        ]).astype(np.float32)

        return obs

    def get_observation_normalized(self) -> np.array:

        """
        Returns the normalized observation of the agent for the current state of the river basin

        :return: Array of shape num_dams x num_features x num_steps with normalized values
        """

        obs_normalized = np.array([
            [
                (
                    (self.features_functions[feature](dam_id) - self.features_min_values[dam_id][feature]) /
                    (self.features_max_values[dam_id][feature] - self.features_min_values[dam_id][feature])
                )
                for feature in self.config.features
            ]
            for dam_id in self.instance.get_ids_of_dams()
        ]).astype(np.float32)

        return obs_normalized

    def print_observation(self, dam_id: str, normalize: bool = False, decimals: int = 2, spacing: int = 15):

        """
        Prints the observation of the agent for the current state of the river basin
        """

        # Get observation, array of shape num_dams x num_features x num_steps
        if not normalize:
            obs = self.get_observation()
        else:
            obs = self.get_observation_normalized()

        # Header
        print(f"Observation for {dam_id}")
        print(''.join([f"{feature:^{spacing}}" for feature in self.config.features]))

        # Rows
        dam_index = self.instance.get_order_of_dam(dam_id) - 1
        for time_step in range(self.config.num_steps_sight):
            print(''.join([
                f"{obs[dam_index, feature_index, time_step]:^{spacing}.{decimals}f}"
                for feature_index, feature in enumerate(self.config.features)
            ]))
        print()

    def get_reward_details(self, epsilon: float = 1e-4) -> dict[str, float]:

        """
        Calculate the values that form the reward obtained
        with the current state of the river basin
        """

        income = self.river_basin.get_income().item()
        startups_penalty = self.river_basin.get_num_startups().item() * self.config.startups_penalty
        limit_zones_penalty = self.river_basin.get_num_times_in_limit().item() * self.config.limit_zones_penalty

        # Calculate flow smoothing penalty
        flow_smoothing_uncompliance = np.any([
            dam.all_previous_variations[-self.config.flow_smoothing-1: -1] * dam.all_previous_variations[-1] < - epsilon
            for dam in self.river_basin.dams
        ])
        flow_smoothing_penalty = flow_smoothing_uncompliance * self.config.flow_smoothing_penalty

        reward_details = dict(
            income=income,
            startups_penalty=-startups_penalty,
            limit_zones_penalty=-limit_zones_penalty,
            flow_smoothing_penalty=-flow_smoothing_penalty,
        )

        return reward_details

    def get_reward(self) -> float:

        """
        Calculate the reward from its components

        We divide the income and penalties by the maximum price in the episode
        to avoid inconsistencies throughout episodes (in which energy prices are normalized differently)
        Note we do not take into account the final volumes here; this is something the agent should tackle on its own
        """

        reward = sum(reward_component for reward_component in self.get_reward_details().values())
        reward = reward / self.instance.get_largest_price()

        return reward

    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:

        """
        Updates the river basin with the given action

        :param action: An array of size num_dams whose meaning depens on the type of action in the configuration
        :return: The next observation, the reward obtained, and whether the episode is finished or not
        """

        # Transform action to flows
        # Remember action is bounded between -1 and 1 in any case
        if self.config.action_type == "exiting_relvars":
            old_flows = self.river_basin.get_clipped_flows().reshape(-1)
            new_flows = old_flows + action * self.max_flows  # noqa
        else:
            new_flows = (action + 1.) / 2. * self.max_flows

        self.river_basin.update(new_flows.reshape(-1, 1))

        next_obs = self.get_observation_normalized()
        reward = self.get_reward()
        done = self.river_basin.time >= self.instance.get_largest_impact_horizon() - 1

        return next_obs, reward, done, False, dict()

    @staticmethod
    def create_instance(
            length_episodes: int,
            constants: dict,
            historical_data: pd.DataFrame,
            info_buffer_start: int = 0,
            info_buffer_end: int = 0,
            initial_row_decisions: int | datetime = None,
    ) -> Instance:

        """
        Create an instance from the data frame of historical data.

        :param length_episodes: Number of time steps of the episodes (including impact buffer)
        :param constants: Dictionary with the constants (e.g. constant physical characteristics of the dams)
        :param historical_data: Data frame with the time-dependent values (e.g. volume of the dams at a particular time)
        :param info_buffer_start: Number of time steps with information before the decisions must be made
        :param info_buffer_end: Number of time steps with information after the decisions have been made
        :param initial_row_decisions: If given, starts the episode in this row or datetime
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

        # Required rows from data frame
        total_rows = len(historical_data.index)
        min_row = info_buffer_start + max(channel_last_lags.values())
        max_row = total_rows - length_episodes - info_buffer_end

        # Initial row decisions
        if isinstance(initial_row_decisions, datetime):
            initial_row_decisions = historical_data.index[
                historical_data["datetime"] == initial_row_decisions
            ].tolist()[0]
        if initial_row_decisions is None:
            initial_row_decisions = randint(min_row, max_row)
        assert initial_row_decisions in range(
            min_row, max_row + 1
        ), f"{initial_row_decisions=} should be between {min_row=} and {max_row=}"

        # Initial row info
        initial_row_info = initial_row_decisions - info_buffer_start

        # Last rows
        last_row_impact = initial_row_decisions + length_episodes - 1
        last_row_decisions = last_row_impact - impact_buffer
        last_row_info = last_row_impact + info_buffer_end

        # Add time-dependent values to the data

        data["datetime"]["start_information"] = historical_data.loc[
            initial_row_info, "datetime"
        ].strftime("%Y-%m-%d %H:%M")
        data["datetime"]["start"] = historical_data.loc[
            initial_row_decisions, "datetime"
        ].strftime("%Y-%m-%d %H:%M")
        data["datetime"]["end_decisions"] = historical_data.loc[
            last_row_decisions, "datetime"
        ].strftime("%Y-%m-%d %H:%M")
        data["datetime"]["end_information"] = historical_data.loc[
            last_row_info, "datetime"
        ].strftime("%Y-%m-%d %H:%M")

        data["incoming_flows"] = historical_data.loc[
            initial_row_info: last_row_info, "incoming_flow"
        ].values.tolist()
        data["energy_prices"] = historical_data.loc[
            initial_row_info: last_row_info, "price"
        ].values.tolist()

        for order, dam_id in enumerate(dam_ids):

            # If dam is not dam1 or dam2,
            # it will be e.g. dam3_dam2copy (a copy of dam2) or dam4_dam1copy (a copy of dam1)
            original_dam_id = dam_id
            if dam_id not in ["dam1", "dam2"]:
                dam_id = dam_id[dam_id.rfind("_") + 1: dam_id.rfind("copy")]

            # Initial volume
            # Not to be confused with the volume at the end of the first time step
            data["dams"][order]["initial_vol"] = historical_data.loc[
                initial_row_info, dam_id + "_vol"
            ]

            initial_lags = historical_data.loc[
                initial_row_info - channel_last_lags[dam_id]: initial_row_info - 1, dam_id + "_flow"
            ].values.tolist()
            initial_lags.reverse()
            data["dams"][order]["initial_lags"] = initial_lags

            if initial_row_info != initial_row_decisions:
                starting_flows = historical_data.loc[
                    initial_row_info: initial_row_decisions - 1, dam_id + "_flow"
                ].values.tolist()
                starting_flows.reverse()
                data["dams"][order]["starting_flows"] = starting_flows

            # Unregulated flow
            # We will only consider the unregulated flow of the original dams,
            # and not of the artificially created extra dams
            if original_dam_id in ["dam1", "dam2"]:
                data["dams"][order]["unregulated_flows"] = historical_data.loc[
                    initial_row_info: last_row_info, dam_id + "_unreg_flow"
                ].values.tolist()
            else:
                data["dams"][order]["unregulated_flows"] = [0 for _ in range(initial_row_info, last_row_info + 1)]

            # Final volume: volume at the decision horizon in the historical record
            # Optional field that may be used to set the objective final volumes for the solvers
            final_vol = historical_data.loc[
                last_row_decisions + 1, dam_id + "_vol"
            ]
            final_vol = min(final_vol, data["dams"][order]["vol_max"])
            data["dams"][order]["final_vol"] = final_vol

        # Complete instance
        instance = Instance.from_dict(data)

        return instance
