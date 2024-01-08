from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
from flowing_basin.solvers.rl import RLConfiguration
from flowing_basin.solvers.rl.feature_extractors import Projector
from cornflow_client.core.tools import load_json
import numpy as np
import gymnasium as gym
import pandas as pd
from datetime import datetime
import pickle
from random import randint
from typing import Callable


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
            projector: Projector,
            update_observation_record: bool = False,
            path_constants: str = None,
            path_historical_data: str = None,
            instance: Instance = None,
            initial_row: int | datetime = None,
            paths_power_models: dict[str, str] = None,
    ):

        super(RLEnvironment, self).__init__()

        if instance is None:
            if path_constants is None or path_historical_data is None:
                raise ValueError(
                    "If you do not give an instance to the RLEnvironment class constructor, "
                    "you need to give both path_constants and path_historical_data "
                    "in order to be able to create random instances."
                )

        self.config = config
        self.projector = projector
        self.update_observation_record = update_observation_record

        # Set data
        self.constants = None
        if path_constants is not None:
            self.constants = Instance.from_dict(load_json(path_constants))
        self.historical_data = None
        if path_historical_data is not None:
            self.historical_data = pd.read_pickle(path_historical_data)

        # Create instance
        self._reset_instance(instance, initial_row)

        # Simulator (the core of the environment)
        self.river_basin = RiverBasin(
            instance=self.instance,
            mode=self.config.mode,
            flow_smoothing=self.config.flow_smoothing if self.config.flow_smoothing_clip else 0,
            paths_power_models=paths_power_models,
            do_history_updates=self.config.do_history_updates
        )

        # Observation space
        if self.config.feature_extractor == 'MLP':

            # Raw or normalized observation shape
            array_length = sum(
                self.config.num_steps_sight[feature, dam_id]
                for feature in self.config.features for dam_id in self.instance.get_ids_of_dams()
                if self.instance.get_order_of_dam(dam_id) == 1 or feature not in self.config.unique_features
            )
            self.obs_shape = (array_length,)

            # Projected observation shape
            if self.projector.n_components is None:
                self.projected_obs_shape = self.obs_shape
            else:
                self.projected_obs_shape = (self.projector.n_components,)

        elif self.config.feature_extractor == 'CNN':

            # Convolutional feature extractors need (Channels x Height x Width) -> (Dams x Lookback x Features)
            # Raw/normalized and projected observation shape must be the same
            self.obs_shape = (
                self.instance.get_num_dams(),
                self.config.num_steps_sight[self.config.features[0], self.instance.get_ids_of_dams()[0]],
                len(self.config.features)
            )
            self.projected_obs_shape = self.obs_shape

        else:
            raise NotImplementedError(f"Feature extractor {self.config.feature_extractor} is not supported yet.")

        self.observation_space = gym.spaces.Box(
            low=self.projector.low,
            high=self.projector.high,
            shape=self.projected_obs_shape,
            dtype=np.float32
        )

        # Record of observations experienced by the agent
        # Array of shape num_observations x num_features (each observation will be flattened)
        self.record_raw_obs = []
        self.record_normalized_obs = []
        self.record_projected_obs = []

        # Action space
        # Action is an array of shape num_dams
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.instance.get_num_dams() * self.config.num_actions_block,),
            dtype=np.float32
        )
        # The action space should be bounded between -1 and 1
        # even if a lower bound of 0 would make more sense
        # Source: https://github.com/hill-a/stable-baselines/issues/473

        # Functions to calculate features
        self.features_functions = self.get_features_functions()
        self.features_max_functions = self.get_features_max_functions()
        self.features_min_functions = self.get_features_min_functions()

        # Observation indeces
        self.obs_indeces = self.config.get_obs_indices()

        # Variables that depend on the instance
        self.obs_min = None
        self.obs_max = None
        self.max_flows = None

        # Initialize these variables
        self._reset_variables()

    def reset(
            self, instance: Instance = None, initial_row: int | datetime = None, seed=None, options=None
    ) -> tuple[np.ndarray, dict]:

        super().reset(seed=seed)

        self._reset_instance(instance, initial_row)
        self.river_basin.reset(self.instance)
        self._reset_variables()

        raw_obs = self.get_obs_array()
        normalized_obs = self.normalize(raw_obs)
        projected_obs = self.project(normalized_obs)

        return projected_obs, dict(raw_obs=raw_obs, normalized_obs=normalized_obs)

    def _reset_instance(self, instance: Instance = None, initial_row: int | datetime = None):

        """
        Reset the current instance
        """

        if instance is None:

            max_sight = max(self.config.num_steps_sight.values())
            self.instance = self.create_instance(
                length_episodes=self.config.length_episodes,
                constants=self.constants,
                historical_data=self.historical_data,
                info_buffer_start=max_sight,
                info_buffer_end=max_sight,
                initial_row_decisions=initial_row,
            )

        else:

            required_info_buffer = max(self.config.num_steps_sight.values())

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

        # Min and max observation arrays (for normalization)
        self.obs_min = self.get_obs_array(values='min')
        self.obs_max = self.get_obs_array(values='max')

        self.max_flows = np.array([
            self.instance.get_max_flow_of_channel(dam_id) for dam_id in self.instance.get_ids_of_dams()
        ])

    def get_features_min_functions(self) -> dict[str, Callable[[str], float | int]]:

        """
        Functions that give the minimum values of the features in the observations
        according to the current instance
        """

        min_values = {
            "past_vols": lambda dam_id: self.instance.get_min_vol_of_dam(dam_id),
            "past_flows": lambda dam_id: 0.,
            "past_variations": lambda dam_id: - self.instance.get_max_flow_of_channel(dam_id),
            "past_prices": lambda dam_id: 0.,
            "future_prices": lambda dam_id: 0.,
            "past_inflows": lambda dam_id: 0.,
            "future_inflows": lambda dam_id: 0.,
            "past_turbined": lambda dam_id: 0.,
            "past_groups": lambda dam_id: 0,
            "past_powers": lambda dam_id: 0.,
            "past_clipped": lambda dam_id: (
                0. if not self.config.flow_smoothing_clip
                else - self.instance.get_max_flow_of_channel(dam_id)
            ),
            "past_periods": lambda dam_id: (
                - self.config.num_steps_sight["past_periods", dam_id] if "past_periods" in self.config.features
                else None
            )
        }

        return min_values

    def get_features_max_functions(self) -> dict[str, Callable[[str], float | int]]:

        """
        Functions that give the maximum values of the features in the observations
        according to the current instance
        """

        max_values = {
            "past_vols": lambda dam_id: self.instance.get_max_vol_of_dam(dam_id),
            "past_flows": lambda dam_id: self.instance.get_max_flow_of_channel(dam_id),
            "past_variations": lambda dam_id: self.instance.get_max_flow_of_channel(dam_id),
            "past_prices": lambda dam_id: self.instance.get_largest_price(),
            "future_prices": lambda dam_id: self.instance.get_largest_price(),
            "past_inflows": lambda dam_id: (
                self.instance.get_max_unregulated_flow_of_dam(dam_id)
                if self.instance.get_order_of_dam(dam_id) > 1
                else self.instance.get_max_unregulated_flow_of_dam(dam_id) + self.instance.get_max_incoming_flow()
            ),
            "future_inflows": lambda dam_id: (
                self.instance.get_max_unregulated_flow_of_dam(dam_id)
                if self.instance.get_order_of_dam(dam_id) > 1
                else self.instance.get_max_unregulated_flow_of_dam(dam_id) + self.instance.get_max_incoming_flow()
            ),
            "past_turbined": lambda dam_id: self.instance.get_max_flow_of_channel(dam_id),
            "past_groups": lambda dam_id: self.instance.get_max_num_power_groups(dam_id),
            "past_powers": lambda dam_id: self.instance.get_max_power_of_power_group(dam_id),
            "past_clipped": lambda dam_id: self.instance.get_max_flow_of_channel(dam_id),
            "past_periods": lambda dam_id: self.instance.get_largest_impact_horizon()
        }

        return max_values

    def get_feature_past_clipped(self, dam_id: str) -> np.ndarray:

        """
        Get the amount of flow by which the past actions were clipped
        """

        assigned_flows = np.flip(
            self.river_basin.all_past_flows.squeeze()[
            -self.config.num_steps_sight["past_clipped", dam_id]:, self.instance.get_order_of_dam(dam_id) - 1
            ]
        )
        actual_flows = np.flip(
            self.river_basin.all_past_clipped_flows.squeeze()[
            -self.config.num_steps_sight["past_clipped", dam_id]:, self.instance.get_order_of_dam(dam_id) - 1
            ]
        )
        flow_clipping = assigned_flows - actual_flows

        # If flow is not being smoothed, then clipping cannot be positive
        # (i.e., the actual flow cannot be higher than the assigned flow)
        # Force it is positive to avoid small negative values due to rounding errors
        if not self.config.flow_smoothing_clip:
            flow_clipping = np.clip(flow_clipping, 0., None)

        return flow_clipping

    def get_features_functions(self) -> dict[str, Callable[[str], np.ndarray]]:

        """
        Map each feature to a function that returns its value for each dam
        """

        features_functions = {

            "past_vols": lambda dam_id: np.flip(
                self.river_basin.all_past_volumes[dam_id].squeeze()[
                -self.config.num_steps_sight["past_vols", dam_id]:
                ]
            ),

            "past_flows": lambda dam_id: np.flip(
                self.river_basin.all_past_clipped_flows.squeeze()[
                -self.config.num_steps_sight["past_flows", dam_id]:, self.instance.get_order_of_dam(dam_id) - 1
                ]
            ),

            "past_variations": lambda dam_id: np.flip(
                self.river_basin.dams[self.instance.get_order_of_dam(dam_id) - 1].all_previous_variations.squeeze()[
                -self.config.num_steps_sight["past_variations", dam_id]:
                ]
            ),

            "past_prices": lambda dam_id: np.flip(
                self.instance.get_all_prices()[
                self.river_basin.time + 1 + self.instance.get_start_information_offset() - self.config.num_steps_sight[
                    "past_prices", dam_id]:
                self.river_basin.time + 1 + self.instance.get_start_information_offset()
                ]
            ),

            "future_prices": lambda dam_id: np.array(
                self.instance.get_all_prices()[
                self.river_basin.time + 1 + self.instance.get_start_information_offset():
                self.river_basin.time + 1 + self.instance.get_start_information_offset() + self.config.num_steps_sight[
                    "future_prices", dam_id]
                ]
            ),

            "past_inflows": lambda dam_id: np.flip(
                self.instance.get_all_unregulated_flows_of_dam(dam_id)[
                self.river_basin.time + 1 + self.instance.get_start_information_offset() - self.config.num_steps_sight[
                    "past_inflows", dam_id]:
                self.river_basin.time + 1 + self.instance.get_start_information_offset()
                ]
            ) + (
               np.flip(
                   self.instance.get_all_incoming_flows()[
                   self.river_basin.time + 1 + self.instance.get_start_information_offset() -
                   self.config.num_steps_sight["past_inflows", dam_id]:
                   self.river_basin.time + 1 + self.instance.get_start_information_offset()
                   ]
               ) if self.instance.get_order_of_dam(dam_id) == 1
               else 0.
           ),

            "future_inflows": lambda dam_id: np.array(
                self.instance.get_all_unregulated_flows_of_dam(dam_id)[
                self.river_basin.time + 1 + self.instance.get_start_information_offset():
                self.river_basin.time + 1 + self.instance.get_start_information_offset() + self.config.num_steps_sight[
                    "future_inflows", dam_id]
                ]
            ) + (
                                                 np.array(
                                                     self.instance.get_all_incoming_flows()[
                                                     self.river_basin.time + 1 + self.instance.get_start_information_offset():
                                                     self.river_basin.time + 1 + self.instance.get_start_information_offset() +
                                                     self.config.num_steps_sight["future_inflows", dam_id]
                                                     ]
                                                 ) if self.instance.get_order_of_dam(dam_id) == 1
                                                 else 0.
                                             ),

            "past_turbined": lambda dam_id: np.flip(
                self.river_basin.all_past_turbined[dam_id].squeeze()[
                -self.config.num_steps_sight["past_turbined", dam_id]:
                ]
            ),

            "past_groups": lambda dam_id: np.flip(
                self.river_basin.all_past_groups[dam_id].squeeze()[
                -self.config.num_steps_sight["past_groups", dam_id]:
                ]
            ),

            "past_powers": lambda dam_id: np.flip(
                self.river_basin.all_past_powers[dam_id].squeeze()[
                -self.config.num_steps_sight["past_powers", dam_id]:
                ]
            ),

            "past_clipped": self.get_feature_past_clipped,

            "past_periods": lambda dam_id: np.array(
                [self.river_basin.time - i for i in range(self.config.num_steps_sight["past_periods", dam_id])]
            ).astype(np.float32)

        }

        return features_functions

    def get_feature(self, feature: str, dam_id: str, value: str = None):

        """
        Returns the value of the given feature in the given dam

        :param feature:
        :param dam_id:
        :param value: Indicates which value to obtain -
            'max' for the max value, 'min' for the min value, and None for the current value
        """

        if value == 'min':
            return np.repeat(
                self.features_min_functions[feature](dam_id), self.config.num_steps_sight[feature, dam_id]
            )
        elif value == 'max':
            return np.repeat(
                self.features_max_functions[feature](dam_id), self.config.num_steps_sight[feature, dam_id]
            )
        else:
            if not self.config.obs_random or feature in self.config.obs_random_features_excluded:
                # Actual value
                return self.features_functions[feature](dam_id)
            else:
                # Random value
                return np.random.uniform(
                    low=self.features_min_functions[feature](dam_id),
                    high=self.features_max_functions[feature](dam_id),
                    size=self.config.num_steps_sight[feature, dam_id]
                )

    def get_obs_array(self, values: str = None) -> np.ndarray:

        """
        Returns the observation array of the agent for the current state of the river basin

        :param values: Indicates which values to obtain -
            'max' for max values, 'min' for min values, and None for current values
        :return: Raw observation array
        """

        if self.config.feature_extractor == 'MLP':
            # Observation is a 1d array
            raw_obs = np.concatenate([
                np.concatenate([
                    self.get_feature(feature, dam_id, values)
                    for feature in self.config.features
                    if self.instance.get_order_of_dam(dam_id) == 1 or feature not in self.config.unique_features
                ])
                for dam_id in self.instance.get_ids_of_dams()
            ]).astype(np.float32)
        elif self.config.feature_extractor == 'CNN':
            # Remember convolutional feature extractors need (Channels x Height x Width) -> (Dams x Lookback x Features)
            # This means the axes in the array Dams x Features x Lookback should be replaced from (0, 1, 2) to (0, 2, 1)
            # So observation is now a 3d array
            raw_obs = np.transpose(np.array([
                [
                    self.get_feature(feature, dam_id, values)
                    for feature in self.config.features
                ]
                for dam_id in self.instance.get_ids_of_dams()
            ]), (0, 2, 1)).astype(np.float32)
        else:
            raise NotImplementedError(f"Feature extractor {self.config.feature_extractor} is not supported yet.")

        if values is None and self.update_observation_record:
            self.record_raw_obs.append(raw_obs.flatten())

        return raw_obs

    def normalize(self, raw_obs: np.ndarray) -> np.ndarray:

        """
        Returns the normalization of the given observation of the agent

        :return: Normalized observation array
        """

        normalized_obs = (raw_obs - self.obs_min) / (self.obs_max - self.obs_min)
        if self.update_observation_record:
            self.record_normalized_obs.append(normalized_obs.flatten())

        return normalized_obs

    def project(self, normalized_obs: np.ndarray) -> np.ndarray:

        """
        Returns the projection of the given normalized observation of the agent

        :return: Projected observation array
        """

        projected_obs = self.projector.project(normalized_obs)
        if self.update_observation_record:
            self.record_projected_obs.append(projected_obs.flatten())

        return projected_obs

    def print_obs(self, obs: np.ndarray, decimals: int = 2, spacing: int = 16):

        """
        Print the given raw, normalized, or projected observation
        """

        # Raw or normalized observation
        if obs.shape == self.obs_shape:
            max_sight = max(self.config.num_steps_sight.values())
            for dam_id in self.instance.get_ids_of_dams():
                # Header
                print(f"Observation for {dam_id}:")
                print(''.join([f"{feature:^{spacing}}" for feature in self.config.features]))
                # Rows
                for time_step in range(max_sight):
                    print(''.join([
                        f"{obs[self.obs_indeces[dam_id, feature, time_step]]:^{spacing}.{decimals}f}"
                        if self.config.num_steps_sight[feature, dam_id] > time_step and (
                            self.instance.get_order_of_dam(dam_id) == 1 or feature not in self.config.unique_features
                        )
                        else f"{'-':^{spacing}}"
                        for feature in self.config.features
                    ]))

        # Projected observation
        elif obs.shape == self.projected_obs_shape:
            # Header
            print(f"Projected observation:")
            print(''.join([f"{f'Feature{i}':^{spacing}}" for i in range(self.projector.n_components)]))
            # Rows
            print(''.join([f"{obs[i]:^{spacing}.{decimals}f}" for i in range(self.projector.n_components)]))

        else:
            raise ValueError(
                f"The given observation array shape, {obs.shape}, does not match"
                f"the raw/normalized shape {self.obs_shape} nor the projected shape {self.projected_obs_shape}."
            )

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
            dam.all_previous_variations[-self.config.flow_smoothing - 1: -1] * dam.all_previous_variations[
                -1] < - epsilon
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

    def is_done(self) -> bool:

        """
        Indicates whether the environment finished
        """

        return self.river_basin.time >= self.instance.get_largest_impact_horizon() - 1

    def step(self, action_block: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:

        """
        Updates the river basin with the given action

        :param action_block: An array of size (num_dams * num_actions_block,)
            whose meaning depends on the type of action in the configuration
        :return: The next observation, the reward obtained, and whether the episode is finished or not
        """

        action_block = action_block.reshape(self.config.num_actions_block, self.instance.get_num_dams())
        reward = 0.
        flows_block = []

        # Execute all actions of the block sequentially
        for action in action_block:

            # Transform action to flows
            # Remember action is bounded between -1 and 1 in any case
            if self.config.action_type == "exiting_relvars":
                old_flows = self.river_basin.get_clipped_flows().reshape(-1)
                new_flows = old_flows + action * self.max_flows  # noqa
            else:
                new_flows = (action + 1.) / 2. * self.max_flows
            flows_block.append(new_flows)

            self.river_basin.update(new_flows.reshape(-1, 1))
            reward += self.get_reward()

            # Do not execute whole block of action if episode finishes in the middle
            # This happens when largest_impact_horizon % num_actions_block != 0 (e.g., with action A111)
            if self.is_done():
                break

        flows_block = np.array(flows_block).reshape(-1)  # Give it the same shape as the original action array
        next_raw_obs = self.get_obs_array()
        next_normalized_obs = self.normalize(next_raw_obs)
        next_projected_obs = self.project(next_normalized_obs)
        done = self.is_done()

        return next_projected_obs, reward, done, False, dict(
            flow=flows_block,
            raw_obs=next_raw_obs,
            normalized_obs=next_normalized_obs,
        )

    @staticmethod
    def create_instance(
            length_episodes: int,
            constants: dict | Instance,
            historical_data: pd.DataFrame,
            info_buffer_start: int = 0,
            info_buffer_end: int = 0,
            initial_row_decisions: int | datetime = None,
            instance_name: str = None
    ) -> Instance:

        """
        Create an instance from the data frame of historical data.

        :param length_episodes: Number of time steps of the episodes (including impact buffer)
        :param constants: Dictionary with the constants (e.g. constant physical characteristics of the dams)
        :param historical_data: Data frame with the time-dependent values (e.g. volume of the dams at a particular time)
        :param info_buffer_start: Number of time steps with information before the decisions must be made
        :param info_buffer_end: Number of time steps with information after the decisions have been made
        :param initial_row_decisions: If given, starts the episode in this row or datetime
        :param instance_name: Name of the instance
        """

        if isinstance(constants, dict):
            constants = Instance.from_dict(constants)

        # Incomplete instance dictionary (we will fill it throughout this method)
        data = pickle.loads(pickle.dumps(constants.to_dict(), -1))

        # Get necessary constants
        dam_ids = constants.get_ids_of_dams()
        channel_last_lags = {
            dam_id: constants.get_relevant_lags_of_dam(dam_id)[-1]
            for dam_id in dam_ids
        }

        # Impact buffer (included in the length of the episode) and information buffer (added on top of it)
        impact_buffer = max(
            [
                constants.get_relevant_lags_of_dam(dam_id)[0]
                for dam_id in constants.get_ids_of_dams()
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

        if instance_name is not None:
            data["instance_name"] = instance_name

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
