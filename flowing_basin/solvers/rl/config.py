from flowing_basin.core import BaseConfiguration, Configuration
from dataclasses import dataclass, asdict, field
import json
import warnings
import numpy as np


@dataclass(kw_only=True)
class GeneralConfiguration(Configuration):  # noqa

    flow_smoothing_clip: bool  # Whether to clip the actions that do not comply with flow smoothing or not
    flow_smoothing: int = 0

    mode: str = "nonlinear"
    do_history_updates: bool = True

    def check(self):

        super(GeneralConfiguration, self).check()

        # Check self.mode
        valid_modes = {"linear", "nonlinear"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid value for 'mode': {self.mode}. Allowed values are {valid_modes}")


# noinspection PyDataclass
@dataclass(kw_only=True)
class ObservationConfiguration(BaseConfiguration):  # noqa

    features: list[str]  # Features included in the observation
    unique_features: list[str]  # Features that should NOT be repeated for each dam
    num_steps_sight: dict[tuple[str, str] | str, int] | int  # Number of time steps for every (feature, dam_id)
    feature_extractor: str  # Either MLP or CNN or mixed

    # Projector
    projector_type: str | list[str]  # Type of dimensionality reduction to apply for the observations (or identity)
    projector_bound: tuple[int, int] | str = None  # Bounds of the projected observations
    projector_extrapolation: float = None  # Percentage of the bounds that are allowed to be exceeded by the projector
    projector_explained_variance: float = None

    # Randomization (for testing purposes)
    obs_random: bool = False
    obs_random_features_excluded: list[str] = field(default_factory=list)  # By default, an empty list

    # Data required to post-process config
    dam_ids: list[str]

    def to_dict(self) -> dict:

        """
        Turn the original dataclass into a JSON-serializable dictionary
        """

        data_json = super(ObservationConfiguration, self).to_dict()

        # Convert 'num_steps_sight' into a list of (key, value) pairs
        data_json["num_steps_sight"] = [{"key": k, "value": v} for k, v in data_json["num_steps_sight"].items()]

        return data_json

    @classmethod
    def from_json(cls, path: str):

        with open(path, "r") as f:
            data_json = json.load(f)

        # Convert 'num_steps_sight' back into a dictionary
        num_steps_sight_list = data_json.pop("num_steps_sight")
        num_steps_sight_dict = {tuple(item["key"]): item["value"] for item in num_steps_sight_list}
        data_json["num_steps_sight"] = num_steps_sight_dict

        return cls(**data_json)

    def post_process(self):

        """
        Turn all keys of 'num_steps_sight' into (feature, dam_id)
        (i.e., remove any "other" or other short-hand values)
        """

        # Post-process 'num_steps_sight'
        # Turn all keys into (feature, dam_id)
        if isinstance(self.num_steps_sight, int):
            self.num_steps_sight = {
                (feature, dam_id): self.num_steps_sight
                for feature in self.features for dam_id in self.dam_ids
            }
        if "other" in self.num_steps_sight.keys():
            self.num_steps_sight = {
                (feature, dam_id):
                    self.num_steps_sight[feature, dam_id] if (feature, dam_id) in self.num_steps_sight.keys() else (
                        self.num_steps_sight[feature] if feature in self.num_steps_sight.keys() else
                        self.num_steps_sight["other"]
                    )
                for feature in self.features for dam_id in self.dam_ids
            }

    def check(self):

        """
        Raises an error if data is not consistent
        """

        super(ObservationConfiguration, self).check()

        # Check self.features
        valid_features = {
            "past_vols", "past_flows", "past_flows_raw", "past_variations", "past_prices",
            "future_prices", "past_inflows", "future_inflows", "past_turbined", "past_groups",
            "past_powers", "past_clipped", "past_periods"
        }
        for feature in self.features:
            if feature not in valid_features:
                raise ValueError(f"Invalid feature: {feature}. Allowed features are {valid_features}")

        # Check self.unique_features
        if not set(self.unique_features).issubset(self.features):
            raise ValueError(
                f"The features marked as unique, {self.unique_features}, "
                f"should be a subset of the features, {self.features}"
            )
        if self.feature_extractor == 'CNN' and len(self.unique_features) != 0:
            raise ValueError(
                f"Feature extractor is CNN (so observation must be box shaped) "
                f"but there are features marked as unique: {self.unique_features}"
            )

        # Check self.num_steps_sight
        if isinstance(self.num_steps_sight, dict):
            flattened_keys = [
                key
                for tup_or_key in self.num_steps_sight.keys()
                for key in (tup_or_key if isinstance(tup_or_key, tuple) else (tup_or_key,))
            ]
            if not set(self.features).issubset(flattened_keys) and "other" not in flattened_keys:
                raise ValueError(
                    f"The features provided are {self.features}, but not all of them "
                    f"have a number of time steps defined: {self.num_steps_sight} "
                    f"(suggestion: use the key 'other' as a wildcard for the remaining features)"
                )
            if self.feature_extractor == 'CNN' and len(set(self.num_steps_sight.values())) != 1:
                raise ValueError(
                    f"Feature extractor is CNN (so observation must be box shaped) but the number of time steps "
                    f"of every feature is not the same: {self.num_steps_sight}"
                )

        # Check self.projector_type
        valid_projectors = {'identity', 'PCA', 'QuantilePseudoDiscretizer'}
        projector_types = self.projector_type if isinstance(self.projector_type, list) else [self.projector_type]
        for projector_type in projector_types:
            if projector_type not in valid_projectors:
                raise ValueError(
                    f"Invalid value for 'projector_type': {projector_type}. Allowed values are {valid_projectors}"
                )
        if self.projector_type != 'identity' and self.feature_extractor == 'CNN':
            raise ValueError(
                f"Cannot use projector `{self.projector_type}` with the CNN feature extractor, "
                f"since projectors flatten the observations"
            )

        # Check projector options are given if required
        projector_options_not_provided = [
            self.projector_extrapolation is None, self.projector_bound is None, self.projector_explained_variance is None
        ]
        if self.projector_type == 'identity':
            if not all(projector_options_not_provided):
                warnings.warn(
                    "Projector type is `identity`, but projector options where provided. These will be ignored."
                )
        else:
            if any(projector_options_not_provided):
                raise ValueError(
                    f"Projector type is `{self.projector_type}` but not all projector options were provided."
                )

        # Check self.projector_bound
        if isinstance(self.projector_bound, str):
            valid_bounds = {'max_min_per_component', 'max_min_all'}
            if self.projector_bound not in valid_bounds:
                raise ValueError(
                    f"Invalid value for 'projector_bound': {self.projector_bound}. Allowed values are {valid_bounds}"
                )

        # Check self.feature_extractor
        valid_extractors = {'MLP', 'CNN', 'mixed'}
        if self.feature_extractor not in valid_extractors:
            raise ValueError(
                f"Invalid value for 'feature_extractor': {self.feature_extractor}. "
                f"Allowed values are {valid_extractors}"
            )

        if len(self.obs_random_features_excluded) > 0 and not self.obs_random:
            warnings.warn(
                f"There are features excluded from the randomization: {self.obs_random_features_excluded}, "
                f"but randomization is not turned on. This will be ignored."
            )


@dataclass(kw_only=True)
class ActionConfiguration(BaseConfiguration):  # noqa

    action_type: str
    num_actions_block: int = 1  # By default, the agent only gives the actions for the current timestep

    def check(self):

        super(ActionConfiguration, self).check()

        # Check self.action_type
        valid_actions = {"exiting_flows", "exiting_relvars", "adjustments"}
        if self.action_type not in valid_actions:
            raise ValueError(f"Invalid value for 'action_type': {self.action_type}. Allowed values are {valid_actions}")


@dataclass(kw_only=True)
class RewardConfiguration(BaseConfiguration):  # noqa

    # noinspection PyUnresolvedReferences
    """

    :param flow_smoothing_penalty:
        Penalty for not fulfilling the flow smoothing parameter
    :param greedy_reference:
        If True, use rl-greedy's average performance on the episode as a reference to compute the reward
    :param reference_ratio:
        Can only be None when greedy_reference is False
        If False, `reward = rew_agent - max(0, avg_rew_greedy)`;
        if True, `reward = (rew_agent - max(0., avg_rew_greedy)) / max(1., avg_rew_greedy)`
    """

    flow_smoothing_penalty: float
    greedy_reference: bool = False
    reference_ratio: bool = None

    def check(self):

        super(RewardConfiguration, self).check()

        # Check self.reference_ratio
        if self.greedy_reference:
            if self.reference_ratio is None:
                raise ValueError(
                    f"Reference ratio must be specified if {self.greedy_reference=}, but {self.reference_ratio=}."
                )


# noinspection PyDataclass
@dataclass(kw_only=True)
class TrainingConfiguration(BaseConfiguration):  # noqa

    length_episodes: int
    num_timesteps: int  # Nuber of time steps in which to train the agent
    learning_rate: float = 3e-4
    replay_buffer_size: int = 1_000_000

    # Monitor logging: reward of episodes seen during training
    log_episode_freq: int

    # Training data callback: periodic evaluation in fixed instances
    training_data_callback: bool
    training_data_timesteps_freq: int = None  # Frequency for evaluating the agent (every X timesteps)
    training_data_instances: list[str] = None  # Names of the instances solved every time

    # Evaluation callback: periodic evaluation in random episodes
    evaluation_callback: bool
    evaluation_timesteps_freq: int = None  # Frequency for evaluating the agent (every X timesteps)
    evaluation_num_episodes: int = None  # Number of episodes run every time
    evaluation_save_best: bool = False  # Whether to save the model with the highest mean reward

    # Checking callback: checking if there is a new best agent
    checkpoint_callback: bool
    checkpoint_timesteps_freq: int = None  # Frequency for checking if there is a new best agent (every X timesteps)

    def check(self):

        super(TrainingConfiguration, self).check()

        required_args_not_given = [self.training_data_timesteps_freq is None, self.training_data_instances is None]
        if self.training_data_callback:
            if any(required_args_not_given):
                raise ValueError(
                    f"Some of the required arguments when {self.training_data_callback=} where not given."
                )
        else:
            if not all(required_args_not_given):
                warnings.warn(
                    f"Some of the arguments for {self.training_data_callback=} where given, even though it is False. "
                    f"These values will be ignored."
                )

        required_args_not_given = [self.evaluation_timesteps_freq is None, self.evaluation_num_episodes is None]
        if self.evaluation_callback:
            if any(required_args_not_given):
                raise ValueError(
                    f"Some of the required arguments when {self.evaluation_callback=} where not given."
                )
        else:
            if not all(required_args_not_given):
                warnings.warn(
                    f"Some of the arguments for {self.evaluation_callback=} where given, even though it is False. "
                    f"These values will be ignored."
                )

        if self.checkpoint_callback:
            if self.checkpoint_timesteps_freq is None:
                raise ValueError(
                    f"When {self.checkpoint_callback=}, then {self.checkpoint_timesteps_freq=} must not be None."
                )
        else:
            if self.checkpoint_timesteps_freq is not None:
                warnings.warn(
                    f"We have {self.checkpoint_callback=}, but {self.checkpoint_timesteps_freq=} is not None. "
                    f"This value will be ignored."
                )


@dataclass(kw_only=True)
class RLConfiguration(GeneralConfiguration, ObservationConfiguration, ActionConfiguration, RewardConfiguration, TrainingConfiguration):  # noqa

    @classmethod
    def from_components(cls, configs: list[BaseConfiguration]):

        config_dicts = [asdict(config) for config in configs]
        result_dict = {}
        for config_dict in config_dicts:
            result_dict.update(config_dict)
        return cls(**result_dict)

    def check(self):

        """
        Raises an error if data is not consistent
        """

        GeneralConfiguration.check(self)
        ObservationConfiguration.check(self)
        ActionConfiguration.check(self)
        RewardConfiguration.check(self)
        TrainingConfiguration.check(self)

        if self.action_type == "adjustments" and self.num_actions_block != self.length_episodes:
            raise ValueError(
                f"With {self.action_type=}, the block size should be equal to {self.length_episodes=}, "
                f"but it is actually {self.num_actions_block=}."
            )

    def post_process(self):

        """
        Extend the sight of all features
        by the number of additional periods per action block.
        """

        GeneralConfiguration.post_process(self)
        ObservationConfiguration.post_process(self)
        ActionConfiguration.post_process(self)
        RewardConfiguration.post_process(self)
        TrainingConfiguration.post_process(self)

        for feature in self.features:
            for dam_id in self.dam_ids:
                self.num_steps_sight[feature, dam_id] += self.num_actions_block - 1
        
    def get_obs_indices(self, flattened: bool = False) -> dict[tuple[str, str, int], int | tuple[int]]:

        """
        Returns the index in the raw or normalized observation array for any dam, feature, and lookback value

        :return: Index in the array
        """

        if flattened and self.feature_extractor != 'CNN':
            warnings.warn(f"Setting {flattened=} has no effect when {self.feature_extractor=}")

        indices = dict()
        running_index = 0
        for d, dam_id in enumerate(self.dam_ids):
            for f, feature in enumerate(self.features):
                if d == 0 or feature not in self.unique_features:
                    for t in range(self.num_steps_sight[feature, dam_id]):
                        if self.feature_extractor == 'MLP':
                            indices[dam_id, feature, t] = running_index
                        elif self.feature_extractor == 'CNN':
                            # Remember the convolutional feature extractor considers (Dams x Lookback x Features)
                            if not flattened:
                                indices[dam_id, feature, t] = (d, t, f)
                            else:
                                indices[dam_id, feature, t] = np.ravel_multi_index(
                                    (d, t, f),
                                    dims=(len(self.dam_ids), self.num_steps_sight[feature, dam_id],
                                          len(self.features))
                                )
                        else:
                            raise NotImplementedError(
                                f"Feature extractor {self.feature_extractor} is not supported yet."
                            )
                        running_index += 1

        return indices

    def check_attribute_equal(self, other: "RLConfiguration", attribute: str) -> str | None:

        """
        Check if the current and given configurations have the same value for the given attribute

        :return: Error message, if the value is not the same; None otherwise
        """

        error_msg = None
        if getattr(self, attribute) != getattr(other, attribute):
            error_msg = (
                f"The {attribute} considered in both configurations must be the same, "
                f"but the current configuration has {getattr(self, attribute)} "
                f"and the other configuration has {getattr(other, attribute)}"
            )
        return error_msg

    def check_observation_compatibility(self, other: "RLConfiguration") -> list[str]:

        """
        Check if the current and given configurations have compatible observations

        :return: List with error messages, if configurations are not observation-compatible.
        """

        observation_attributes = ["features", "unique_features", "num_steps_sight", "num_actions_block"]

        errors = []
        for attribute in observation_attributes:
            error_msg = self.check_attribute_equal(other, attribute)
            if error_msg is not None:
                errors.append(error_msg)

        return errors
