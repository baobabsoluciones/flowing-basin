from flowing_basin.core import BaseConfiguration, Configuration
from dataclasses import dataclass, asdict
import json
import warnings


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


@dataclass(kw_only=True)
class ObservationConfiguration(BaseConfiguration):  # noqa

    features: list[str]  # Features included in the observation
    unique_features: list[str]  # Features that should NOT be repeated for each dam
    num_steps_sight: dict[tuple[str, str] | str, int] | int  # Number of time steps for every (feature, dam_id)
    projector_type: str | list[str]  # Type of dimensionality reduction to apply for the observations (or identity)
    feature_extractor: str  # Either MLP or CNN or mixed

    # Data required to post-process config
    dam_ids: list[str]

    # Optional RL environment's observation options
    projector_bound: tuple[int, int] | str = None  # bounds of the projected observations
    projector_extrapolation: float = None  # percentage of the bounds that are allowed to be exceeded by the projector
    projector_explained_variance: float = None

    def to_dict(self) -> dict:

        """
        Turn the dataclass into a JSON-serializable dictionary
        """

        data_json = asdict(self)

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
            "past_vols", "past_flows", "past_variations", "past_prices", "future_prices", "past_inflows",
            "future_inflows", "past_turbined", "past_groups", "past_powers", "past_clipped", "past_periods"
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


@dataclass(kw_only=True)
class ActionConfiguration(BaseConfiguration):  # noqa

    # RL environment's action options
    action_type: str

    def check(self):

        super(ActionConfiguration, self).check()

        # Check self.action_type
        valid_actions = {"exiting_flows", "exiting_relvars"}
        if self.action_type not in valid_actions:
            raise ValueError(f"Invalid value for 'action_type': {self.action_type}. Allowed values are {valid_actions}")


@dataclass(kw_only=True)
class RewardConfiguration(BaseConfiguration):  # noqa

    flow_smoothing_penalty: float  # Penalty for not fulfilling the flow smoothing parameter


@dataclass(kw_only=True)
class TrainingConfiguration(BaseConfiguration):  # noqa

    length_episodes: int
    num_timesteps: int = None  # Nuber of time steps in which to train the agent
    log_episode_freq: int = None

    training_data_callback: bool = None
    evaluation_callback: bool = None
    checkpoint_callback: bool = None
    # TODO: once all have been cleaned, make all arguments above this comment mandatory (i.e., without `= None`)

    # Training data: periodic evaluation in fixed instances
    training_data_timesteps_freq: int = None  # Frequency for evaluating the agent (every X timesteps)
    training_data_instances: list[str] = None  # Names of the instances solved every time

    # Periodic evaluation in random episodes
    evaluation_timesteps_freq: int = None  # Frequency for evaluating the agent (every X timesteps)
    evaluation_num_episodes: int = None  # Number of episodes run every time

    # Checking if there is a new best agent
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