from flowing_basin.solvers.rl import RLConfiguration
from sklearn.decomposition import PCA
import numpy as np
from abc import ABC, abstractmethod


class Projector(ABC):

    def __init__(self, observations: np.array = None, obs_config: RLConfiguration = None):

        """
        Abstract class for the projector

        :param observations: The observation NumPy arrays which will be used to train the projector
        :param obs_config: The configuration used when generating the observations
            (which must match the current configuration in some fields)
        """

        self.observations = observations
        self.obs_config = obs_config

        self.n_components = None
        self.transformed_observations = None

        # Default bounds values
        self.low = 0
        self.high = 1

    def check_config_attribute(self, other_config: RLConfiguration, attribute: str):

        assert getattr(self.obs_config, attribute) == getattr(other_config, attribute), (
            f"The {attribute} considered in both configurations must be the same, "
            f"but the observations used by the projector has {getattr(self.obs_config, attribute)} "
            f"and the given configuration has {getattr(other_config, attribute)}"
        )

    def check_config(self, other_config: RLConfiguration):

        if self.obs_config is not None:
            observation_attributes = ["features", "unique_features", "num_steps_sight"]
            for attribute in observation_attributes:
                self.check_config_attribute(other_config, attribute)

    @abstractmethod
    def project(self, normalized_obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def _get_projector_constructor(proj_type: str):
        if proj_type == 'identity':
            return IdentityProjector
        elif proj_type == 'PCA':
            return PCAProjector
        elif proj_type == 'QuantilePseudoDiscretizer':
            return QuantilePseudoDiscretizer
        else:
            raise NotImplementedError(f"Projector {proj_type} is not supported yet.")

    @classmethod
    def create_projector_single(
            cls, proj_type: str, proj_config: RLConfiguration, observations: np.ndarray, obs_config: RLConfiguration
    ):

        """
        Select the projector based on the given configuration
        """

        proj_constructor = cls._get_projector_constructor(proj_type)
        if proj_type == 'PCA':
            kwargs = dict(
                bounds=proj_config.projector_bound,
                extrapolation=proj_config.projector_extrapolation,
                explained_variance=proj_config.projector_explained_variance,
            )
        else:
            kwargs = dict()

        projector = proj_constructor(
            observations=observations,
            obs_config=obs_config,
            **kwargs
        )

        # Check the configuration and the observation's configuration matches in the required fields
        projector.check_config(proj_config)

        return projector

    @classmethod
    def create_projector(cls, proj_config: RLConfiguration, observations: np.ndarray, obs_config: RLConfiguration):

        """
        Select the projector based on the given configuration
        """

        if isinstance(proj_config.projector_type, list):
            projector = ProjectorList(proj_config, observations, obs_config)
        else:
            projector = cls.create_projector_single(proj_config.projector_type, proj_config, observations, obs_config)

        return projector


class ProjectorList(Projector):

    def __init__(self, proj_config: RLConfiguration, observations: np.ndarray, obs_config: RLConfiguration):

        """
        Initialize the quantile-based pseudo-discretizer
        """

        super(ProjectorList, self).__init__(
            observations=observations, obs_config=obs_config
        )

        self.projectors = []
        prev_transformed_obs = self.observations
        for proj_type in proj_config.projector_type:
            projector = self.create_projector_single(proj_type, proj_config, prev_transformed_obs, self.obs_config)
            self.projectors.append(projector)
            prev_transformed_obs = projector.transformed_observations

        last_projector = self.projectors[-1]
        self.n_components = last_projector.n_components
        self.transformed_observations = last_projector.transformed_observations
        self.low = last_projector.low
        self.high = last_projector.high

    def project(self, normalized_obs: np.ndarray) -> np.ndarray:

        transformed_obs = normalized_obs.copy()
        for projector in self.projectors:
            transformed_obs = projector.project(transformed_obs)

        return transformed_obs


class IdentityProjector(Projector):

    def __init__(self):

        # Since the IdentityProjector does not change the observations,
        # the lower and higher bounds are just the original 0 and 1
        super(IdentityProjector, self).__init__()

    def project(self, normalized_obs: np.ndarray) -> np.ndarray:
        return normalized_obs


class PCAProjector(Projector):

    def __init__(
            self, observations: np.ndarray, obs_config: RLConfiguration, explained_variance: float,
            bounds: tuple[float, float] | str, extrapolation: float
    ):

        """
        Initialize the PCA projector

        :param explained_variance: Explained variance of the resulting PCA model
        :param bounds: The bounds of the normalized observations.
            They can be manually set with a (min, max) tuple, or
            inferred from the provided observations (by passing the str 'max_min_per_component' or 'max_min_all')
        :param extrapolation: The degree of extrapolation allowed, as a fraction of the original bounds
        """

        super(PCAProjector, self).__init__(observations, obs_config)

        # Fitted model
        self.model = PCA(n_components=explained_variance)
        self.model.fit(self.observations)

        # Bounds
        self.transformed_observations = self.model.transform(self.observations)
        if bounds == 'max_min_per_component':
            self.low = np.min(self.transformed_observations, axis=0).astype(np.float32)
            self.high = np.max(self.transformed_observations, axis=0).astype(np.float32)
        elif bounds == 'max_min_all':
            self.low = np.min(self.transformed_observations).astype(np.float32)
            self.high = np.max(self.transformed_observations).astype(np.float32)
        elif isinstance(bounds, tuple):
            self.low = bounds[0]
            self.high = bounds[1]
        else:
            raise ValueError(
                f"Invalid value for `bounds`: {bounds}. "
                f"Allowed values are 'max_min_per_component', 'max_min_all', or a tuple with (min, max) floats."
            )

        # Allow some degree of extrapolation
        self.low = self.low - np.abs(self.low) * extrapolation
        self.high = self.high + np.abs(self.high) * extrapolation

        # Number of components
        self.n_components = self.model.n_components_

    def project(self, normalized_obs: np.ndarray) -> np.ndarray:
        transformed_obs = self.model.transform(normalized_obs.reshape(1, -1)).reshape(-1)
        clipped_obs = np.clip(transformed_obs, self.low, self.high, dtype=np.float32)
        return clipped_obs


class QuantilePseudoDiscretizer(Projector):

    def __init__(self, observations: np.ndarray, obs_config: RLConfiguration, num_quantiles: int = 100):

        """
        Initialize the quantile-based pseudo-discretizer
        """

        super(QuantilePseudoDiscretizer, self).__init__(
            observations=observations, obs_config=obs_config
        )

        # Compute the quantiles
        self.num_quantiles = num_quantiles
        self.quantiles = np.quantile(self.observations, q=np.linspace(0, 1, self.num_quantiles), axis=0)

        # Compute the transformed observations
        self.transformed_observations = np.empty(self.observations.shape)
        num_features = self.observations.shape[1]
        for f in range(num_features):
            self.transformed_observations[:, f] = np.digitize(self.observations[:, f], bins=self.quantiles[:, f])
        self.transformed_observations = self.transformed_observations / self.num_quantiles

    def project(self, normalized_obs: np.ndarray) -> np.ndarray:

        """
        Use the quantiles to pseudo-discretize the observation
        """

        # Get the quantile of each value
        digitalized_obs = np.array([
            np.digitize(feature_value, bins=self.quantiles[:, feature_index])
            for feature_index, feature_value in enumerate(normalized_obs)
        ], dtype=np.float32)

        # Bring back to the range [0, 1]
        digitalized_obs = digitalized_obs / self.num_quantiles

        return digitalized_obs
