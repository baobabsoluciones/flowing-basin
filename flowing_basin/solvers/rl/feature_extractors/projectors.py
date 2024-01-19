from flowing_basin.solvers.rl import RLConfiguration
from sklearn.decomposition import PCA
import numpy as np
from abc import ABC, abstractmethod


class Projector(ABC):

    def __init__(self, observations: np.array = None):

        """
        Abstract class for the projector

        :param observations: Array of shape num_observations x flattened_observation_size
            with the observations that will be used to train the projector
        """

        self.observations = observations
        self.n_components = None
        self.transformed_observations = None

        # Default bounds values
        self.low = 0
        self.high = 1

    def project(self, normalized_obs: np.ndarray) -> np.ndarray:

        """
        Apply the projector to a single observation
        :param normalized_obs: Array of shape (num_features,)
        :return: Array of shape (num_components,)
        """

        return self.transform(normalized_obs.reshape(1, -1)).reshape(-1)

    @abstractmethod
    def transform(self, normalized_obs: np.ndarray) -> np.ndarray:

        """
        Apply the projector to many observations
        :param normalized_obs: Array of shape (num_observations, num_features)
        :return: Array of shape (num_observations, num_components)
        """

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
            cls, proj_type: str, proj_config: RLConfiguration = None, observations: np.ndarray = None
    ):

        """
        Select the projector based on the given configuration
        """

        if proj_type != 'identity' and any([proj_config is None, observations is None]):
            raise ValueError(
                "The parameters `proj_config` and `observations` "
                "can only be None when the projector is of type 'identity'."
            )

        proj_constructor = cls._get_projector_constructor(proj_type)
        kwargs = dict()
        if proj_type != 'identity':
            kwargs.update(dict(
                observations=observations,
            ))
        if proj_type == 'PCA':
            kwargs.update(dict(
                bounds=proj_config.projector_bound,
                extrapolation=proj_config.projector_extrapolation,
                explained_variance=proj_config.projector_explained_variance,
            ))
        projector = proj_constructor(**kwargs)

        return projector

    @classmethod
    def create_projector(cls, proj_config: RLConfiguration, observations: np.ndarray = None):

        """
        Select the projector based on the given configuration
        """

        if proj_config.projector_type != 'identity' and observations is None:
            raise ValueError(
                "The parameter `observations` can only be None when the projector is of type 'identity'."
            )

        if isinstance(proj_config.projector_type, list):
            projector = ProjectorList(proj_config, observations)
        else:
            projector = cls.create_projector_single(proj_config.projector_type, proj_config, observations)

        return projector


class ProjectorList(Projector):

    def __init__(self, proj_config: RLConfiguration, observations: np.ndarray):

        """
        Initialize the quantile-based pseudo-discretizer
        """

        super(ProjectorList, self).__init__(observations)

        self.projectors = []
        prev_transformed_obs = self.observations
        for proj_type in proj_config.projector_type:
            projector = self.create_projector_single(proj_type, proj_config, prev_transformed_obs)
            self.projectors.append(projector)
            prev_transformed_obs = projector.transformed_observations

        last_projector = self.projectors[-1]
        self.n_components = last_projector.n_components
        self.transformed_observations = last_projector.transformed_observations
        self.low = last_projector.low
        self.high = last_projector.high

    def transform(self, normalized_obs: np.ndarray) -> np.ndarray:

        """
        Apply the projectors sequentially to many observations
        :param normalized_obs: Array of shape (num_observations, num_features)
        """

        transformed_obs = normalized_obs.copy()
        for projector in self.projectors:
            transformed_obs = projector.transform(transformed_obs)

        return transformed_obs


class IdentityProjector(Projector):

    def __init__(self):

        # Since the IdentityProjector does not change the observations,
        # the lower and higher bounds are just the original 0 and 1
        super(IdentityProjector, self).__init__()

    def transform(self, normalized_obs: np.ndarray) -> np.ndarray:
        return normalized_obs


class PCAProjector(Projector):

    def __init__(
            self, observations: np.ndarray, explained_variance: float,
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

        super(PCAProjector, self).__init__(observations)

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

    def transform(self, normalized_obs: np.ndarray) -> np.ndarray:

        """
        Apply the PCA to many observations
        :param normalized_obs: Array of shape (num_observations, num_features)
        """

        transformed_obs = self.model.transform(normalized_obs)
        clipped_obs = np.clip(transformed_obs, self.low, self.high, dtype=np.float32)
        return clipped_obs


class QuantilePseudoDiscretizer(Projector):

    def __init__(self, observations: np.ndarray, num_quantiles: int = 100):

        """
        Initialize the quantile-based pseudo-discretizer
        """

        super(QuantilePseudoDiscretizer, self).__init__(observations)

        # Compute the quantiles
        self.num_quantiles = num_quantiles
        self.quantiles = np.quantile(self.observations, q=np.linspace(0, 1, self.num_quantiles), axis=0)

        # Compute the transformed observations using these quantiles
        self.transformed_observations = self.transform(self.observations)
        self.n_components = self.observations.shape[1]

    def transform(self, normalized_obs: np.ndarray) -> np.ndarray:

        """
        Use the quantiles to pseudo-discretize many observations
        :param normalized_obs: Array of shape (num_observations, num_features)
        """

        # Get the quantile of each value
        transformed_obs = np.empty(normalized_obs.shape)
        num_features = normalized_obs.shape[1]
        for f in range(num_features):
            transformed_obs[:, f] = np.digitize(normalized_obs[:, f], bins=self.quantiles[:, f])

        # Bring back to the range [0, 1]
        transformed_obs = transformed_obs / self.num_quantiles

        return transformed_obs
