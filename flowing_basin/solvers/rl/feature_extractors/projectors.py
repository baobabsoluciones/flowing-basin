from flowing_basin.solvers.rl import RLConfiguration
from sklearn.decomposition import PCA
import numpy as np
from abc import ABC, abstractmethod
import os


class Projector(ABC):

    def __init__(
        self, bounds: tuple[float, float] | str = None, extrapolation: float = 0., path_observations_folder: str = None
    ):

        """
        Initialize the projector

        :param bounds: The bounds of the normalized observations.
            They can be manually set with a (min, max) tuple, or
            inferred from the provided observations (by passing the str 'max_min_per_component' or 'max_min_all')
        :param path_observations_folder: A folder with
            1) the observation NumPy arrays 'observations.npy'
            2) the configuration used to generate them 'obs_config.json'
        """

        # Open folder
        if path_observations_folder is not None:
            self.observations = np.load(os.path.join(path_observations_folder, 'observations.npy'))
            self.obs_config = RLConfiguration.from_json(os.path.join(path_observations_folder, 'config.json'))
        else:
            self.observations = None
            self.obs_config = None
        self.n_components = None

        # Check the bounds can be inferred from data if required
        if self.observations is None and not isinstance(bounds, tuple):
            raise ValueError(
                f"Bounds is `{bounds}`, "
                f"but no observation data was provided."
            )

        # Define bounds
        if isinstance(bounds, tuple):
            self.low = bounds[0]
            self.high = bounds[1]
        else:
            self.low = None
            self.high = None
        self.extrapolation = extrapolation

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


class IdentityProjector(Projector):

    def __init__(self):

        # Since the IdentityProjector does not change the observations,
        # the lower and higher bounds are just the original 0 and 1
        super(IdentityProjector, self).__init__(bounds=(0., 1.))

    def project(self, normalized_obs: np.ndarray) -> np.ndarray:
        return normalized_obs


class PCAProjector(Projector):

    def __init__(
            self, bounds: tuple[float, float] | str, extrapolation: float, path_observations_folder: str,
            explained_variance: float
    ):

        """
        Initialize the PCA projector

        :param explained_variance: Explained variance of the resulting PCA model
        """

        super(PCAProjector, self).__init__(bounds, extrapolation, path_observations_folder)

        # Fitted model
        self.model = PCA(n_components=explained_variance)
        self.model.fit(self.observations)

        # Bounds
        transformed_observations = self.model.transform(self.observations)
        if bounds == 'max_min_per_component':
            self.low = np.min(transformed_observations, axis=0).astype(np.float32)
            self.high = np.max(transformed_observations, axis=0).astype(np.float32)
        elif bounds == 'max_min_all':
            self.low = np.min(transformed_observations).astype(np.float32)
            self.high = np.max(transformed_observations).astype(np.float32)
        # else, self.low and self.high should be the specified bounds, set in parent class

        # Allow some degree of extrapolation
        self.low = self.low - np.abs(self.low) * self.extrapolation
        self.high = self.high + np.abs(self.high) * self.extrapolation

        # Number of components
        self.n_components = self.model.n_components_

    def project(self, normalized_obs: np.ndarray) -> np.ndarray:
        transformed_obs = self.model.transform(normalized_obs.reshape(1, -1)).reshape(-1)
        clipped_obs = np.clip(transformed_obs, self.low, self.high, dtype=np.float32)
        return clipped_obs
