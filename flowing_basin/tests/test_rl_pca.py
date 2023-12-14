import numpy as np
from flowing_basin.core import Instance
from flowing_basin.solvers.rl import RLEnvironment, RLConfiguration
from flowing_basin.solvers.rl.feature_extractors import Projector
from matplotlib import pyplot as plt
import os
import math


def plot_histogram(obs: np.ndarray, env: RLEnvironment, title: str):

    # Raw or normalized flattened observation
    if np.prod(obs.shape[1:]) == np.prod(env.obs_shape):
        indices = env.get_obs_indices(flattened=True)
        for dam_id in env.constants.get_ids_of_dams():
            max_sight = max(env.config.num_steps_sight[feature, dam_id] for feature in env.config.features)
            num_features = len(env.config.features)
            fig, axs = plt.subplots(max_sight, num_features)
            fig.suptitle(f"Histograms of {title} for {dam_id}")
            for feature_index, feature in enumerate(env.config.features):
                for lookback in range(env.config.num_steps_sight[feature, dam_id]):
                    ax = axs[lookback, feature_index]
                    if env.constants.get_order_of_dam(dam_id) == 1 or feature not in env.config.unique_features:
                        index = indices[dam_id, feature, lookback]
                        ax.hist(obs[:, index], bins='auto')
                    ax.set_yticklabels([])  # Hide y-axis tick labels
                    ax.yaxis.set_ticks_position('none')  # Hide y-axis tick marks
                    if lookback == 0:
                        ax.set_title(feature)
            plt.tight_layout()
            plt.show()

    # Projected observation
    elif obs.shape[1:] == env.projected_obs_shape:
        n_components = env.projected_obs_shape[0]
        num_cols = math.ceil(math.sqrt(n_components))
        # We want to guarantee that
        # num_rows * num_cols > n_components ==> num_rows = math.ceil(n_components / num_cols)
        num_rows = math.ceil(n_components / num_cols)
        fig, axs = plt.subplots(num_rows, num_cols)
        fig.suptitle(f"Histograms of {title}")
        component = 0
        for row in range(num_rows):
            for col in range(num_cols):
                if component < n_components:
                    ax = axs[row, col]
                    ax.hist(obs[:, component], bins='auto')
                    ax.set_yticklabels([])  # Hide y-axis tick labels
                    ax.yaxis.set_ticks_position('none')  # Hide y-axis tick marks
                    ax.set_title(f"Component {component}")
                component += 1
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Given observation shape does not match any of the environment's shapes.")


OBSERVATION_TYPE = "O2"

FIXED = '_fixed' if OBSERVATION_TYPE == 'O1' else ''
OBSERVATIONS_FOLDER = f"reports/observations_data/observations{OBSERVATION_TYPE}{FIXED}_1000ep_"
CONSTANTS = "../data/constants/constants_2dams.json"
HISTORICAL_DATA = "../data/history/historical_data_clean.pickle"
OBSERVATIONS_JSON = f"reports/observations_data/observations{OBSERVATION_TYPE}{FIXED}_1000ep_/config.json"

constants = Instance.from_json(CONSTANTS)
config = RLConfiguration.from_json(OBSERVATIONS_JSON)

config.projector_type = ["QuantilePseudoDiscretizer", "PCA"]
config.projector_bound = "max_min_per_component"
config.projector_extrapolation = 0.5
config.projector_explained_variance = .98
observations = np.load(os.path.join(OBSERVATIONS_FOLDER, 'observations.npy'))
obs_config = RLConfiguration.from_json(os.path.join(OBSERVATIONS_FOLDER, 'config.json'))
projector = Projector.create_projector(config, observations, obs_config)

env = RLEnvironment(
    config=config,
    projector=projector,
    path_constants=CONSTANTS,
    path_historical_data=HISTORICAL_DATA,
)

plot_histogram(projector.observations, env, title=f"original observations {OBSERVATION_TYPE}")
indicate_variance = lambda proj_type: f"({config.projector_explained_variance*100:.0f}%)" if proj_type =='PCA' else ''
if isinstance(config.projector_type, list):
    for proj, proj_type in zip(projector.projectors, config.projector_type):
        plot_histogram(
            proj.transformed_observations, env,
            title=f"observations {OBSERVATION_TYPE} after applying {proj_type} {indicate_variance(proj_type)}"
        )
else:
    plot_histogram(
        projector.transformed_observations, env,
        title=f"observations {OBSERVATION_TYPE} after applying {config.projector_type} {indicate_variance(config.projector_type)}"
    )
