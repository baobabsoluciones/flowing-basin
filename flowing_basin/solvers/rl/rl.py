from flowing_basin.core import Instance
from flowing_basin.solvers.rl import (
    GeneralConfiguration, ObservationConfiguration, ActionConfiguration, RewardConfiguration, TrainingConfiguration,
    RLConfiguration, RLTrain, RLEnvironment
)
from flowing_basin.solvers.rl.feature_extractors import Projector
from cornflow_client.core.tools import load_json
import numpy as np
import math
from matplotlib import pyplot as plt
import os
import re
import warnings


class ReinforcementLearning:

    # RL data
    configs_info = {
        "G": (os.path.join(os.path.dirname(__file__), "../../rl_data/configs/general"), GeneralConfiguration),
        "O": (os.path.join(os.path.dirname(__file__), "../../rl_data/configs/observation"), ObservationConfiguration),
        "A": (os.path.join(os.path.dirname(__file__), "../../rl_data/configs/action"), ActionConfiguration),
        "R": (os.path.join(os.path.dirname(__file__), "../../rl_data/configs/reward"), RewardConfiguration),
        "T": (os.path.join(os.path.dirname(__file__), "../../rl_data/configs/training"), TrainingConfiguration)
    }
    observation_records_folder = os.path.join(os.path.dirname(__file__), "../../rl_data/observation_records")

    # General data
    constants_path = os.path.join(os.path.dirname(__file__), "../../data/constants/constants_2dams.json")
    train_data_path = os.path.join(os.path.dirname(__file__), "../../data/history/historical_data_clean_train.pickle")
    test_data_path = os.path.join(os.path.dirname(__file__), "../../data/history/historical_data_clean_test.pickle")

    models_folder = os.path.join(os.path.dirname(__file__), "../../rl_data/models")

    def __init__(self, config_name: str, verbose: int = 1):

        self.verbose =  verbose
        config_name = config_name
        self.config_names = self.extract_substrings(config_name)
        self.config_full_name = ''.join(self.config_names.values())  # Identical to `config_name`, but in alphabetical order

        self.config = self.get_config(self.config_names)
        self.agent_path = os.path.join(
            ReinforcementLearning.models_folder, f"rl-{''.join(self.config_full_name)}"
        )
        self.constants = Instance.from_dict(load_json(ReinforcementLearning.constants_path))

        # The first two digits in the observation name (e.g., "O211" -> "O21")
        # indicate the type of observations that should be used for the projector
        self.obs_records_path = os.path.join(
            ReinforcementLearning.observation_records_folder, self.config_names["O"][0:3]
        )

    def train(self) -> RLTrain | None:

        """
        Train an agent with the given configuration.
        """

        if os.path.exists(self.agent_path):
            warnings.warn(f"Training aborted. Folder '{self.agent_path}' already exists.")
            return

        train = RLTrain(
            config=self.config,
            update_observation_record=False,
            projector=self.get_projector(),
            path_constants=ReinforcementLearning.constants_path,
            path_train_data=ReinforcementLearning.train_data_path,
            path_test_data=ReinforcementLearning.test_data_path,
            path_folder=self.agent_path,
            verbose=self.verbose
        )
        train.solve()
        return train

    def collect_obs(self) -> RLEnvironment | None:

        """
        Collect observations for the observation type (e.g., for "O211", the observation type is "O21")
        """

        if os.path.exists(self.obs_records_path):
            warnings.warn(f"Observation collection aborted. Folder '{self.obs_records_path}' already exists.")
            return

        # The agent must be executed using no projector
        # This is done taking only the first 2 values of the observation config name (e.g., "O211" -> "O2")
        reduced_config_names = self.config_names.copy()
        reduced_config_names["O"] = reduced_config_names["O"][0:2]
        reduced_config = self.get_config(reduced_config_names)

        # Get the collection method, indicated by the second digit in the observation name (e.g., "O211" -> "1")
        collection_method = {
            "1": "training",
            "2": "random"
        }[self.config_names["O"][1]]

        if collection_method == 'training':
            if self.verbose >= 1:
                print("Collecting observations while training agent...")
            reduced_agent_path = os.path.join(
                ReinforcementLearning.models_folder, f"rl-{''.join(reduced_config_names)}"
            )
            train = RLTrain(
                config=reduced_config,
                projector=Projector.create_projector(reduced_config),
                path_constants=ReinforcementLearning.constants_path,
                path_train_data=ReinforcementLearning.train_data_path,
                path_test_data=ReinforcementLearning.test_data_path,
                path_folder=reduced_agent_path,
                update_observation_record=True,
                verbose=self.verbose
            )
            train.solve()
            env = train.train_env

        elif collection_method == 'random':
            if self.verbose >= 1:
                print("Collecting observations with random agent...")
            env = RLEnvironment(
                config=reduced_config,
                projector=Projector.create_projector(reduced_config),
                path_constants=ReinforcementLearning.constants_path,
                path_historical_data=ReinforcementLearning.train_data_path,
                update_observation_record=True
            )
            num_timesteps = 0
            while num_timesteps < reduced_config.num_timesteps:
                env.reset()
                done = False
                while not done:
                    action = env.action_space.sample()
                    _, _, done, _, _ = env.step(action)
                    num_timesteps += 1

        else:
            raise ValueError(f"Invalid value for `collection_method`: {collection_method}.")

        if self.verbose >= 1:
            print("Observations collected, (num_observations, num_features):", env.record_normalized_obs.shape)
        os.makedirs(self.obs_records_path)
        np.save(os.path.join(self.obs_records_path, 'observations.npy'), env.record_normalized_obs)
        reduced_config.to_json(os.path.join(self.obs_records_path, 'config.json'))
        if self.verbose >= 1:
            print(f"Created folder '{self.obs_records_path}'.")

        return env

    def get_projector(self) -> Projector:

        """
        Get the projector corresponding to the given configuration,
        assuming the required observations folder exists.
        """

        if os.path.exists(self.obs_records_path):
            observations = np.load(os.path.join(self.obs_records_path, 'observations.npy'))
            obs_config = RLConfiguration.from_json(os.path.join(self.obs_records_path, 'config.json'))
            if self.verbose >= 1:
                print(f"Using observations from '{self.obs_records_path}' for projector.")
        else:
            if self.config.projector_type != "identity":
                raise FileNotFoundError(
                    f"Cannot build projector because the projector type is not 'identity' "
                    f"and the observations folder '{self.obs_records_path}' does not exist."
                )
            observations = None
            obs_config = None
        projector = Projector.create_projector(self.config, observations, obs_config)
        return projector

    def plot_histogram(self, obs: np.ndarray, projected: bool, title: str):

        """

        :param obs: Array of shape num_observations x num_features with the flattened observations
        :param projected: Indicates if the observations are projected observations or raw/normalized observations
        :param title: Title of the histogram
        """

        # Raw or normalized flattened observation
        if not projected:
            indices = self.config.get_obs_indices(flattened=True)
            for dam_id in self.constants.get_ids_of_dams():
                max_sight = max(self.config.num_steps_sight[feature, dam_id] for feature in self.config.features)
                num_features = len(self.config.features)
                fig, axs = plt.subplots(max_sight, num_features)
                fig.suptitle(f"Histograms of {title} for {dam_id}")
                for feature_index, feature in enumerate(self.config.features):
                    for lookback in range(self.config.num_steps_sight[feature, dam_id]):
                        ax = axs[lookback, feature_index]
                        if self.constants.get_order_of_dam(dam_id) == 1 or feature not in self.config.unique_features:
                            index = indices[dam_id, feature, lookback]
                            ax.hist(obs[:, index], bins='auto')
                        ax.set_yticklabels([])  # Hide y-axis tick labels
                        ax.yaxis.set_ticks_position('none')  # Hide y-axis tick marks
                        if lookback == 0:
                            ax.set_title(feature)
                plt.tight_layout()
                plt.show()

        # Projected observation
        else:
            n_components = obs.shape[1]
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

    def plot_histograms_projector(self):

        projector = self.get_projector()
        obs_type = self.config_names['O'][0:3]

        def indicate_variance(proj_type: str):
            if proj_type not in {'identity', 'QuantilePseudoDiscretizer'}:
                return f"({self.config.projector_explained_variance * 100:.0f}%)"
            else:
                return ""

        self.plot_histogram(projector.observations, projected=False, title=f"Original observations {obs_type}")
        if isinstance(self.config.projector_type, list):
            proj_types = []
            for proj, proj_type in zip(projector.projectors, self.config.projector_type):  # noqa
                proj_types.append(proj_type)
                projected = proj_type not in {'identity', 'QuantilePseudoDiscretizer'}
                self.plot_histogram(
                    proj.transformed_observations,
                    projected=projected,
                    title=f"Observations {obs_type} after applying {', '.join(proj_types)} {indicate_variance(proj_type)}"
                )
        else:
            projected = self.config.projector_type not in {'identity', 'QuantilePseudoDiscretizer'}
            self.plot_histogram(
                projector.transformed_observations,
                projected=projected,
                title=f"Observations {obs_type} after applying {self.config.projector_type} "
                      f"{indicate_variance(self.config.projector_type)}"
            )

    @staticmethod
    def extract_substrings(input_string: str) -> dict[str, str]:

        """
        Break down a string of alphanumeric characters
        into substrings of numbers lead by different alphabetic characters.
        Example: "G0O21A1R1T0" is broken down into {"A": "A1", "G": "G0", "O": "O21", "R": "R1", "T": "T0"}.
        """

        pattern = re.compile(r'([A-Za-z]+)(\d+)')
        matches = pattern.findall(input_string)
        result = ["".join(match) for match in matches]
        result = sorted(result)
        result = {substring[0]: substring for substring in result}
        return result

    @staticmethod
    def get_config(config_names: dict[str, str]) -> RLConfiguration:

        """
        Get the RLConfiguration object from the configuration substrings,
        for example {"A": "A1", "G": "G0", "O": "O21", "R": "R1", "T": "T0"}.
        """

        configs = []
        for config_letter, config_name in config_names.items():
            config_folder, config_class = ReinforcementLearning.configs_info[config_letter]
            if config_letter == "O" and len(config_name) > 2:
                # Replace the observation collection method by a generic `X` (e.g., "O211" -> "O2X1")
                config_name = config_name[:2] + "X" + config_name[3:]
            config_path = os.path.join(config_folder, config_name + ".json")
            configs.append(config_class.from_json(config_path))
        config = RLConfiguration.from_components(configs)
        return config




