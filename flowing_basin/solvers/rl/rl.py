from flowing_basin.solvers.rl import (
    GeneralConfiguration, ObservationConfiguration, ActionConfiguration, RewardConfiguration, TrainingConfiguration,
    RLConfiguration, RLTrain, RLEnvironment
)
from flowing_basin.solvers.rl.feature_extractors import Projector
import numpy as np
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

        # The first two digits in the observation name (e.g., "O211" -> "O21")
        # indicate the type of observations that should be used for the projector
        self.obs_records_path = os.path.join(
            ReinforcementLearning.observation_records_folder, self.config_names["O"][0:3]
        )

    def train(self) -> RLTrain | None:

        """
        Train agent
        """

        if os.path.exists(self.agent_path):
            warnings.warn(f"Training aborted. Folder '{self.agent_path}' already exists.")
            return

        train = RLTrain(
            config=self.config,
            update_observation_record=False,
            path_constants=ReinforcementLearning.constants_path,
            path_train_data=ReinforcementLearning.train_data_path,
            path_test_data=ReinforcementLearning.test_data_path,
            path_observations_folder=self.obs_records_path if os.path.exists(self.obs_records_path) else None,
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
                update_observation_record=True,
                path_constants=ReinforcementLearning.constants_path,
                path_train_data=ReinforcementLearning.train_data_path,
                path_test_data=ReinforcementLearning.test_data_path,
                path_observations_folder=None,
                path_folder=reduced_agent_path,
                verbose=self.verbose
            )
            train.solve()
            env = train.train_env

        elif collection_method == 'random':
            if self.verbose >= 1:
                print("Collecting observations with random agent...")
            projector = Projector.create_projector(reduced_config)
            env = RLEnvironment(
                config=reduced_config,
                projector=projector,
                update_observation_record=True,
                path_constants=ReinforcementLearning.constants_path,
                path_historical_data=ReinforcementLearning.train_data_path,
                paths_power_models=None,
                instance=None,
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




