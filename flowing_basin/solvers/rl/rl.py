from flowing_basin.solvers.rl import GeneralConfiguration, ObservationConfiguration, ActionConfiguration, RewardConfiguration, TrainingConfiguration, RLConfiguration, RLTrain
import numpy as np
import os
import re


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

    def __init__(self, config_name: str, update_observation_record: bool = False):

        # Configuration and observation records
        configs = []
        observation_records = None
        config_substrings = self.extract_substrings(config_name)
        for config_substring in config_substrings:
            config_folder, config_class = ReinforcementLearning.configs_info[config_substring[0]]
            config_path = os.path.join(config_folder, config_substring + ".json")
            configs.append(config_class.from_json(config_path))
            if config_substring.startswith("O"):
                basic_obs = config_substring[0:3]
                observation_records = os.path.join(ReinforcementLearning.observation_records_folder, basic_obs)
        config = RLConfiguration.from_components(configs)

        # Folder in which to save the agent
        agent_folder = os.path.join(ReinforcementLearning.models_folder, f"rl-{config_name}")

        # Train agent
        train = RLTrain(
            config=config,
            update_observation_record=update_observation_record,
            path_constants=ReinforcementLearning.constants_path,
            path_train_data=ReinforcementLearning.train_data_path,
            path_test_data=ReinforcementLearning.test_data_path,
            path_observations_folder=observation_records if os.path.exists(observation_records) else None,
            path_folder=agent_folder
        )
        train.solve()

        # Save observations experienced by agent during training
        if update_observation_record and not os.path.exists(observation_records):
            os.makedirs(observation_records)
            np.save(os.path.join(observation_records, 'observations.npy'), train.train_env.record_normalized_obs)
            config.to_json(os.path.join(observation_records, 'config.json'))
            print(f"Created folder '{observation_records}'.")

    @staticmethod
    def extract_substrings(input_string):

        """
        Break down a string of alphanumeric characters
        into substrings of numbers lead by different alphabetic characters.
        Example: "G0O21A1R1T0" is broken down into "G0", "O21", "A1", "R1", "T0".
        """

        pattern = re.compile(r'([A-Za-z]+)(\d+)')
        matches = pattern.findall(input_string)
        result = ["".join(match) for match in matches]
        return result




