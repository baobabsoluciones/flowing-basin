from flowing_basin.core import Instance, Solution, TrainingData
from flowing_basin.solvers.rl import (
    GeneralConfiguration, ObservationConfiguration, ActionConfiguration, RewardConfiguration, TrainingConfiguration,
    RLConfiguration, RLEnvironment, RLTrain, RLRun
)
from flowing_basin.solvers.rl.feature_extractors import Projector
from flowing_basin.solvers.common import (
    get_all_instances, get_all_baselines, barchart_instances, CONSTANTS_PATH, print_save_csv
)
from cornflow_client.core.tools import load_json
from stable_baselines3 import SAC, A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_checker import check_env
from flowing_basin.rl_zoo.rl_zoo3 import linear_schedule
import torch
import numpy as np
import math
from matplotlib import pyplot as plt
import os
import re
import warnings
from time import perf_counter
from datetime import datetime
import csv
import json


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
    train_data_path = os.path.join(os.path.dirname(__file__), "../../data/history/historical_data_clean_train.pickle")
    test_data_path = os.path.join(os.path.dirname(__file__), "../../data/history/historical_data_clean_test.pickle")

    models_folder = os.path.join(os.path.dirname(__file__), "../../rl_data/models")
    tensorboard_folder = os.path.join(os.path.dirname(__file__), "../../rl_data/tensorboard_logs")
    best_hyperparams_folder = os.path.join(os.path.dirname(__file__), "../../rl_zoo/best_hyperparams")

    observation_records = ["record_raw_obs", "record_normalized_obs", "record_projected_obs"]  # As the attributes in RLEnvironment
    static_projectors = ["identity", "QuantilePseudoDiscretizer"]

    obs_basic_type_length = 2  # Length of the observation code that indicates the basic type (e.g., "O121" -> "O1")
    obs_type_length = 3  # Length of the observation code that indicates the type (e.g., "O121" -> "O12")
    obs_collection_pos = 2  # Position of the digit that MAY indicate a collection method (e.g., "O12" -> "2")
    obs_collection_codes = {'1', '2'}  # Digits that DO indicate a collection method (e.g., not "0" in "O10")

    def __init__(self, config_name: str, verbose: int = 2):

        self.verbose = verbose
        config_name = config_name
        self.config_names = self.extract_substrings(config_name)
        self.config_full_name = ''.join(self.config_names.values())  # Identical to `config_name`, but in alphabetical order

        self.config = self.get_config(self.config_names)
        self.constants_path = CONSTANTS_PATH.format(num_dams=self.config.num_dams)
        self.agent_name = f"rl-{self.config_full_name}"
        self.constants = Instance.from_dict(load_json(self.constants_path))

        # The first two digits in the observation name (e.g., "O211" -> "O21")
        # indicate the type of observations that should be used for the projector
        folder_name = self.config_names["O"][:ReinforcementLearning.obs_type_length]
        if len(folder_name) == 2:
            # Assume the record method is the random agent, "O1" -> "O12"
            folder_name += "2"
        if self.config.num_timesteps != 99_000:
            folder_name += f"_steps{self.config.num_timesteps}"
        if self.config.num_actions_block != 1:
            folder_name += f"_block{self.config.num_actions_block}"
        self.obs_records_path = os.path.join(ReinforcementLearning.observation_records_folder, folder_name)

    def train(
            self, save_agent: bool = True, save_replay_buffer: bool = False, save_obs: bool = False,
            save_tensorboard: bool = True, num_timesteps: int = None
    ) -> RLTrain | None:

        """
        Train an agent with the given configuration.

        :param save_agent: Whether to save the agent or not
            (not saving the agent may be interesting for testing purposes).
        :param save_replay_buffer: Whether to save the replay buffer or not
            (the replay buffer may occupy several GB for big observation spaces).
        :param save_obs: Whether to save the observations experienced during training or not
            (these observations may occupy several GB for big observation spaces).
        :param save_tensorboard: Whether to save the tensorboard logs or not.
        :param num_timesteps: If given, override the number of timesteps in the configuration (only for testing purposes)
        """

        # Define model
        if self.verbose >= 1:
            print(f"Defining agent {self.agent_name}...")
        train = RLTrain(
            config=self.config,
            projector=self.create_projector(),
            update_observation_record=save_obs,
            save_replay_buffer=save_replay_buffer,
            path_constants=self.constants_path,
            path_train_data=ReinforcementLearning.train_data_path,
            path_test_data=ReinforcementLearning.test_data_path,
            path_folder=self.get_agent_folder_path() if save_agent else None,
            path_tensorboard=ReinforcementLearning.tensorboard_folder if save_agent and save_tensorboard else None,
            experiment_id=self.agent_name,
            verbose=self.verbose,
            num_timesteps=num_timesteps
        )

        # Train model
        if self.verbose >= 1:
            print(f"Training agent {self.agent_name} for {self.config.num_timesteps} timesteps...")
        start = perf_counter()
        train.solve()
        if self.verbose >= 1:
            print(
                f"Trained agent {self.agent_name} for {self.config.num_timesteps} timesteps "
                f"in {perf_counter() - start}s."
            )

        # Saving the observations allows studying them later
        if save_obs and save_agent:
            for obs_record in ReinforcementLearning.observation_records:
                obs_record_path = os.path.join(self.get_agent_folder_path(), f'{obs_record}.npy')
                obs = np.array(getattr(train.train_env, obs_record))
                np.save(obs_record_path, obs)
                if self.verbose >= 1:
                    print(
                        f"Saved {obs_record} for {self.agent_name} with {obs.shape} observations "
                        f"in file '{obs_record_path}'."
                    )

        return train

    def collect_obs(self) -> RLEnvironment | None:

        """
        Collect observations for the observation type (e.g., for "O211", the observation type is "O21")
        """

        if os.path.exists(self.obs_records_path):
            warnings.warn(f"Observation collection aborted. Folder '{self.obs_records_path}' already exists.")
            return

        if len(self.config_names["O"]) < 3:
            warnings.warn(
                f"Observation collection aborted. "
                f"The observation config name does not indicate the collection method (no second digit)."
            )
            return

        # Get the collection method, indicated by the second digit in the observation name (e.g., "O211" -> "1")
        collection_code = self.config_names["O"][ReinforcementLearning.obs_collection_pos]
        if collection_code not in ReinforcementLearning.obs_collection_codes:
            warnings.warn(
                f"Observation collection aborted. "
                f"The observation config name does not indicate a collection method (second digit is not '1' or '2')."
            )
            return
        collection_method = {
            "1": "training",
            "2": "random"
        }[collection_code]

        # The agent must be executed using no projector
        # This is done taking only the first 2 values of the observation config name (e.g., "O211" -> "O2")
        reduced_config_names = self.config_names.copy()
        reduced_config_names["O"] = reduced_config_names["O"][:ReinforcementLearning.obs_basic_type_length]
        reduced_config = self.get_config(reduced_config_names)
        reduced_agent_name = f"rl-{''.join(reduced_config_names.values())}"

        if collection_method == 'training':
            if self.verbose >= 1:
                print(f"Collecting observations for {reduced_config.num_timesteps} timesteps while training agent...")
            reduced_agent_path = os.path.join(ReinforcementLearning.models_folder, reduced_agent_name)
            train = RLTrain(
                config=reduced_config,
                projector=Projector.create_projector(reduced_config),
                path_constants=CONSTANTS_PATH.format(num_dams=reduced_config.num_dams),
                path_train_data=ReinforcementLearning.train_data_path,
                path_test_data=ReinforcementLearning.test_data_path,
                path_folder=reduced_agent_path,
                path_tensorboard=ReinforcementLearning.tensorboard_folder,
                experiment_id=reduced_agent_name,
                update_observation_record=True,
                verbose=self.verbose
            )
            start = perf_counter()
            train.solve()
            if self.verbose >= 1:
                print(f"Collected for {reduced_config.num_timesteps} timesteps in {perf_counter() - start}s.")
            env = train.train_env

        elif collection_method == 'random':
            if self.verbose >= 1:
                print(f"Collecting observations for {reduced_config.num_timesteps} timesteps with random agent...")
            env = RLEnvironment(
                config=reduced_config,
                projector=Projector.create_projector(reduced_config),
                path_constants=CONSTANTS_PATH.format(num_dams=reduced_config.num_dams),
                path_historical_data=ReinforcementLearning.train_data_path,
                update_observation_record=True
            )
            start = perf_counter()
            num_timesteps = 0
            while num_timesteps < reduced_config.num_timesteps:
                env.reset()
                done = False
                while not done:
                    action = env.action_space.sample()
                    _, _, done, _, _ = env.step(action)
                    num_timesteps += 1
                    if self.verbose >= 2 and num_timesteps % 495 == 0:
                        print(
                            f"Collected {num_timesteps} / {reduced_config.num_timesteps} "
                            f"({num_timesteps / reduced_config.num_timesteps * 100:.1f}%) observations "
                            f"in {perf_counter() - start}s so far"
                        )
            if self.verbose >= 1:
                print(f"Collected for {reduced_config.num_timesteps} timesteps in {perf_counter() - start}s.")

        else:
            raise ValueError(f"Invalid value for `collection_method`: {collection_method}.")

        normalized_obs = np.array(env.record_normalized_obs)
        if self.verbose >= 1:
            print("Observations collected, (num_observations, num_features):", normalized_obs.shape)
        os.makedirs(self.obs_records_path)
        np.save(os.path.join(self.obs_records_path, 'observations.npy'), normalized_obs)
        reduced_config.to_json(os.path.join(self.obs_records_path, 'config.json'))
        if self.verbose >= 1:
            print(f"Created folder '{self.obs_records_path}'.")

        return env

    def check_train_env(
            self, initial_date: str = None, max_timestep: int = float('inf'),
            obs_types: list[str] = None,  update_obs: bool = False, seed: int = None
    ):

        """
        Check the training environment works as expected

        :param max_timestep: Number of timesteps for which to run the environment (default, until the episode finishes)
        :param initial_date: Initial date of the instance of the environment, in format '%Y-%m-%d %H:%M' (default, random)
        :param obs_types: Observation types to show (default, all of them: "raw", "normalized", and "projected")
        :param update_obs: Whether to update the observation record of the environment or not
        :param seed: Seed for the random actions
        """

        if obs_types is None:
            obs_types = ["raw", "normalized", "projected"]

        env = self.create_train_env(update_obs)

        # Check the environment does not have any errors according to StableBaselines3
        check_env(env)

        if initial_date is not None:
            initial_date = datetime.strptime(initial_date, '%Y-%m-%d %H:%M')

        obs, info = env.reset(options=dict(initial_row=initial_date))

        # Instance
        print("\nINSTANCE:")
        print(env.instance.data)
        instance_checks = env.instance.check()
        if instance_checks:
            raise RuntimeError(f"There are problems with the created instance: {instance_checks}.")

        # Initial observation
        print(f"\n---- Timestep -1 ----")
        for obs_type in obs_types:
            print(f"\n{obs_type} OBSERVATION:".upper())
            obs_to_print = info[f'{obs_type}_obs'] if obs_type != 'projected' else obs
            env.print_obs(obs_to_print)

        # Seed for random actions
        if seed is not None:
            env.action_space.seed(seed)

        done = False
        timestep = 0
        while not done and timestep < max_timestep:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            print(f"\n---- Timestep {timestep} ----")
            print("\nACTION:", action)
            print("FLOW:", info['flow'])
            print("\nREWARD:", reward)
            if self.config.action_type == "adjustments":
                print("ABSOLUTE REWARDS IN PAST ITERATIONS:", env.total_rewards)
            if self.config.num_actions_block == 1:
                print("REWARD DETAILS:", env.get_reward_details())
            for obs_type in obs_types:
                print(f"\n{obs_type} OBSERVATION:".upper())
                obs_to_print = info[f'{obs_type}_obs'] if obs_type != 'projected' else obs
                env.print_obs(obs_to_print)

            timestep += 1

    def create_train_env(self, update_obs: bool = True) -> RLEnvironment:

        """
        Create the training environment for the agent
        """

        env = RLEnvironment(
            config=self.config,
            projector=self.create_projector(),
            path_constants=self.constants_path,
            path_historical_data=ReinforcementLearning.train_data_path,
            update_observation_record=update_obs
        )
        return env

    def create_projector(self) -> Projector:

        """
        Get the projector corresponding to the given configuration,
        assuming the required observations folder exists.
        """

        observations = None
        if self.config.projector_type != "identity":
            observations = self.load_observation_record()
            if observations is not None:
                if self.verbose >= 1:
                    print(f"Using observations from '{self.obs_records_path}' for projector.")
            else:
                warnings.warn(
                    f"For agent {self.agent_name}, projector type is not 'identity' "
                    f"but the observations folder '{self.obs_records_path}' does not exist. "
                    f"Running the observation collection method to get the required observation record..."
                )
                self.collect_obs()
                observations = self.load_observation_record()
                assert observations is not None, "Observation record cannot be None after observation collection."
        projector = Projector.create_projector(self.config, observations)
        return projector

    def load_observation_record(self) -> np.ndarray | None:

        """
        Load the observation record for the agent.
        """

        observations = None

        if os.path.exists(self.obs_records_path):

            observations = np.load(os.path.join(self.obs_records_path, 'observations.npy'))
            obs_config = RLConfiguration.from_json(os.path.join(self.obs_records_path, 'config.json'))

            # Check observations of the observation record are compatible with the current configuration
            compatibility_errors = self.config.check_observation_compatibility(obs_config)
            if compatibility_errors:
                raise Exception(f"The configurations are not observation-compatible: {compatibility_errors}")

        return observations

    def get_agent_folder_path(self, training_iter: int = None):

        """
        Get the path to the folder of the RL agent.
        :param training_iter: Training iteration of the agent; default is to take the last one.
        :return:
        """

        def get_full_agent_path(i: int):
            agent_path = os.path.join(ReinforcementLearning.models_folder, self.agent_name)
            suffix = f"_{i}" if i > 0 else ""
            return agent_path + suffix

        if training_iter is not None:
            # Get the folder of the specified training iteration
            full_agent_path = get_full_agent_path(training_iter)

        else:
            # Get the folder of the last training iteration (i.e., of the last folder saved in disk)
            i = 0
            full_agent_path = get_full_agent_path(i)
            i += 1
            while os.path.exists(get_full_agent_path(i)):
                full_agent_path = get_full_agent_path(i)
                i += 1

        return full_agent_path

    def get_model_path(self, model_type: str = "best_model"):

        """
        Get the path to a trained model.

        :param model_type: Either "model" (the model in the last timestep of training)
            or "best_model" (the model with the highest evaluation from StableBaselines3 EvalCallback)
        """

        # In T0, the best model is called "model_best" instead of "best_model"
        if self.config_names['T'] == 'T0' and model_type == "best_model":
            model_type = "model_best"

        model_path = os.path.join(self.get_agent_folder_path(), f"{model_type}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"There is no trained model {self.agent_name}. File {model_path} doesn't exist."
            )

        return model_path

    def get_normalization_path(self, model_type: str = "best_model") -> str | None:

        """
        Get the path to the environment's normalization statistics.
        Returns None when normalization is turned off.
        :param model_type: Either "model" (the statistics in the last moment of training)
            or "best_model" (the statistics when the model was at its best)
        :return:
        """

        if model_type == "best_model":
            norm_file = "best_vecnormalize.pkl"
        else:
            norm_file = "vecnormalize.pkl"

        normalization_path = None
        if self.config.normalization:
            normalization_path = os.path.join(self.get_agent_folder_path(), norm_file)
            if not os.path.exists(normalization_path):
                raise FileNotFoundError(
                    f"There are no normalization statistics for {self.agent_name}."
                    f" File {normalization_path} doesn't exist."
                )

        return normalization_path

    def load_model(self, model_type: str = "best_model") -> BaseAlgorithm:

        """
        Load a trained model.

        :param model_type: See method `get_model_path`
        """

        # To avoid a KeyError, you must indicate the env and its observation_space and action_space
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/1682#issuecomment-1813338493
        # You must also pass a lr_schedule to avoid a warning
        # See pull request https://github.com/DLR-RM/stable-baselines3/pull/336
        model_path = self.get_model_path(model_type)
        env = self.create_train_env()
        env = env.get_vec_env(is_eval_env=False, path_normalization=self.get_normalization_path(model_type))
        algorithm = dict(SAC=SAC, A2C=A2C, PPO=PPO)[self.config.algorithm]

        lr_schedule = lambda _: self.config.learning_rate
        if self.config.lr_schedule_name is not None:
            if self.config.lr_schedule_name == "linear":
                lr_schedule = linear_schedule(self.config.learning_rate)

        model = algorithm.load(
            model_path,
            env=env,
            custom_objects={
                'observation_space': env.observation_space,
                'action_space': env.action_space,
                'lr_schedule': lr_schedule,
            }
        )

        return model

    def integrated_gradients(self):

        """
        At the moment, this does nothing, just tests things
        """

        try:
            from captum.attr import IntegratedGradients  # noqa
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Module 'captum' must be installed to execute method 'integrated_gradients'.")

        model = self.load_model()  # type: BaseAlgorithm
        obs_record = self.load_observation_record()  # type: np.ndarray # shape (num_obs, num_features)

        obs = obs_record[0]
        obs = torch.from_numpy(obs).reshape(1, -1)

        action = model.policy(obs, deterministic=True)
        print(action)

        # Equivalent:
        action = model.policy.actor(obs, deterministic=True)
        print(action)

        # Evaluation of both critics
        evaluation_critics = model.policy.critic(obs=obs, actions=action)
        print(evaluation_critics)

        # Evaluation of only the first critic
        obs_action_pair = torch.cat([obs, action], dim=1)
        evaluation_critic1 = model.policy.critic.q_networks[0](obs_action_pair)
        print(evaluation_critic1)

    def plot_histogram(self, obs: np.ndarray, projected: bool, title: str, show_lookback: bool = True):

        """

        :param obs: Array of shape num_observations x num_features with the flattened observations
        :param projected: Indicates if the observations are projected observations or raw/normalized observations
        :param title: Title of the histogram
        :param show_lookback: Whether to show all the lagged versions of the variable or not
        """

        # Method 'auto' raises an error for giving almost 0 width bins
        # when input array is small and the array has extreme outliers.
        # (This was observed with the feature past_clipped when having 1001 observations.)
        # This is avoided by forcing the usage of the Sturges method for bin width estimation,
        # which is better for small datasets.
        bins_method = 'sturges' if obs.shape[0] < 50_000 else 'auto'

        # Raw or normalized flattened observation
        if not projected:

            indices = self.config.get_obs_indices(flattened=True)
            num_features = len(self.config.features)

            # When show_lookback=False, all dams are plotted on the same histogram
            # and there is an additional column at the beginning to indicate the dam
            fig, axs = None, None
            if not show_lookback:
                fig, axs = plt.subplots(
                    self.constants.get_num_dams(), num_features + 1, figsize=(12, int(0.5 * num_features)),
                    gridspec_kw={'width_ratios': [0.4] + [1] * num_features}
                )
                fig.suptitle(f"Histogram of original state variables")

            for dam_id in self.constants.get_ids_of_dams():

                max_sight = max(self.config.num_steps_sight[feature, dam_id] for feature in self.config.features)
                dam_index = self.constants.get_order_of_dam(dam_id) - 1

                if show_lookback:
                    fig, axs = plt.subplots(max_sight, num_features)
                    fig.suptitle(f"Histograms of {title} for {dam_id}")
                else:
                    # Put the dam ID to the left
                    fig.text(0.05, 1.0 - (0.35 + dam_index / self.constants.get_num_dams()), dam_id,
                             va='bottom', ha='center', rotation=90)

                for feature_index, feature in enumerate(self.config.features):
                    for lookback in range(self.config.num_steps_sight[feature, dam_id]):

                        # When show_lookback=False, the vertical dimension is used for dams instead of lookback
                        if show_lookback:
                            ax = axs[lookback, feature_index]
                        else:
                            ax = axs[dam_index, feature_index + 1]

                        if self.constants.get_order_of_dam(dam_id) == 1 or feature not in self.config.unique_features:
                            index = indices[dam_id, feature, lookback]
                            if feature == "past_clipped" or feature == "past_vols":
                                # The automatic bin method does not work correctly with these features
                                ax.hist(obs[:, index], bins=15)
                            else:
                                ax.hist(obs[:, index], bins=bins_method)

                        ax.set_xlim(0, 1)
                        ax.set_xticks([0, 1])
                        ax.set_yticklabels([])  # Hide y-axis tick labels
                        ax.yaxis.set_ticks_position('none')  # Hide y-axis tick marks
                        if lookback == 0:
                            if show_lookback:
                                ax.set_title(feature)
                            else:
                                # Change feature names to make it more descriptive
                                name_map = {
                                    "vols": "volumes",
                                    "flows": "outflows",
                                    "turbined": "turbine flows",
                                    "groups": "turbines",
                                    "clipped": "outflow cuts"
                                }
                                feature_name = feature.split("_")[-1]
                                ax.set_title(name_map[feature_name] if feature_name in name_map else feature_name)
                                break

                if show_lookback:
                    # Plot dam by dam
                    plt.tight_layout()
                    plt.show()
                else:
                    # Hide the first column, which should be left free for the dam label
                    axs[dam_index, 0].axis('off')

            # Plot all dams at the same time
            if not show_lookback:
                plt.tight_layout()
                plt.savefig(f"reports/histograms_{self.config_names['O']}.eps", format="eps")
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

    def plot_histograms_projector_obs(self, show_lookback: bool = True, show_projected: bool = True):

        """
        Plot the histograms of the observations used to train the projector,
        as well as these observations after being transformed by this projector.
        """

        projector = self.create_projector()
        obs_type = self.config_names['O'][:ReinforcementLearning.obs_type_length]

        self.plot_histogram(
            projector.observations, projected=False,
            title=f"Original observations {obs_type}", show_lookback=show_lookback
        )
        if show_projected:
            self.plot_histograms_projected(projector.observations, projector, title=str(obs_type))

    def plot_histograms_agent_obs(self, apply_projections: bool = True):

        """
        Plot the histogram of the record of raw, normalized and projected observations experienced by the agent

        :param apply_projections: If False, get the projected observations from "record_projected_obs" in the folder;
        if True, manually apply the projector to get the projected observations
        """

        obs_normalized = None
        for obs_record in ReinforcementLearning.observation_records:
            obs_folder = os.path.join(self.get_agent_folder_path(), f'{obs_record}.npy')
            try:
                obs = np.load(obs_folder)
            except FileNotFoundError:
                warnings.warn(f"Histograms of {obs_record} not plotted. File '{obs_folder}' does not exist.")
                continue

            if obs_record != "record_projected_obs":
                self.plot_histogram(obs, projected=False, title=obs_record)

            # Get the projected observations from the folder
            if obs_record == "record_projected_obs" and not apply_projections:
                proj_type = self.config.projector_type
                proj_type = proj_type if not isinstance(proj_type, list) else ', '.join(proj_type)
                projected = proj_type not in ReinforcementLearning.static_projectors
                self.plot_histogram(obs, projected=projected, title=f"{obs_record} ({proj_type})")

            if obs_record == "record_normalized_obs":
                obs_normalized = obs

        # Manually apply the projections
        if apply_projections:
            if obs_normalized is None:
                warnings.warn("Cannot apply projections because the normalized observations were not found.")
                return
            projector = self.create_projector()
            self.plot_histograms_projected(obs_normalized, projector)

    def plot_histograms_projected(self, obs: np.ndarray, projector: Projector, title: str = None):

        """
        Plot the histograms of the projections of the given observations
        """

        if title is None:
            title = self.agent_name

        def indicate_variance(proj_type: str):
            if proj_type not in ReinforcementLearning.static_projectors:
                return f"({self.config.projector_explained_variance * 100:.0f}%)"
            else:
                return ""

        if isinstance(self.config.projector_type, list):
            proj_types = []
            for proj, proj_type in zip(projector.projectors, self.config.projector_type):  # noqa
                proj_types.append(proj_type)
                is_projected = proj_type not in ReinforcementLearning.static_projectors
                obs = proj.transform(obs)
                self.plot_histogram(
                    obs,
                    projected=is_projected,
                    title=f"Observations from {title} after applying "
                          f"{', '.join(proj_types)} {indicate_variance(proj_type)}"
                )
        else:
            is_projected = self.config.projector_type not in ReinforcementLearning.static_projectors
            obs = projector.project(obs)
            self.plot_histogram(
                obs,
                projected=is_projected,
                title=f"Observations from {title} after applying "
                      f"{self.config.projector_type} {indicate_variance(self.config.projector_type)}"
            )

    def run_agent(
            self, instance: Instance | str | list[Instance | str], model_type: str = "best_model"
    ) -> RLRun | list[RLRun]:

        """
        Solve the given instance with the current agent

        :param instance: Instance to solve, its name, or a list of instances/names
        :param model_type: See method `load_model`
        :return: Run or list of runs (depending on whether a single or multiple instances where given)
        """

        if not isinstance(instance, list):
            instance = [instance]

        runs = []
        model = self.load_model(model_type)
        projector = self.create_projector()
        for inst in instance:
            if isinstance(inst, str):
                inst = Instance.from_name(inst, num_dams=self.config.num_dams)
            run = RLRun(
                config=self.config,
                instance=inst,
                projector=projector,
                path_normalization=self.get_normalization_path(model_type),
                solver_name=self.agent_name
            )
            run.solve(model.policy, skip_rewards=True)
            runs.append(run)

        if len(runs) == 1:
            runs = runs.pop()

        return runs

    def run_named_policy(self, policy_name: str, instance: Instance, update_to_decisions: bool = False) -> Solution:

        """
        Solve the given instance with a special policy ("random" or "greedy")
        """

        if policy_name.split("_")[0] not in RLRun.named_policies:
            raise ValueError(
                f"Invalid value for `policy_name`: {policy_name}. Allowed values are {RLRun.named_policies}."
            )

        run = RLRun(
            config=self.config,
            instance=instance,
            projector=self.create_projector(),
            solver_name=f"rl-{policy_name}" + ("_biased" if update_to_decisions else ""),
            update_to_decisions=update_to_decisions
        )
        run.solve(policy_name)
        return run.solution

    def run_imitator(self, solution: Solution, instance: Instance) -> RLRun:

        """
        Solve the given instance with the agent that imitates the given solution.
        """

        run = RLRun(
            config=self.config,
            instance=instance,
            projector=self.create_projector(),
            solver_name=f"{self.agent_name}_imitating_{solution.get_solver()}"
        )
        run.solve(solution)
        return run

    def plot_training_curve_agent(
            self, agents: list[str] = None, baselines: list[str] = None,
            values: list[str] = None, instances: str | list[str] = 'fixed'
    ):

        """
        Plot the training curve of the agent
        compared with the training curve of the given agents
        and the values of the given baseline solvers
        """

        if agents is None:
            agents = []

        ReinforcementLearning.plot_training_curves(
            agents=[self.agent_name, *agents], baselines=baselines, values=values, instances=instances
        )

    def barchart_instances_rewards(
            self, reward_configs: list[str], named_policy: str = None, model_type: str = "best_model"
    ):

        """
        Show a barchart with the reward across all fixed instances
        of the agent with each of the given reward configuration.

        :param reward_configs:
        :param named_policy: Named policy to run instead of the current agent
        :param model_type: See method `load_model`
        :return:
        """

        agent_name = self.agent_name if named_policy is None else f"rl-{named_policy}"
        values = dict()

        for reward_config in reward_configs:

            # Change the reward configuration
            new_config_names = self.config_names.copy()
            new_config_names["R"] = reward_config
            new_config = self.get_config(new_config_names)
            new_agent_name = f"rl-{''.join(new_config_names.values())}"
            avg_rewards = []
            values[reward_config] = dict()

            if named_policy is None:
                policy = self.load_model(model_type).policy
            else:
                policy = named_policy

            for instance in self.get_all_fixed_instances(new_config.num_dams):

                if self.verbose >= 2:
                    print(f"[{agent_name}] [{reward_config}] [{instance.get_instance_name()}] Running...")
                run = RLRun(
                    config=new_config,
                    instance=instance,
                    projector=self.create_projector(),
                    path_normalization=self.get_normalization_path(model_type),
                    solver_name=new_agent_name
                )
                run.solve(policy)
                avg_reward = sum(run.rewards_per_period) / len(run.rewards_per_period)
                avg_rewards.append(avg_reward)
                values[reward_config][instance.get_instance_name()] = avg_reward

            if self.verbose >= 1:
                print(f"[{agent_name}] [{reward_config}] Average rewards:", avg_rewards)
                print(f"[{agent_name}] [{reward_config}] Average rewards (mean):", sum(avg_rewards) / len(avg_rewards))

        self.barchart_instances(
            values, value_type="Average reward", agent_name=agent_name, general_config=self.config_names['G']
        )

    @staticmethod
    def plot_all_training_curves(
            agents_regex_filter: str | list[str] = '.*', permutation: str = 'AGORT', baselines: list[str] = None,
            values: list[str] = None, instances: str | list[str] = 'fixed'
    ):

        """
        Plot the training curve of all agents matching the given regex pattern
        """

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        ReinforcementLearning.plot_training_curves(
            agents=agents, baselines=baselines, values=values, instances=instances
        )

    @staticmethod
    def plot_training_curves(
            agents: list[str], baselines: list[str] = None,
            values: list[str] = None, instances: str | list[str] = 'fixed'
    ):

        """
        Plot the training curves of the given agents
        and the values of the given baseline solvers.

        :param agents: List of agent IDs in rl_data/models (e.g., ["rl-A1G0O22R1T0", "rl-A1G0O221R1T0"]).
        :param baselines: List of solvers in the baselines folder (e.g., ["MILP", "rl-random", "rl-greedy"]).
        :param values: Can be "income" or "acc_reward".
        :param instances: Can be "fixed", "random", or a list of specific fixed instances.
        """

        if baselines is None:
            baselines = ['MILP', 'rl-random', 'rl-greedy']
        if values is None:
            values = ['income']

        # Ensure all agents have the same General configuration
        general_config = ReinforcementLearning.common_general_config(
            agents, warning_msg="The plotted graphs will not be comparable and baselines will not be shown."
        )

        # For each agent, create a TrainingData object from "training_data.json"
        # and then add "evaluations.npz" from the EvalCallback to the TrainingData object
        training_objects = []
        for agent in agents:
            training_object = ReinforcementLearning.get_training_data(agent)
            training_object.add_random_instances(agent, ReinforcementLearning.get_path_evaluations(agent))
            training_objects.append(training_object)
        training = sum(training_objects)

        # Add baselines
        if general_config is not None:
            for baseline in ReinforcementLearning.get_all_baselines(general_config):
                if baseline.get_solver() in baselines:
                    training += baseline

        _, ax = plt.subplots()
        training.plot_training_curves(ax, values=values, instances=instances)  # noqa
        plt.show()

    @staticmethod
    def barchart_training_times(agents_regex_filter: str | list[str] = '.*', permutation: str = 'AGORT', hours: bool = False):

        """
        Show the training time of all agents matching the given regex in a barchart
        """

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        training_times = ReinforcementLearning.get_training_times(agents)
        unit = 'min'
        if hours:
            training_times = [training_time / 60 for training_time in training_times]
            unit = 'hours'

        plt.bar(agents, training_times)
        plt.xticks(rotation='vertical')  # Put the agent IDs vertically
        plt.xlabel('Agents')
        plt.ylabel(f'Training time ({unit})')
        plt.title('Training time of agents')
        plt.tight_layout()  # Avoid the agent IDs being cut down at the bottom of the figure
        plt.show()

    @staticmethod
    def print_training_times(
            agents_regex_filter: str | list[str] = '.*', permutation: str = 'AGORT',
            hours: bool = False, csv_filepath: str = None
    ):

        """
        Show the training time of all agents matching the given regex in a barchart
        """

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        training_times = ReinforcementLearning.get_training_times(agents)
        unit = 'min'
        if hours:
            training_times = [training_time / 60 for training_time in training_times]
            unit = 'hours'

        results = [
            ["agent", f"training_time_{unit}"]
        ]
        for agent, training_time in zip(agents, training_times):
            results.append([agent, training_time])

        # Save results in .csv file and print them
        print_save_csv(results, csv_filepath=csv_filepath)

    @staticmethod
    def barchart_instances_incomes(
            agents_regex_filter: str | list[str] = '.*', permutation: str = 'AGORT', baselines: list[str] = None,
            read_from_data: bool = False
    ):

        """
        Show a barchart with the income across all fixed instances
        of the given agents and baselines, as stored in the JSON files.

        :param agents_regex_filter:
        :param permutation:
        :param baselines:
        :param read_from_data:
        :return:
        """

        if baselines is None:
            baselines = ['MILP', 'rl-random', 'rl-greedy']

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        general_config = ReinforcementLearning.common_general_config(
            agents, warning_msg="Cannot show barchart (the results are not comparable and no baseline can be found)."
        )
        if general_config is None:
            return

        # Agent values
        values = dict()
        for agent in agents:
            if read_from_data:
                # Get the max avg income saved in the evaluation data (this value is biased in old agents)
                training_data_agent = ReinforcementLearning.get_training_data(agent)
                values[training_data_agent.handle_no_agent_id()] = training_data_agent.get_agent_best_instances_values()
            else:
                # Calculate a fresh avg income using the best model
                rl = ReinforcementLearning(agent)
                values[agent] = dict()
                runs = rl.run_agent(ReinforcementLearning.get_all_fixed_instances(rl.config.num_dams))
                for run in runs:
                    if rl.config.action_type == "adjustments":
                        # Get the best solution achieved, not the latest one
                        sol = max(run.solutions, key=lambda s: s.get_objective_function())
                    else:
                        sol = run.solution
                    income = sol.get_objective_function()
                    values[agent][run.instance.get_instance_name()] = income

        # Baseline values
        training_data_baselines = TrainingData.create_empty()
        all_baselines = ReinforcementLearning.get_all_baselines(general_config)
        for baseline in all_baselines:
            if baseline.get_solver() in baselines:
                training_data_baselines += baseline
        values.update(training_data_baselines.get_baseline_instances_values())
        print("Plotting the values:", values)

        # Plot the values gathered
        ReinforcementLearning.barchart_instances(
            values, value_type="Income (€)", agent_name="agents and baselines", general_config=general_config
        )

    @staticmethod
    def barchart_instances(values: dict[str, dict[str, float]], value_type: str, agent_name: str, general_config: str):

        """
        Plot a barchart with the value of each solver at every instance.
        The values may be incomes, rewards, or anything else.
        The 'solvers' may actually be something else, e.g. different reward configurations.

        :param values: dict[solver, dict[instance, value]]
        :param value_type: Indicate which value is being plotted (income, reward...)
        :param agent_name: Name of the agent(s) that will appear on the title
        :param general_config:
        """

        barchart_instances(values=values, value_type=value_type, title=agent_name, general_config=general_config)

    @staticmethod
    def print_max_avg_incomes(
            agents_regex_filter: str | list[str] = '.*', permutation: str = 'AGORT', baselines: list[str] = None,
            read_from_data: bool = False, take_average: bool = True, csv_filepath: str = None
    ) -> list[list[str]] | None:

        """
        Print a CSV table with the maximum average income of each agent
        and the corresponding average income of the baselines
        """

        if baselines is None:
            baselines = ['MILP', 'rl-random', 'rl-greedy']

        results = [
            ["method", "max_avg_income"]
        ]

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        general_config = ReinforcementLearning.common_general_config(
            agents, warning_msg="Cannot print CSV table (the results are not comparable and no baseline can be found)."
        )
        if general_config is None:
            return

        def append_last_row_to_file():
            if csv_filepath is not None:
                with open(csv_filepath, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(results[-1])

        # Add rows with max average income of agents
        for agent in agents:

            # Get the avg income of the agent
            print(f"Calculating average income of {agent}...")
            if read_from_data:
                # Get the max avg income saved in the evaluation data (this value is biased in old agents)
                training_data_agent = ReinforcementLearning.get_training_data(agent)
                if take_average:
                    results.append([agent, training_data_agent.get_max_avg_value()])
                    append_last_row_to_file()
                else:
                    for instance_name, income in training_data_agent.get_agent_best_instances_values().items():
                        results.append([f"{agent}_{instance_name}", training_data_agent.get_max_avg_value()])
                        append_last_row_to_file()
            else:
                # Calculate a fresh avg income using the best model
                rl = ReinforcementLearning(agent)
                incomes = []
                runs = rl.run_agent(ReinforcementLearning.get_all_fixed_instances(rl.config.num_dams))
                for run in runs:
                    if rl.config.action_type == "adjustments":
                        # Get the best solution achieved, not the latest one
                        sol = max(run.solutions, key=lambda s: s.get_objective_function())
                    else:
                        sol = run.solution
                    income = sol.get_objective_function()
                    if take_average:
                        incomes.append(income)
                    else:
                        results.append([f"{agent}_{run.instance.get_instance_name()}", income])
                        append_last_row_to_file()
                if take_average:
                    results.append([agent, sum(incomes) / len(incomes)])
                    append_last_row_to_file()

        training_data_baselines = TrainingData.create_empty()
        all_baselines = ReinforcementLearning.get_all_baselines(general_config)
        for baseline in all_baselines:
            if baseline.get_solver() in baselines:
                training_data_baselines += baseline

        # Add rows with baseline average incomes
        if take_average:
            for baseline_solver, baseline_avg_income in training_data_baselines.get_baseline_avg_values().items():  # noqa
                results.append([baseline_solver, baseline_avg_income])
                append_last_row_to_file()
        else:
            for baseline_solver, baseline_values in training_data_baselines.get_baseline_instances_values().items():  # noqa
                for instance_name, value in baseline_values.items():
                    results.append([f"{baseline_solver}_{instance_name}", value])
                    append_last_row_to_file()

        # Expand each row with the performance w.r.t. the baselines
        if take_average:
            for baseline_solver, baseline_avg_income in training_data_baselines.get_baseline_avg_values().items():  # noqa
                results[0] += [baseline_solver]  # Add the baseline to the table header
                for result in results[1:]:
                    result_avg_income = result[1]
                    perf_over_baseline = (result_avg_income - baseline_avg_income) / baseline_avg_income  # noqa
                    result += [f"{perf_over_baseline:+.2%}"]
        else:
            for baseline_solver, baseline_values in training_data_baselines.get_baseline_instances_values().items():  # noqa
                results[0] += [baseline_solver]
                for result in results[1:]:
                    result_instance = result[0].split("_")[1]
                    result_income = result[1]
                    baseline_income = baseline_values[result_instance]
                    perf_over_baseline = (result_income - baseline_income) / baseline_income  # noqa
                    result += [f"{perf_over_baseline:+.2%}"]

        # Turn results to string and print them
        for line in results:
            line = [f'{int(round(el)):,}' if isinstance(el, float) else el for el in line]
            print(';'.join(line))

        return results

    @staticmethod
    def print_spaces(agents_regex_filter: str | list[str] = '.*', permutation: str = 'AGORT'):

        """
        Print the shapes of the action and observation spaces of each agent
        :param agents_regex_filter:
        :param permutation:
        :return:
        """

        print("agent;action_space;obs_space")
        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        for agent in agents:
            config = ReinforcementLearning.get_config(
                ReinforcementLearning.extract_substrings(agent)
            )
            env = RLEnvironment(
                config=config,
                projector=Projector.create_projector(config),
                path_constants=CONSTANTS_PATH.format(num_dams=config.num_dams),
                path_historical_data=ReinforcementLearning.train_data_path,
                update_observation_record=False
            )
            print(f"{agent};{env.action_space.shape};{env.observation_space.shape}")

    @staticmethod
    def multi_level_sorting_key(agent: str, permutation: str) -> tuple:

        """
        Breaks down string "rl-A1G0O22R1T1" into the numbers (A) "1", (G) "0", (O) "22", (R) "1", (T) "1",
        and then sorts these numbers by the given permutation of the letters AGORT.

        :param agent: Input agent ID string
        :param permutation: Permutation of the config letters, for example 'AGORT' or 'GTOAR'
        """

        substrings = ReinforcementLearning.extract_substrings(agent)
        sorted_values = [substrings[letter] for letter in permutation]
        return tuple(sorted_values)

    @staticmethod
    def get_avg_training_time(agents_regex_filter: str | list[str] = '.*', permutation: str = 'AGORT') -> float:

        """
        Get the average training time of all agents matching the given regex
        """

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        training_times = ReinforcementLearning.get_training_times(agents)

        return sum(training_times) / len(training_times)

    @staticmethod
    def get_training_times(agents: list[str]) -> list[float]:

        """
        Get the training times of the given agents in minutes
        """

        training_times = []
        for agent in agents:
            training_data = ReinforcementLearning.get_training_data(agent)
            training_times.append(training_data.get_training_time())
        return training_times

    @staticmethod
    def get_training_data(agent: str) -> TrainingData:

        """
        Get the training data of the given agent.
        """

        agent_folder = os.path.join(ReinforcementLearning.models_folder, agent)
        training_data_path = os.path.join(agent_folder, "training_data.json")
        training_data = TrainingData.from_json(training_data_path)
        return training_data

    @staticmethod
    def get_path_evaluations(agent: str) -> str:

        agent_folder = os.path.join(ReinforcementLearning.models_folder, agent)
        evaluation_data_path = os.path.join(agent_folder, "evaluations.npz")
        return evaluation_data_path

    @staticmethod
    def common_general_config(agents: list[str], warning_msg: str = '') -> str | None:

        """
        Get the general configuration shared by all given agents
        and give a warning if this is not the case.
        """

        agents_substrings = [ReinforcementLearning.extract_substrings(agent) for agent in agents]
        agents_general = {substrings['G'] for substrings in agents_substrings}
        if len(agents_general) == 1:
            general_config = agents_general.pop()
        else:
            warnings.warn(f"Agents have different general configurations. {warning_msg}")
            general_config = None
        return general_config

    @staticmethod
    def get_all_agents(regex_filter: str | list[str] = '.*', permutation: str = 'AGORT') -> list[str]:

        """
        Get all agent IDs in alphabetical order, filtering by the given regex pattern

        :param regex_filter: Regex pattern matched by all desired agents, or list of regex patterns
        :param permutation: Order in which the agents are returned
        :return: List of agent IDs in the desired order
        """

        parent_directory = ReinforcementLearning.models_folder
        all_items = os.listdir(parent_directory)

        if isinstance(regex_filter, list):
            regex_filter = "$|".join(regex_filter) + "$"

        # Filter to take only the existing directories and those matching the regex pattern
        regex = re.compile(regex_filter)
        all_models = [
            item for item in all_items if os.path.isdir(os.path.join(parent_directory, item)) and regex.match(item)
        ]

        all_models.sort(
            key=lambda model: ReinforcementLearning.multi_level_sorting_key(model, permutation=permutation)
        )

        return all_models

    @staticmethod
    def get_all_baselines(general_config: str) -> list[Solution]:

        """
        Scan the folder with solutions that act as baselines for the RL agents.

        :param general_config: General configuration ("G0" or "G1")
        """

        return get_all_baselines(general_config)

    @staticmethod
    def get_all_configs(config_letter: str, relevant_digits: int = None) -> list[str]:

        """
        Get all config names (e.g., "A0", "A1", "A110", ...) of the given config letter (e.g., "A")
        :param config_letter:
        :param relevant_digits: Number of digits for which two configs are considered different
            (default is to treat all digits as relevant)
        :return:
        """

        config_names = []
        config_names_explored = set()
        config_path, _ = ReinforcementLearning.configs_info[config_letter]
        for file in os.listdir(config_path):
            if file.endswith('.json'):
                filename = os.path.basename(file)  # Remove the head (path); Documents/book.txt -> book.txt
                filename = os.path.splitext(filename)[0]  # Remove the extension; book.txt -> book
                filename_relevant = filename[:(relevant_digits + 1)] if relevant_digits is not None else filename
                if filename_relevant not in config_names_explored:
                    config_names_explored.add(filename_relevant)
                    config_names.append(filename)
        return config_names

    @staticmethod
    def get_all_fixed_instances(num_dams: int) -> list[Instance]:
        """
        Get all fixed instances that are being used to evaluate the agents and baselines.
        These instances are Percentile00, Percentile10, ..., Percentile100, from driest to rainiest.
        """
        return get_all_instances(num_dams)

    @staticmethod
    def get_slope_intercept(agent: str, solver: str) -> tuple[float, float]:

        """
        Get the slope and intercept of the given solver's reward with respect to rl-greedy's reward
        :param agent:
        :param solver:
        :return:
        """

        def get_solution_avg_reward(sol: Solution, num_dams: int) -> float:
            """
            Get the reward per timestep of an agent imitating the solution
            :param sol:
            :param num_dams:
            :return:
            """
            run = rl.run_imitator(
                solution=sol, instance=Instance.from_name(sol.get_instance_name(), num_dams=num_dams)
            )
            avg_reward = sum(run.rewards_per_period) / len(run.rewards_per_period)
            return avg_reward

        rl = ReinforcementLearning(agent)
        solver_values = dict()
        greedy_values = dict()
        for baseline in ReinforcementLearning.get_all_baselines(rl.config_names['G']):
            if baseline.get_solver() == solver:
                solver_values[baseline.get_instance_name()] = get_solution_avg_reward(
                    baseline, num_dams=rl.config.num_dams
                )
            elif baseline.get_solver() == 'rl-greedy':
                greedy_values[baseline.get_instance_name()] = get_solution_avg_reward(
                    baseline, num_dams=rl.config.num_dams
                )

        sorted_greedy_values = dict(sorted(greedy_values.items()))  # noqa
        print(f"Instance values of rl-greedy:", sorted_greedy_values)
        x = np.array(list(sorted_greedy_values.values()))

        sorted_solver_values = dict(sorted(solver_values.items()))  # noqa
        print(f"Instance values of solver {solver}:", sorted_solver_values)
        y = np.array(list(sorted_solver_values.values()))

        slope, intercept = np.polyfit(x, y, 1)  # noqa
        print(
            f"Fitted line for solver {solver} in {rl.config_names['G']}: "
            f"{slope} * x + {intercept} | R = {np.corrcoef(x, y)[0, 1]}"
        )
        return slope, intercept

    @staticmethod
    def process_config(config: RLConfiguration, config_names: dict[str, str]) -> RLConfiguration:

        """
        Add missing information to the configuration:
        - the MILP or rl-random reward structures in the reward R23 and R24
        - the tuned hyperparameters given by RL Zoo

        :param config:
        :param config_names:
        :return:
        """

        config = ReinforcementLearning.add_reward_structures(config, config_names)
        config = ReinforcementLearning.add_hyperparams(config, config_names)
        return config

    @staticmethod
    def add_reward_structures(config: RLConfiguration, config_names: dict[str, str]) -> RLConfiguration:

        """
        Add the MILP or rl-random reward structures to the configuration
        (when using the reward R23 and R24)

        :param config:
        :param config_names:
        :return:
        """

        # Define the agent used to calculate the reward R1 structure of MILP and rl-random
        # Note that if the reward R23 or R24 is left unchanged, there would be an infinite loop of processing configs
        updated_config_names = config_names.copy()
        updated_config_names['R'] = 'R1'
        agent = ''.join(updated_config_names.values())

        if config.milp_reference and (config.milp_slope is None or config.milp_intercept is None):
            slope, intercept = ReinforcementLearning.get_slope_intercept(agent=agent, solver='MILP')
            config.milp_slope = slope
            config.milp_intercept = intercept
            print(f"Set MILP's slope to {config.milp_slope} and its intercept to {config.milp_intercept}.")

        if config.random_reference and (config.random_slope is None or config.random_intercept is None):
            slope, intercept = ReinforcementLearning.get_slope_intercept(agent=agent, solver='rl-random')
            config.random_slope = slope
            config.random_intercept = intercept
            print(f"Set rl-random's slope to {config.random_slope} and its intercept to {config.random_intercept}.")

        return config

    @staticmethod
    def add_hyperparams(config: RLConfiguration, config_names: dict[str, str]) -> RLConfiguration:

        """
        Add the tuned hyperparameters given by RL Zoo to the configuration
        (when hyperparams == "rl_zoo")

        :param config:
        :param config_names:
        :return:
        """

        if config.hyperparams != "rl_zoo":
            return config

        algo = config.algorithm.lower()
        action = config_names['A']
        general = config_names['G']
        norm_str = "normalize" if config.normalization else ""

        folder_path = ReinforcementLearning.best_hyperparams_folder
        filename = f"hyperparams_{algo}_{action}{general}O231R1T0{norm_str}_1.json"
        file_path = os.path.join(folder_path, filename)
        print(f"Loading tuned hyperparameters from file {filename} in folder {folder_path}...")
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                hyperparams = json.load(file)
        else:
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Replace keys with the specific attribute names in RLConfiguration
        if "learning_rate" in hyperparams:
            config.learning_rate = hyperparams["learning_rate"]
            del hyperparams["learning_rate"]
        if "lr_schedule" in hyperparams:
            config.lr_schedule_name = hyperparams["lr_schedule"]
            del hyperparams["lr_schedule"]
        if "buffer_size" in hyperparams:
            config.replay_buffer_size = hyperparams["buffer_size"]
            del hyperparams["buffer_size"]
        if "net_arch" in hyperparams:
            # The meaning of "tiny", "small", etc. is taken from flowing_basin/rl_zoo/rl_zoo3/hyperparams_opt.py
            actor_critic_layers = {
                "tiny": [64],
                "small": [64, 64],
                "medium": [256, 256],
                "big": [400, 300]
            }[hyperparams["net_arch"]]
            config.actor_layers = actor_critic_layers
            config.critic_layers = actor_critic_layers
            del hyperparams["net_arch"]
        if "activation_fn" in hyperparams:
            config.activation_fn_name = hyperparams["activation_fn"]
            del hyperparams["activation_fn"]
        if "log_std_init" in hyperparams:
            config.log_std_init = hyperparams["log_std_init"]
            del hyperparams["log_std_init"]
        if "ortho_init" in hyperparams:
            config.ortho_init = hyperparams["ortho_init"]
            del hyperparams["ortho_init"]

        # NOTE: I attempted to use gSDE when tuning with continuous actions in RL Zoo,
        # but I forgot to put `use_sde: true` in the .yml files.
        # This means gSDE was actually not used during tuning
        # and that the `log_std_init` and `sde_sample_freq` values suggested by RL Zoo actually have no effect,
        # so config.use_sde should keep its default value of False.

        # Put the remaining hyperparameters in the 'other hyperparams' attribute of RLConfiguration
        config.hyperparams = hyperparams

        return config

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
                collection_pos = ReinforcementLearning.obs_collection_pos
                maybe_collection_code = config_name[collection_pos]
                if maybe_collection_code in ReinforcementLearning.obs_collection_codes:
                    # Replace the observation collection method for projector by a generic `X` (e.g., "O211" -> "O2X1")
                    config_name = config_name[:collection_pos] + "X" + config_name[collection_pos + 1:]
            config_path = os.path.join(config_folder, config_name + ".json")
            configs.append(config_class.from_json(config_path))
        config = RLConfiguration.from_components(configs)
        config = ReinforcementLearning.process_config(config, config_names)
        return config




