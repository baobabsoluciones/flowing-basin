from flowing_basin.core import Instance, Solution, Experiment
from flowing_basin.solvers.rl import RLEnvironment, RLConfiguration
from flowing_basin.solvers.rl.feature_extractors.convolutional import VanillaCNN
from flowing_basin.solvers.rl.callbacks import SaveOnBestTrainingRewardCallback, IncomeEvalCallback
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np
from matplotlib import pyplot as plt
import os
import json


class RLTrain(Experiment):

    def __init__(
            self,
            config:
            RLConfiguration,
            path_constants: str,
            path_train_data: str,
            path_test_data: str,
            path_folder: str = '.',
            paths_power_models: dict[str, str] = None,
            instance: Instance = None,
            solution: Solution = None,
            verbose: int = 1,
    ):

        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        self.verbose = verbose

        # Configuration, environment and model (RL agent)
        self.config = config
        self.path_folder = path_folder
        self.train_env = RLEnvironment(
            config=self.config,
            path_constants=path_constants,
            path_historical_data=path_train_data,
            paths_power_models=paths_power_models,
            instance=instance,
        )

        os.makedirs(self.path_folder, exist_ok=True)
        filepath_log = os.path.join(self.path_folder, ".")
        self.train_env = Monitor(self.train_env, filename=filepath_log)

        self.model = None
        self.initialize_agent()

        # Variables for periodic evaluation of agent during training
        self.eval_env = RLEnvironment(
            config=self.config,
            path_constants=path_constants,
            path_historical_data=path_test_data,
            paths_power_models=paths_power_models,
            instance=instance,
        )
        self.eval_env = Monitor(self.eval_env)

    def initialize_agent(self):

        if self.config.feature_extractor == 'MLP':

            policy_type = "MlpPolicy"
            policy_kwargs = None

        elif self.config.feature_extractor == 'CNN':

            policy_type = "CnnPolicy"
            fe_variant = self.config.get("feature_extractor_variant", "vanilla")

            extractor_class = None
            if fe_variant == "vanilla":
                extractor_class = VanillaCNN

            policy_kwargs = dict(
                features_extractor_class=extractor_class,
                features_extractor_kwargs=dict(features_dim=128),
            )

        else:

            raise NotImplementedError(
                f"Feature extractor of type {self.config.feature_extractor} is not supported. Either MLP or CNN"
            )

        self.model = SAC(
            policy_type, self.train_env,
            verbose=1, tensorboard_log=self.config.get("tensorboard_log"), policy_kwargs=policy_kwargs
        )
        
    def solve(self, num_episodes: int, options: dict) -> dict:  # noqa

        """
        Train the model and save it in the given path.
        :param num_episodes: Nuber of episodes (days) in which to train the agent
        :param options: Logging options
        """

        episode_length = self.train_env.instance.get_largest_impact_horizon()
        total_timesteps = num_episodes * episode_length

        # Set evaluation callback
        #ToDo ensure correct initialization of callback functions. Are they redundant?
        if options['evaluation'] == 'reward':
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=None,
                log_path=self.path_folder,
                eval_freq=options['eval_ep_freq'] * episode_length,
                n_eval_episodes=options['eval_num_episodes'],
                deterministic=True,
                render=False,
                verbose=self.verbose
            )
        elif options['evaluation'] == 'income':
            eval_callback = IncomeEvalCallback(
                eval_freq=options['eval_ep_freq'] * episode_length,
                instances=options['evaluation_instances'],
                config=self.config,
                verbose=self.verbose
            )
        else:
            raise ValueError(
                f"Invalid value for `evaluation`: '{options['evaluation']}'. "
                f"Allowed values are None, 'reward' or 'income'."
            )

        # Set checkpoint callback
        checkpoint_callback = SaveOnBestTrainingRewardCallback(
            check_freq=options['checkpoint_ep_freq'] * episode_length,
            log_dir=self.path_folder,
            verbose=self.verbose
        )

        # Train model
        #ToDO I dont want to override your logic, decide where to integrate (config, arg in here? the log directory etc)
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=options['log_ep_freq'],
            callback=CallbackList([checkpoint_callback, eval_callback]),
            tb_log_name=self.config.get("tensorboard_log", "SAC")
        )

        # Save model
        filepath_agent = os.path.join(self.path_folder, "model.zip")
        self.model.save(filepath_agent)
        if self.verbose >= 1:
            print(f"Created ZIP file '{filepath_agent}'.")

        # Save configuration
        filepath_config = os.path.join(self.path_folder, "config.json")
        self.config.to_json(filepath_config)
        if self.verbose >= 1:
            print(f"Created JSON file '{filepath_config}'.")

        # Save evaluation data
        filepath_evaluation = os.path.join(self.path_folder, "evaluation.json")
        if options['evaluation'] == 'reward':
            with np.load(os.path.join(self.path_folder, "evaluations.npz")) as data:
                # for file in data.files:
                #     print(file, data[file])
                evaluation_data = {
                    "episode": (data["timesteps"] // episode_length).tolist(),
                    "average_result": data["results"].mean(axis=1).tolist(),
                }
            with open(filepath_evaluation, "w") as f:
                json.dump(evaluation_data, f, indent=4, sort_keys=True)
            if self.verbose >= 1:
                print(f"Created JSON file '{filepath_evaluation}'.")
        elif options['evaluation'] == 'income':
            eval_callback.save_evaluation_data(filepath_evaluation)

        return dict()

    @staticmethod
    def plot_evaluation_data(filepath: str, ax: plt.Axes):

        """
        Plot the data of the evaluation JSON file
        """

        IncomeEvalCallback.plot_evaluation_data(filepath, ax)
        # TODO: provide support if options['evaluation'] == 'reward' instead of 'income'


