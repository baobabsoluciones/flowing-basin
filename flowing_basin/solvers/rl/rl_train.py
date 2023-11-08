from flowing_basin.core import Instance, Solution, Experiment
from .rl_env import RLEnvironment, RLConfiguration
from .feature_extractors.convolutional import VanillaCNN
from .callbacks import SaveOnBestTrainingRewardCallback
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np
from matplotlib import pyplot as plt
import warnings
import os
import tempfile
import json


class RLTrain(Experiment):

    def __init__(
            self,
            config:
            RLConfiguration,
            path_constants: str,
            path_train_data: str,
            path_test_data: str,
            paths_power_models: dict[str, str] = None,
            instance: Instance = None,
            solution: Solution = None,
    ):

        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        # Configuration, environment and model (RL agent)
        self.config = config
        self.train_env = RLEnvironment(
            config=self.config,
            path_constants=path_constants,
            path_historical_data=path_train_data,
            paths_power_models=paths_power_models,
            instance=instance,
        )
        self.train_env = Monitor(self.train_env, '.')
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
        self.eval_episodes = None
        self.eval_avg_results = None

    def initialize_agent(self):


        if self.config.feature_extractor in ['mlp', 'perceptron']:
            policy_type = "MlpPolicy"
            policy_kwargs = None
        elif self.config.feature_extractor in ['cnn', 'convolutional']:
            policy_type = "CnnPolicy"
            fe_variant = self.config.get("feature_extractor_variant", "vanilla")
            if fe_variant == "vanilla":
                extractor_class = VanillaCNN

            policy_kwargs = dict(
                features_extractor_class=extractor_class,
                features_extractor_kwargs=dict(features_dim=128),
            )
        else:
            raise ValueError(f"Feature extractor of type {self.config.feature_extractor} is not supported. Either mlp or cnn")

        self.model = SAC(policy_type, self.train_env, verbose=1, tensorboard_log=self.config.get("tensorboard_log"), policy_kwargs=policy_kwargs)
        
    def solve(self, num_episodes: int, options: dict) -> dict:  # noqa

        """
        Train the model and save it in the given path.
        :param num_episodes: Nuber of episodes (days) in which to train the agent
        :param options: Logging options
        """

        episode_length = self.train_env.instance.get_largest_impact_horizon()
        total_timesteps = num_episodes * episode_length

        # Set evaluation callback
        temp_dir = None
        eval_callback = None
        #ToDo ensure correct initialization of callback functions. Are they redundant?
        if options['periodic_evaluation'] == 'reward':
            temp_dir = tempfile.TemporaryDirectory()
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=None,
                log_path=temp_dir.name,
                eval_freq=options['eval_ep_freq'] * episode_length,
                n_eval_episodes=options['eval_num_episodes'],
                deterministic=True,
                render=False
            )
        elif options['periodic_evaluation'] == 'income':
            pass
        else:
            raise ValueError(
                f"Invalid value for `periodic_evaluation`: '{options['periodic_evaluation']}'. "
                f"Allowed values are None, 'reward' or 'income'."
            )

        # Set checkpoint callback
        checkpoint_callback = SaveOnBestTrainingRewardCallback(check_freq=self.config.eval_ep_freq * episode_length,log_dir='.', verbose=1)

        # Train model
        #ToDO I dont want to override your logic, decide where to integrate (config, arg in here? the log directory etc)
        self.model.learn(total_timesteps=total_timesteps, log_interval=options['log_ep_freq'], callback=CallbackList([checkpoint_callback, eval_callback]),
                         tb_log_name=self.config.get("tensorboard_log", "SAC"))

        # Store evaluation data
        if options['periodic_evaluation'] == 'reward':
            with np.load(os.path.join(temp_dir.name, "evaluations.npz")) as data:
                # for file in data.files:
                #     print(file, data[file])
                self.eval_episodes = data["timesteps"] // episode_length
                self.eval_avg_results = data["results"].mean(axis=1)
        temp_dir.cleanup()

        return dict()

    def save_model(self, path_agent: str):

        """
        Save the agent's neural network parameters
        in a .zip file (behaviour of SB3's method)
        :param path_agent: Path where to save the agent
        """

        self.model.save(path_agent)

    def plot_training_curve(self) -> plt.Axes:

        if self.eval_episodes is None or self.eval_avg_results is None:
            warnings.warn(
                "No evaluation data found. Please make sure you have called the `solve` method of `RLTrain` "
                "with `periodic_evaluation=True`."
            )
            return  # noqa

        fig, ax = plt.subplots()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average result")
        ax.plot(self.eval_episodes, self.eval_avg_results)

        return ax

    def save_training_data(self, path: str):

        if self.eval_episodes is None or self.eval_avg_results is None:
            warnings.warn(
                "No evaluation data found. Please make sure you have called the `solve` method of `RLTrain` "
                "with `periodic_evaluation=True`."
            )
            return

        training_data = {
            "episode": self.eval_episodes.tolist(),
            "average_result": self.eval_avg_results.tolist(),
        }
        with open(path, "w") as f:
            json.dump(training_data, f, indent=4, sort_keys=True)


