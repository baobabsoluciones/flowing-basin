from flowing_basin.core import Instance, Solution, Experiment
from .rl_env import RLEnvironment, RLConfiguration
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
from matplotlib import pyplot as plt
import warnings
import os
import tempfile


class RLTrain(Experiment):

    def __init__(
            self,
            config: RLConfiguration,
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
        self.model = SAC("MlpPolicy", self.train_env, verbose=1)

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

    def solve(self, num_episodes: int, path_agent: str, periodic_evaluation: bool = False, options: dict = None) -> dict:  # noqa

        """
        Train the model and save it in the given path.
        """

        episode_length = self.train_env.instance.get_largest_impact_horizon()
        total_timesteps = num_episodes * episode_length

        # Set evaluation callback
        temp_dir = tempfile.TemporaryDirectory()
        eval_callback = None
        if periodic_evaluation:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=None,
                log_path=temp_dir.name,
                eval_freq=self.config.eval_ep_freq * episode_length,
                n_eval_episodes=self.config.eval_num_episodes,
                deterministic=True,
                render=False
            )

        # Train and save model
        self.model.learn(total_timesteps=total_timesteps, log_interval=self.config.log_ep_freq, callback=eval_callback)
        self.model.save(path_agent)

        # Save evaluation data
        if periodic_evaluation:
            with np.load(os.path.join(temp_dir.name, "evaluations.npz")) as data:
                # for file in data.files:
                #     print(file, data[file])
                self.eval_episodes = data["timesteps"] // episode_length
                self.eval_avg_results = data["results"].mean(axis=1)
        temp_dir.cleanup()

        return dict()

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


