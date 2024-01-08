from flowing_basin.core import Instance, TrainingData
from flowing_basin.solvers.rl import RLConfiguration, RLRun
from flowing_basin.solvers.rl.feature_extractors import Projector
import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from time import perf_counter


class SaveOnBestTrainingRewardCallback(BaseCallback):

    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):

        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)

        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model_checkpoint.zip")
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:

                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - "
                        f"Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


class TrainingDataCallback(BaseCallback):

    """
    Callback for evaluating a model (the evaluation is done every ``eval_freq`` steps)
    using the RL environment's rewards (similar to SB3's ``EvalCallback``) and the actual income
    in the given instances.

    :param eval_freq:
    :param config: Configuration for the RLEnvironment
    :param instances: Paths to the instances to solve periodically
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
            self, eval_freq: int, config: RLConfiguration, projector: Projector, instances: list[str],
            policy_id: str, verbose: int = 1
    ):

        super(TrainingDataCallback, self).__init__(verbose)

        self.eval_freq = eval_freq
        self.config = config

        instance_objects = [Instance.from_name(instance) for instance in instances]
        self.runs = [
            RLRun(instance=instance_object, config=self.config, projector=projector)
            for instance_object in instance_objects
        ]

        self.policy_id = policy_id
        self.values = []
        self.timesteps = []
        self.time_stamps = []
        self.start_time = perf_counter()
        self.training_data = None

    def _on_step(self) -> bool:

        if self.n_calls % self.eval_freq == 0:

            timestep_values = []
            incomes = []
            acc_rewards = []
            for run in self.runs:
                info = run.solve(self.model.policy)
                income = run.solution.get_objective_function()
                acc_reward = sum(info['rewards'])
                timestep_values.append(
                    {"instance": run.instance.get_instance_name(), "income": income, "acc_reward": acc_reward}
                )
                incomes.append(income)
                acc_rewards.append(acc_reward)

            # Add to training data
            self.values.append(timestep_values)
            self.timesteps.append(self.n_calls)
            self.time_stamps.append(perf_counter() - self.start_time)

            # Add to tensorboard
            self.logger.record("training_data/income", sum(incomes) / len(incomes))
            self.logger.record("training_data/acc_reward", sum(acc_rewards) / len(acc_rewards))

        return True

    def _on_training_end(self):

        """
        Fill the training data attribute
        """

        self.training_data = TrainingData.from_dict(
            [
                {
                    "id": self.policy_id,
                    "fixed_instances": {
                        "values": self.values,
                        "timesteps": self.timesteps,
                        "time_stamps": self.time_stamps,
                    },
                    "configuration": self.config.to_dict()
                }
            ]
        )
