from flowing_basin.core import Instance, TrainingData
from flowing_basin.solvers.rl import RLConfiguration, RLRun
from flowing_basin.solvers.rl.feature_extractors import Projector
import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
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
            policy_id: str, training_data: TrainingData = None, verbose: int = 1
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
        self.start_time = perf_counter()

        if training_data is None:
            self.training_data = TrainingData.create_empty_fixed_instances(
                agent_id=self.policy_id, config=self.config.to_dict()
            )
            self.start_timestamp_offset = 0.
            self.start_timestep_offset = 0
        else:
            self.training_data = training_data
            try:
                self.start_timestamp_offset = self.training_data.get_time_stamps()[-1]
                self.start_timestep_offset = self.training_data.get_timesteps()[-1]
            except IndexError:
                self.start_timestamp_offset = 0.
                self.start_timestep_offset = 0

    def _on_step(self) -> bool:

        if self.n_calls % self.eval_freq == 0:

            incomes = []
            acc_rewards = []
            new_values = []

            for run in self.runs:

                # Sync training and run's env if there is VecNormalize
                # This is also done in SB3's ``EvalCallback``
                if self.model.get_vec_normalize_env() is not None:
                    try:
                        sync_envs_normalization(self.training_env, run.env)
                    except AttributeError as e:
                        raise AssertionError(
                            "Training env and RLRun env are not wrapped the same way."
                        ) from e

                run.solve(self.model.policy)
                income = run.solution.get_objective_function()
                acc_reward = sum(run.rewards_per_step)

                incomes.append(income)
                acc_rewards.append(acc_reward)
                new_values.append(
                    {
                        "instance": run.instance.get_instance_name(),
                        "income": income,
                        "acc_reward": acc_reward
                    }
                )

            # Add to training data
            self.training_data.add_timestep_fixed_instances(
                agent_id=self.policy_id,
                new_timestep=self.n_calls + self.start_timestep_offset,
                new_time_stamp=perf_counter() - self.start_time + self.start_timestamp_offset,
                new_values=new_values
            )

            # Add to tensorboard
            self.logger.record("training_data/income", sum(incomes) / len(incomes))
            self.logger.record("training_data/acc_reward", sum(acc_rewards) / len(acc_rewards))

        return True
