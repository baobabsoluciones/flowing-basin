from flowing_basin.core import Instance, Training
from flowing_basin.solvers.rl import RLConfiguration, RLRun
import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
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
        self.save_path = os.path.join(log_dir, "model_best.zip")
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
                    # Example for saving best model
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
    :param baseline_policy: Policy with which to compare the actual model (can be "random")
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
            self, eval_freq: int, config: RLConfiguration, instances: list[str],
            baseline_policy: BasePolicy | str, verbose: int = 1
    ):

        super(TrainingDataCallback, self).__init__(verbose)

        self.eval_freq = eval_freq
        self.baseline_policy = baseline_policy
        self.config = config

        instance_objects = [
            Instance.from_json(instance)
            for instance in instances
        ]
        self.runs = [
            RLRun(instance_object, self.config)
            for instance_object in instance_objects
        ]

        self.start_time = perf_counter()
        self.policy_names = ["model", "random"]
        self.timesteps = []
        self.time_stamps = []
        self.values = {policy_name: [] for policy_name in self.policy_names}
        self.training_data = None

    def _on_step(self) -> bool:

        if self.n_calls % self.eval_freq == 0:

            policies = {
                "model": self.model.policy,
                "random": self.baseline_policy
            }
            for policy_name, policy in policies.items():
                timestep_values = []
                for run in self.runs:
                    info = run.solve(policy)
                    income = run.solution.get_objective_function()
                    acc_reward = sum(info['rewards'])
                    timestep_values.append(
                        {"instance": run.env.instance.get_instance_name(), "income": income, "acc_reward": acc_reward}
                    )
                self.values[policy_name].append(timestep_values)

            self.timesteps.append(self.n_calls)
            self.time_stamps.append(perf_counter() - self.start_time)

        return True

    def _on_training_end(self):

        """
        Fill the training data attribute
        """

        self.training_data = Training.from_dict(
            {
                "configuration": self.config.to_dict(),
                "timesteps": self.timesteps,
                "time_stamps": self.time_stamps,
                "agents": [
                    {
                        "agent": policy_name,
                        "values": self.values[policy_name]
                    }
                    for policy_name in self.policy_names
                ]
            }
        )
