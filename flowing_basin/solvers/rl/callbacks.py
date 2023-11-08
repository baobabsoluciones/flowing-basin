from flowing_basin.core import Instance
from flowing_basin.solvers.rl import RLConfiguration, RLRun
import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from time import perf_counter
import json
import matplotlib.pyplot as plt


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


class IncomeEvalCallback(BaseCallback):

    """
    Callback for evaluating a model (the evaluation is done every ``eval_freq`` steps)
    using the RL environment's rewards (similar to SB3's ``EvalCallback``) and the actual income
    in the given instances.

    :param eval_freq:
    :param instances: Paths to the instances to solve periodically
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, eval_freq: int, config: RLConfiguration, instances: list[str], verbose: int = 1):

        super(IncomeEvalCallback, self).__init__(verbose)

        self.eval_freq = eval_freq

        instance_objects = [
            Instance.from_json(instance)
            for instance in instances
        ]
        self.runs = [
            RLRun(instance_object, config)
            for instance_object in instance_objects
        ]

        self.start_time = perf_counter()
        self.steps = []
        self.time_stamps = []
        self.all_incomes = []
        self.all_acc_rewards = []

    def _on_step(self) -> bool:

        if self.n_calls % self.eval_freq == 0:

            incomes = []
            acc_rewards = []
            for run in self.runs:
                info = run.solve(self.model)
                income = run.solution.get_objective_function()
                acc_reward = sum(info['rewards'])
                incomes.append(income)
                acc_rewards.append(acc_reward)

            self.steps.append(self.n_calls)
            self.time_stamps.append(perf_counter() - self.start_time)
            self.all_incomes.append(incomes)
            self.all_acc_rewards.append(acc_rewards)

        return True

    def save_evaluation_data(self, filepath: str):

        """
        Save the recorded data in the given JSON file
        """

        evaluation_data = {
            "timesteps": self.steps,
            "time_stamps": self.time_stamps,
            "incomes": self.all_incomes,
            "accumulated_rewards": self.all_acc_rewards,
        }
        with open(filepath, "w") as f:
            json.dump(evaluation_data, f, indent=4)
        if self.verbose >= 1:
            print(f"Created JSON file '{filepath}'.")

    @staticmethod
    def plot_evaluation_data(filepath: str, ax: plt.Axes):

        with open(filepath, "r") as f:
            evaluation_data = json.load(f)

        timesteps = evaluation_data['timesteps']
        avg_income = [
            sum(incomes) / len(incomes)
            for incomes in evaluation_data['incomes']
        ]
        avg_acc_reward = [
            sum(acc_rewards) / len(acc_rewards)
            for acc_rewards in evaluation_data['accumulated_rewards']
        ]

        ax.plot(timesteps, avg_income, color='b', linestyle='-', label="Average income (€)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Average income (€)")
        ax.set_title(f"Evaluation")
        ax.legend()

        twinax = ax.twinx()
        twinax.plot(timesteps, avg_acc_reward, color='b', linestyle='--', label="Average accumulated rewards")
        twinax.set_ylabel("Average accumulated rewards")
        twinax.legend()
