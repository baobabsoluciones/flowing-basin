from cornflow_client.core import SolutionCore
from cornflow_client.core.tools import load_json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


class Training(SolutionCore):

    """
    Class to save the training data of an RL agent
    """

    schema = load_json(
        os.path.join(os.path.dirname(__file__), "../schemas/training.json")
    )

    @classmethod
    def from_dict(cls, data: list) -> "Training":

        # Change list into dictionary indexed by agent name
        data_p = {el["id"]: el for el in data}

        return cls(data_p)

    def to_dict(self):

        # Change dictionary of dams into list, to undo de changes of from_dict
        # Use pickle to work with a copy of the data and avoid changing the property of the class
        data_p = pickle.loads(pickle.dumps(self.data, -1))
        data_p = list(data_p.values())

        return data_p

    def check_inconsistencies(self):

        inconsistencies = dict()

        for agent_id in self.get_agent_ids():
            num_timesteps = len(self.get_timesteps(agent_id))
            num_values = len(self.data[agent_id]['values'])
            if num_timesteps != num_values:
                inconsistencies.update(
                    {
                        f"The number of values for agent {agent_id} does snot equal the number of timesteps":
                            f"The number of values for agent {agent_id} is {num_values}, "
                            f"and the number of timesteps is {num_timesteps}"
                    }
                )

        return inconsistencies

    def check(self):

        inconsistencies = self.check_inconsistencies()
        schema_errors = self.check_schema()
        if schema_errors:
            inconsistencies.update(
                {"The given data does not follow the schema": schema_errors}  # noqa
            )

        return inconsistencies

    def get_agent_ids(self) -> list[str]:

        return list(self.data.keys())

    def get_agent_name(self, agent_id: str) -> str | None:

        name = self.data[agent_id].get("name")
        if name is None:
            name = agent_id

        return name

    def get_timesteps(self, agent_id: str) -> list[float] | None:

        """
        Get the time steps of training for each evaluation value
        """

        timesteps = self.data[agent_id]['timesteps']

        return timesteps

    def get_time_stamps(self, agent_id: str) -> list[float] | None:

        """
        Get the time stamps for the evaluation results
        """

        time_stamps = self.data[agent_id]['time_stamps']

        return time_stamps

    def get_avg_incomes(self, agent_id: str, instances: list[str] = None) -> list[float]:

        """
        Average income of given agent across all instances
        for each timestep
        """

        if instances is None:
            avg_incomes = [
                sum(instance['income'] for instance in timestep) /
                len([instance['income'] for instance in timestep])
                for timestep in self.data[agent_id]['values']
            ]
        else:
            avg_incomes = [
                sum(instance['income'] for instance in timestep if instance['instance'] in instances) /
                len([instance['income'] for instance in timestep if instance['instance'] in instances])
                for timestep in self.data[agent_id]['values']
            ]

        return avg_incomes

    def get_avg_acc_rewards(self, agent_id: str, instances: list[str] = None) -> list[float]:

        """
        Average income of given agent across all instances
        for each timestep
        """

        if instances is None:
            avg_incomes = [
                sum(instance['acc_reward'] for instance in timestep) /
                len([instance['acc_reward'] for instance in timestep])
                for timestep in self.data[agent_id]['values']
            ]
        else:
            avg_incomes = [
                sum(instance['acc_reward'] for instance in timestep if instance['instance'] in instances) /
                len([instance['acc_reward'] for instance in timestep if instance['instance'] in instances])
                for timestep in self.data[agent_id]['values']
            ]

        return avg_incomes

    def plot_training_curves(self, ax: plt.Axes, instances: list[str] = None):

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Average income (€)")
        ax.set_title(f"Evaluation")

        twinax = ax.twinx()
        twinax.set_ylabel("Average accumulated rewards")

        colors = [plt.get_cmap('hsv')(color) for color in np.linspace(0, 1, len(self.get_agent_ids()), endpoint=False)]
        for agent_id, color in zip(self.get_agent_ids(), colors):
            name = self.get_agent_name(agent_id)
            timesteps = self.get_timesteps(agent_id)
            avg_income = self.get_avg_incomes(agent_id, instances)
            avg_acc_reward = self.get_avg_acc_rewards(agent_id, instances)
            ax.plot(
                timesteps, avg_income, color=color, linestyle='-', label=f"Average income of {name}"
            )
            twinax.plot(
                timesteps, avg_acc_reward, color=color, linestyle='--', label=f"Average accumulated rewards of {name}"
            )
        ax_lines, ax_labels = ax.get_legend_handles_labels()
        twinax_lines, twinax_labels = twinax.get_legend_handles_labels()
        ax.legend(ax_lines + twinax_lines, ax_labels + twinax_labels, loc=0)
