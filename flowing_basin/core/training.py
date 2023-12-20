from cornflow_client.core import SolutionCore
from cornflow_client.core.tools import load_json
from flowing_basin.core import Solution
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


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
            for instances in ['fixed', 'random']:
                if not self.has_agent_instances(agent_id, instances):
                    continue
                data = self.data[agent_id][self.get_instances_name(instances)]
                num_timesteps = len(data['timesteps'])
                num_values = len(data['values'])
                if num_timesteps != num_values:
                    inconsistencies.update(
                        {
                            f"The number of values for agent {agent_id} in instances {instances} "
                            f"does not equal the number of timesteps":
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

        return [key for key in self.data.keys() if key != "baselines"]

    def get_instances_ids(self, instances: list[str] | str) -> list[str]:

        """
        Get the names or IDs of the instances
        :param: Can be 'fixed', 'random', or a list of specific fixed instances
        """

        if isinstance(instances, list):
            return instances
        elif instances == 'fixed':
            # Fixed instances that appear in the first timestep of the first agent
            values_first_timestep = self.data[self.get_agent_ids()[0]][self.get_instances_name('fixed')]['values'][0]
            fixed_instances = [value['instance'] for value in values_first_timestep]
            return fixed_instances
        else:
            return []

    def get_instances_name(self, instances: list[str] | str) -> str:

        """
        Returns 'fixed_instances' or 'random_instances'

        :param: Can be 'fixed', 'random', or a list of specific fixed instances
        """

        if isinstance(instances, str):
            valid_instances = {'fixed', 'random'}
            if instances not in valid_instances:
                raise ValueError(f"Invalid value for `instances`: '{instances}'. Allowed values are {valid_instances}")

        instances_name = instances if isinstance(instances, str) else 'fixed'
        instances_name = dict(fixed='fixed_instances', random='random_instances')[instances_name]

        return instances_name

    def has_agent_instances(self, agent_id: str, instances: list[str] | str) -> bool:

        """
        Checks whether the provided data has values stored for the given agent and instance
        """

        instances_name = self.get_instances_name(instances)
        return self.data[agent_id].get(instances_name) is not None

    def has_agent_instances_values(self, agent_id: str, instances: list[str] | str, values: str):

        if not self.has_agent_instances(agent_id, instances):
            return False

        # Check if data has that value stored for the first timestep and the first instance
        instances_name = self.get_instances_name(instances)
        return self.data[agent_id][instances_name]['values'][0][0].get(values) is not None

    def check_has_agent_instances(self, agent_id: str, instances: list[str] | str):

        if not self.has_agent_instances(agent_id, instances):
            raise ValueError(f"Provided data has no info on instances `{instances}` for agent `{agent_id}`.")

    def check_has_agent_instances_values(self, agent_id: str, instances: list[str] | str, values: str):

        if not self.has_agent_instances_values(agent_id, instances, values):
            raise ValueError(
                f"Provided data does not save the value `{values}` on instances `{instances}` for agent `{agent_id}`."
            )

    def get_agent_name(self, agent_id: str) -> str | None:

        name = self.data[agent_id].get("name")
        if name is None:
            name = agent_id

        return name

    def get_timesteps(self, agent_id: str, instances: list[str] | str) -> list[float] | None:

        """
        Get the time steps of training for each evaluation value
        """

        self.check_has_agent_instances(agent_id, instances)
        timesteps = self.data[agent_id][self.get_instances_name(instances)]['timesteps']

        return timesteps

    def get_time_stamps(self, agent_id: str, instances: list[str] | str) -> list[float] | None:

        """
        Get the time stamps for the evaluation results

        :param agent_id:
        :param instances:
        """

        self.check_has_agent_instances(agent_id, instances)
        time_stamps = self.data[agent_id][self.get_instances_name(instances)]['time_stamps']

        return time_stamps

    def get_avg_values(self, agent_id: str, instances: list[str] | str, values: str) -> list[float]:

        """
        Average values (incomes, accumulated rewards...) of given agent across all instances
        for each timestep

        :param values: Can be 'income' or 'acc_reward'
        :param agent_id:
        :param instances:
        """

        # Check values
        valid_values = {'income', 'acc_reward'}
        if values not in valid_values:
            raise ValueError(f"Invalid value for `values`: {values}. Allowed values are {valid_values}")

        self.check_has_agent_instances(agent_id, instances)
        self.check_has_agent_instances_values(agent_id, instances, values)

        timesteps = self.data[agent_id][self.get_instances_name(instances)]["values"]
        if isinstance(instances, str):  # Random or fixed
            avg_values = [
                sum(instance[values] for instance in timestep) /
                len([instance[values] for instance in timestep])
                for timestep in timesteps
            ]
        else:  # List of specific fixed instances
            avg_values = [
                sum(instance[values] for instance in timestep if instance['instance'] in instances) /
                len([instance[values] for instance in timestep if instance['instance'] in instances])
                for timestep in timesteps
            ]

        return avg_values

    def get_baseline_values(self, instances: list[str] | str, values: str) -> dict[str, float]:

        """
        Average values (final incomes) of each solver across all given instances

        :param values: Can be 'income' or 'acc_reward'
        :param instances:
        """

        if self.data.get("baselines") is None or instances == 'random' or values != 'income':
            return dict()

        instances_ids = self.get_instances_ids(instances)
        baseline_objectives = dict()
        for solver, instance_objective in self.data["baselines"].items():
            if set(instances_ids).issubset(instance_objective.keys()):
                objectives = [obj for instance, obj in instance_objective.items() if instance in instances_ids]
                baseline_objectives[solver] = sum(objectives) / len(objectives)
        return baseline_objectives

    def add_random_instances(self, agent_id: str, path_evaluations: str):

        """


        """

        with np.load(path_evaluations) as data:
            timesteps = data["timesteps"].tolist()
            results = data["results"].tolist()

        acc_rewards = [
            [
                {
                    "acc_reward": reward,
                }
                for reward in result
            ]
            for result in results
        ]

        self.data[agent_id]["random_instances"] = {
            "values": acc_rewards,
            "timesteps": timesteps,
        }

    def remove_agent(self, agent_id: str):

        del self.data[agent_id]

    def __add__(self, other: Union["Training", Solution]):

        """
        Add the data of another Training object or a Solution (baseline)
        """

        if isinstance(other, Training):
            # Combine the data of two agents
            self_data = [
                self.data[agent_id]
                for agent_id in self.get_agent_ids()
            ]
            other_data = [
                other.data[agent_id]
                for agent_id in other.get_agent_ids()
            ]
            merged_data = [*self_data, *other_data]
            return Training.from_dict(merged_data)
        elif isinstance(other, Solution):
            solver = other.get_solver()
            instance_name = other.get_instance_name()
            if self.data.get("baselines") is None:
                self.data["baselines"] = dict()
            if self.data["baselines"].get(solver) is None:
                self.data["baselines"][solver] = dict()
            self.data["baselines"][solver][instance_name] = other.get_objective_function()
            return self
        else:
            raise TypeError(
                "The object added to the Training instance must be another Training instance or a Solution instance"
            )

    def __radd__(self, other):

        """
        Allow the use of the sum() function
        """

        if other == 0:
            return self
        else:
            return self.__add__(other)

    def plot_training_curves(self, ax: plt.Axes, values: list[str], instances: list[str] | str):

        """

        :param ax:
        :param values: List of values to plot (e.g., ['income', 'acc_reward'])
        :param instances: Can be 'fixed', 'random', or a list of specific fixed instances
        """

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Average income (€)")
        ax.set_title(f"Evaluation")

        twinax = ax.twinx()
        twinax.set_ylabel("Average accumulated rewards")

        colors = [plt.get_cmap('hsv')(color) for color in np.linspace(0, 1, len(self.get_agent_ids()), endpoint=False)]
        agent_colors = [
            (agent_id, color)
            for agent_id, color in zip(self.get_agent_ids(), colors)
        ]
        vals_linestyles_axes = [
            ("income", '-', ax),
            ("acc_reward", '--', twinax)
        ]

        # Plot learning curve of agents
        for agent_id, color in agent_colors:
            if not self.has_agent_instances(agent_id, instances):
                continue
            name = self.get_agent_name(agent_id)
            timesteps = self.get_timesteps(agent_id, instances)
            for val, linestyle, axes in vals_linestyles_axes:
                if val not in values or not self.has_agent_instances_values(agent_id, instances, val):
                    continue
                avg_values = self.get_avg_values(agent_id, instances, val)
                axes.plot(
                    timesteps, avg_values, color=color, linestyle=linestyle, label=f"Average {val} of {name}"
                )

        # Plot baselines as horizontal lines
        for val in values:
            baseline_values = self.get_baseline_values(instances, val)
            baseline_colors = [plt.get_cmap('hsv')(color) for color in np.linspace(0, 1, len(baseline_values), endpoint=False)]
            for (solver, value), color in zip(baseline_values.items(), baseline_colors):
                ax.axhline(y=value, color='gray', linestyle='-', label=solver)

        ax_lines, ax_labels = ax.get_legend_handles_labels()
        twinax_lines, twinax_labels = twinax.get_legend_handles_labels()
        ax.legend(ax_lines + twinax_lines, ax_labels + twinax_labels, loc=0)
