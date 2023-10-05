from flowing_basin.core import Instance, Solution, Experiment, Configuration
from flowing_basin.tools import RiverBasin
from cornflow_client.constants import SOLUTION_STATUS_FEASIBLE, STATUS_UNDEFINED
import numpy as np
import scipy
from matplotlib import pyplot as plt
from dataclasses import dataclass, asdict
import warnings
import json
import time
from datetime import datetime
import os
import typing
import my_pyswarms as ps
from my_pyswarms.utils.plotters import plot_cost_history
from my_pyswarms.utils.search import RandomSearch
# Download my_pyswarms with pip install git+https://github.com/RodrigoCastroF/my-pyswarms


@dataclass(kw_only=True)
class PSOConfiguration(Configuration):  # noqa

    num_particles: int
    max_iterations: int
    max_time: int

    # PySwarms optimizer options
    cognitive_coefficient: float
    social_coefficient: float
    inertia_weight: float

    # Particles represent flows, or flow variations? In the second case, are they capped?
    use_relvars: bool
    max_relvar: float = 0.5  # Used only when use_relvars=True

    # RiverBasin simulator options
    flow_smoothing: int = 0
    mode: str = "nonlinear"

    def __post_init__(self):
        valid_modes = {"linear", "nonlinear"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid value for 'mode': {self.mode}. Allowed values are {valid_modes}")


class PSO(Experiment):
    def __init__(
        self,
        instance: Instance,
        config: PSOConfiguration,
        paths_power_models: dict[str, str] = None,
        solution: Solution = None,
    ):

        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        # Information of how the solution was found
        self.solver_info = dict()

        self.config = config
        self.num_dimensions = (
            self.instance.get_num_dams() * self.instance.get_largest_impact_horizon()
        )
        self.objective_function_history = None

        if self.config.mode == "nonlinear" and paths_power_models is None:
            raise TypeError(
                "Parameter 'paths_power_models' is required when 'mode' is 'nonlinear', but it was not given."
            )

        if self.config.use_relvars:

            # Each element of a particle represents a relvar, bounded between -max_relvar and max_relvar
            max_bound = self.config.max_relvar * np.ones(self.num_dimensions)
            min_bound = -max_bound
            self.bounds = (min_bound, max_bound)

        else:

            # Each element of a particle represents a flow, bounded between 0 and max_flow_of_channel
            max_bound = np.tile(
                [
                    self.instance.get_max_flow_of_channel(dam_id)
                    for dam_id in self.instance.get_ids_of_dams()
                ],
                self.instance.get_largest_impact_horizon(),
            )
            min_bound = np.zeros(self.num_dimensions)
            self.bounds = (min_bound, max_bound)

        self.river_basin = RiverBasin(
            instance=self.instance,
            paths_power_models=paths_power_models,
            flow_smoothing=self.config.flow_smoothing,
            mode=self.config.mode,
        )

    def reshape_as_swarm(self, flows_or_relvars: np.ndarray) -> np.ndarray:

        """
        Reshape the given flows or relvars, so they can represent a swarm.

        :param flows_or_relvars:
            Array of shape num_time_steps x num_dams x num_particles with
            the flows or relvars assigned for the whole planning horizon
        :return:
            Array of shape num_particles x num_dimensions with
            candidate solutions (particles)
        """

        num_scenarios = flows_or_relvars.shape[-1]

        # Assert we are given an array of shape num_time_steps x num_dams x num_particles (as required by RiverBasin)
        assert flows_or_relvars.shape == (
            self.instance.get_largest_impact_horizon(),
            self.instance.get_num_dams(),
            num_scenarios,
        ), f"{flows_or_relvars.shape=} should actually be {(self.instance.get_largest_impact_horizon(), self.instance.get_num_dams(), num_scenarios)=}"

        # Reshape the array into num_dimensions x num_particles
        swarm_t = flows_or_relvars.reshape((self.num_dimensions, num_scenarios))

        # Transpose the array, turning its shape into num_particles x num_dimensions (as required by PySwarms)
        swarm = swarm_t.transpose()

        return swarm

    def reshape_as_flows_or_relvars(self, swarm: np.ndarray) -> np.ndarray:

        """
        Reshape the given swarm, so it can represent flows or relvars for the simulation model.

        :param swarm:
            Array of shape num_particles x num_dimensions with
            candidate solutions (particles)
        :return:
            Array of shape num_time_steps x num_dams x num_particles with
            the flows or relvars assigned in the whole planning horizon
        """

        num_particles = swarm.shape[0]

        # Assert  we are given an array of shape num_particles x num_dimensions (as required by PySwarms)
        assert swarm.shape == (
            num_particles,
            self.num_dimensions,
        ), f"{swarm.shape=} should actually be {(num_particles, self.num_dimensions)=}"

        # Transpose the array, turning its shape into num_dimensions x num_particles
        swarm_t = swarm.transpose()

        # Reshape the array into num_time_steps x num_dams x num_particles (as required by RiverBasin)
        flows_or_relvars = swarm_t.reshape(
            (
                self.instance.get_largest_impact_horizon(),
                self.instance.get_num_dams(),
                num_particles,
            )
        )
        return flows_or_relvars

    def check_env_updated(self):

        """
        Give a warning if environment (river basin) is not fully updated
        """

        # Make sure river basin has been fully updated; give a warning otherwise
        if self.river_basin.time != self.instance.get_largest_impact_horizon() - 1:
            warnings.warn(
                f"Calculated objective function values when the river basin is not fully updated: "
                f"{self.river_basin.time=}, when it should be {self.instance.get_largest_impact_horizon() - 1=}"
            )

    def objective_function_values_env(self) -> dict:

        """
        Objective function values of the environment (river basin) in its current state (presumably updated).

        :return:
            Dictionary formed by arrays of size num_particles with
            the objective function values reached by each particle
        """

        self.check_env_updated()

        obj_values = dict()

        obj_values.update(
            {
                "income": self.river_basin.get_acc_income(),
                "startups": self.river_basin.get_acc_num_startups(),
                "times_in_limit": self.river_basin.get_acc_num_times_in_limit()
            }
        )

        final_volumes = self.river_basin.get_final_volume_of_dams()
        for dam_id in self.instance.get_ids_of_dams():
            obj_values.update(
                {
                    # Calculate volume shortage and exceedance at the end of the decision horizon
                    f"{dam_id}_shortage": np.maximum(
                        0, self.config.volume_objectives[dam_id] - final_volumes[dam_id]
                    ),
                    f"{dam_id}_exceedance": np.maximum(
                        0, final_volumes[dam_id] - self.config.volume_objectives[dam_id]
                    ),
                }
            )

        return obj_values

    def objective_function_env(self) -> np.ndarray:

        """
        Objective function of the environment (river basin) in its current state (presumably updated).
        This is the objective function to minimize, as PySwarm's default behaviour is minimization.

        :return:
            Array of size num_particles with
            the objective function reached by each particle
        """

        self.check_env_updated()

        income = self.river_basin.get_acc_income()
        final_volumes = self.river_basin.get_final_volume_of_dams()
        volume_shortages = np.array(
            [
                np.maximum(
                    0, self.config.volume_objectives[dam_id] - final_volumes[dam_id]
                )
                for dam_id in self.instance.get_ids_of_dams()
            ]
        ).sum(axis=0)
        volume_exceedances = np.array(
            [
                np.maximum(
                    0, final_volumes[dam_id] - self.config.volume_objectives[dam_id]
                )
                for dam_id in self.instance.get_ids_of_dams()
            ]
        ).sum(axis=0)
        num_startups = self.river_basin.get_acc_num_startups()
        num_limit_zones = self.river_basin.get_acc_num_times_in_limit()

        penalty = (
            self.config.volume_shortage_penalty * volume_shortages
            + self.config.startups_penalty * num_startups
            + self.config.limit_zones_penalty * num_limit_zones
        )
        bonus = self.config.volume_exceedance_bonus * volume_exceedances

        return -income - bonus + penalty

    def calculate_objective_function(
        self, swarm: np.ndarray, is_relvars: bool
    ) -> np.ndarray:

        """
        Function that gives the objective function of the given swarm, as required by PySwarms.
        """

        self.river_basin.deep_update(
            self.reshape_as_flows_or_relvars(swarm), is_relvars=is_relvars, fast_mode=True
        )
        return self.objective_function_env()

    def optimize(
        self, options: dict[str, float], num_particles: int, max_iters: int, max_time: int = None
    ) -> tuple[float, np.ndarray]:

        """
        :param options: Dictionary with options given to the PySwarms optimizer
         - "c1": cognitive coefficient.
         - "c2": social coefficient
         - "w": inertia weight
        :param num_particles: Number of particles of the swarm
        :param max_iters: Max number of iterations with which to run the PSO
        :param max_time: Max number of seconds with which to run the PSO
        :return: Best objective function value found
        """

        optimizer = ps.single.GlobalBestPSO(
            n_particles=num_particles,
            dimensions=self.num_dimensions,
            options=options,
            bounds=self.bounds,
        )

        kwargs = {
            "is_relvars": self.config.use_relvars
        }  # Argument of `self.calculate_objective_function`
        cost, position = optimizer.optimize(
            self.calculate_objective_function, iters=max_iters, **kwargs, max_time=max_time
        )

        self.objective_function_history = optimizer.cost_history

        return cost, position

    def solve(self, options: dict = None) -> dict:

        """
        Fill the 'solution' attribute of the object, with the optimal solution found by the PSO algorithm.

        :param options: Unused argument, inherited from Experiment
        :return: A dictionary with status codes
        """

        start_time = time.perf_counter()
        ps_options = {
            "c1": self.config.cognitive_coefficient,
            "c2": self.config.social_coefficient,
            "w": self.config.inertia_weight
        }
        cost, optimal_particle = self.optimize(
            options=ps_options, num_particles=self.config.num_particles,
            max_iters=self.config.max_iterations, max_time=self.config.max_time
        )
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        self.river_basin.deep_update(
            self.reshape_as_flows_or_relvars(swarm=optimal_particle.reshape(1, -1)),
            is_relvars=self.config.use_relvars,
        )
        # print("DEBUG - optimal particle's original unclipped flows' history")
        # print(self.river_basin.history.to_string())
        optimal_flows = self.river_basin.all_past_clipped_flows
        self.solution = Solution.from_flows_array(
            optimal_flows, dam_ids=self.instance.get_ids_of_dams()
        )

        self.solver_info = dict(
            execution_time=execution_time,
            objective=-cost,
        )

        return dict(
            status_sol=SOLUTION_STATUS_FEASIBLE,
            status=STATUS_UNDEFINED,
        )

    def study_configuration(
        self,
        num_replications: int = 20,
        confidence: float = 0.95,
    ):

        """
        Get the mean and confidence interval of
        the income, objective function and final volume exceedances
        that the current PSO configuration gives.
        """

        # Dictionaries that will store the data and the results
        data = results = {
            "execution_times": [],
            "objectives": [],
            "incomes": [],
            **{
                "volume_exceedances_" + dam_id: []
                for dam_id in self.instance.get_ids_of_dams()
            },
        }

        # Execute `solve` multiple times to fill data
        for i in range(num_replications):

            self.solve()
            data["execution_times"].append(self.solver_info["execution_time"])
            data["objectives"].append(self.solver_info["objective"])

            self.river_basin.deep_update(self.solution.get_exiting_flows_array(), is_relvars=False)
            obj_values = self.objective_function_values_env()
            data["incomes"].append(obj_values["income"])
            for dam_id in self.instance.get_ids_of_dams():
                data[f"volume_exceedances_{dam_id}"].append(
                    obj_values[f"{dam_id}_exceedance"]
                    - obj_values[f"{dam_id}_shortage"]
                )
        print(data)

        # Get mean and confidence interval of each variable (assuming it follows a normal distribution)
        alfa = 1 - confidence
        for variable, values in data.items():
            a = np.array(values)
            n = a.size
            mean, std_error = np.mean(a), scipy.stats.sem(a)
            h = std_error * scipy.stats.t.ppf(1 - alfa / 2.0, n - 1)
            results[variable] = [mean, mean - h, mean + h]

        return results

    def search_best_options(
        self,
        options: dict[str, list[float]],
        num_particles: int = 200,
        num_iters_selection: int = 100,
        num_iters_each_test: int = 10,
    ) -> tuple[float, dict]:

        """
        Use PySwarm's RandomSearch method to find the best options.

        :param options: Dictionary with the combinations of options to try
        :param num_particles: Number of particles of the swarm in every execution
        :param num_iters_selection: Number of different combinations tried
        :param num_iters_each_test: Number of iterations with which every combination is tried
        :return: Best cost and the parameter configuration with which it was achieved.
        """

        g = RandomSearch(
            ps.single.GlobalBestPSO,
            n_particles=num_particles,
            dimensions=self.num_dimensions,
            bounds=self.bounds,
            objective_func=self.calculate_objective_function,
            options=options,
            iters=num_iters_each_test,
            n_selection_iters=num_iters_selection,
        )
        best_score, best_options = g.search()

        return best_score, best_options

    def get_objective(self, solution: Solution = None) -> float:

        """

        :return: The full objective function value of the current or given solution
        """

        if solution is None:
            assert (
                self.solution is not None
            ), "Cannot get objective if no solution has been given and `solve` has not been called yet."
            solution = self.solution

        self.river_basin.deep_update(solution.get_exiting_flows_array(), is_relvars=False)
        obj = self.objective_function_env()

        return - obj.item()

    def write_details_sol(self, details: typing.TextIO):

        """
        Write details about the current solution in the given TextIO object.
        """

        # Information about the instance
        format_datetime = "%Y-%m-%d %H:%M"
        start_datetime, end_datetime = self.instance.get_start_end_datetimes()
        start = start_datetime.strftime(format_datetime)
        end = end_datetime.strftime(format_datetime)
        solved = datetime.now().strftime(format_datetime)
        details.write(
            f"==== Solution details ====\n\n"
            f"Solved river basin instance between {start} and {end}.\n"
            f"The instance was solved at {solved} with the PSO algorithm.\n\n"
        )

        # Information about the solver
        details.write("---- Solver configuration ----\n\n")
        details.write("Configuration:\n")
        details.write(json.dumps(asdict(self.config), indent=2))
        details.write("\n\n")
        details.write("Solver information:\n")
        details.write(json.dumps(self.solver_info, indent=2))
        details.write("\n\n")

    def write_details_env(self, details: typing.TextIO):

        """
        Write details in the given TextIO object
        about the environment (river basin) in its current state (presumably updated).

        """

        self.check_env_updated()

        details.write("---- Objective function ----\n\n")
        details.write("Objective function values:\n")
        obj_values = self.objective_function_values_env()
        obj_values = {k: v.item() for k, v in obj_values.items()}
        details.write(json.dumps(obj_values, indent=2))
        details.write("\n\n")
        details.write("Objective function (to maximize):\n")
        obj = - self.objective_function_env()
        details.write(str(obj.item()))
        details.write("\n\n")

        details.write("---- River basin history ----\n\n")
        details.write(self.river_basin.history.to_string())
        details.write("\n\n")

    def plot_cost(self) -> plt.Axes:

        """
        Plot the value of the best solution throughout every iteration of the PSO.

        :return: Axes object with the cost plot
        """

        if self.objective_function_history is None:
            warnings.warn(
                "Cannot plot objective function history if `solve` has not been called yet."
            )
            return  # noqa

        ax = plot_cost_history(cost_history=self.objective_function_history)
        return ax

    def save_solution_info(self, path_parent: str, dir_name: str):

        """
        Save current solution, cost plot, details, and history plot
        """

        path_dir = os.path.join(path_parent, dir_name)
        os.mkdir(path_dir)
        print(f"Directory {path_dir} created")

        # Save current solution
        path_sol = os.path.join(path_dir, "solution.json")
        self.solution.to_json(path_sol)
        print(f"JSON file {path_sol} created")

        # Save cost plot
        path_cost_plot = os.path.join(path_dir, "cost_plot.png")
        self.plot_cost()
        plt.savefig(path_cost_plot)
        print(f"Figure {path_cost_plot} created")

        # Save details and history of river basin updated with the current solution
        self.river_basin.deep_update(self.solution.get_exiting_flows_array(), is_relvars=False)

        path_details = os.path.join(path_dir, "details.txt")
        with open(path_details, "w") as file:
            self.write_details_sol(file)
            self.write_details_env(file)
        print(f"Text file {path_details} created")

        path_history_plot = os.path.join(path_dir, "history_plot.png")
        self.river_basin.plot_history()
        plt.savefig(path_history_plot)
        print(f"Figure {path_history_plot} created")
