from flowing_basin.core import Instance, Solution, Experiment
from flowing_basin.tools import RiverBasin
from cornflow_client.constants import SOLUTION_STATUS_FEASIBLE, STATUS_UNDEFINED
import numpy as np
import scipy
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.search import RandomSearch
from matplotlib import pyplot as plt
from dataclasses import dataclass
import time
import os


@dataclass
class PSOConfiguration:

    # Objective final volumes, the penalty for unfulfilling them, and the bonus for exceeding them (in €/m3)
    volume_objectives: list[float]
    volume_shortage_penalty: float
    volume_exceedance_bonus: float

    # Other parameters
    flow_smoothing: int = 0
    max_relvar: float = 0.5  # Used only when use_relvars=True


class PSO(Experiment):
    def __init__(
        self,
        instance: Instance,
        paths_power_models: dict[str, str],
        config: PSOConfiguration,
        use_relvars: bool,
        solution: Solution = None,
    ):

        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        self.config = config
        self.use_relvars = use_relvars
        self.num_dimensions = (
            self.instance.get_num_dams() * self.instance.get_num_time_steps()
        )
        self.metadata = dict()
        self.metadata.update(
            {
                "k": self.config.flow_smoothing,
                "v": int(
                    self.config.volume_shortage_penalty != 0
                    and self.config.volume_exceedance_bonus != 0
                )
            }
        )
        self.objective_function_history = None

        if self.use_relvars:

            # Each element of a particle represents a relvar, bounded between -max_relvar and max_relvar
            max_bound = self.config.max_relvar * np.ones(self.num_dimensions)
            min_bound = -max_bound
            self.bounds = (min_bound, max_bound)
            self.metadata.update({"m": self.config.max_relvar})

        else:

            # Each element of a particle represents a flow, bounded between 0 and max_flow_of_channel
            max_bound = np.tile(
                [
                    self.instance.get_max_flow_of_channel(dam_id)
                    for dam_id in self.instance.get_ids_of_dams()
                ],
                self.instance.get_num_time_steps(),
            )
            min_bound = np.zeros(self.num_dimensions)
            self.bounds = (min_bound, max_bound)

        self.river_basin = RiverBasin(
            instance=self.instance, paths_power_models=paths_power_models, flow_smoothing=self.config.flow_smoothing
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
            self.instance.get_num_time_steps(),
            self.instance.get_num_dams(),
            num_scenarios,
        ), f"{flows_or_relvars.shape=} should actually be {(self.instance.get_num_time_steps(), self.instance.get_num_dams(), num_scenarios)=}"

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
                self.instance.get_num_time_steps(),
                self.instance.get_num_dams(),
                num_particles,
            )
        )
        return flows_or_relvars

    def deep_update_river_basin(self, swarm: np.ndarray, relvars: bool):

        """
        Update the river basin for the whole planning horizon with the solutions represented by the given swarm.

        :param swarm:
            Array of shape num_particles x num_dimensions with
            candidate solutions (particles)
        :param relvars: Whether the particles represent relvars or flows
        """

        num_particles = swarm.shape[0]
        self.river_basin.reset(num_scenarios=num_particles)
        flows_or_relvars = self.reshape_as_flows_or_relvars(swarm)

        if relvars:
            self.river_basin.deep_update_relvars(relvars=flows_or_relvars)
            return

        self.river_basin.deep_update_flows(flows=flows_or_relvars)
        return

    def turn_into_flows(self, swarm: np.ndarray, relvars: bool) -> np.ndarray:

        """
        Turn particle (solution) into the corresponding array of flows.

        :param swarm:
            Array of shape num_particles x num_dimensions with
            candidate solutions (particles)
        :param relvars: Whether the particles represent relvars or flows
        :return:
            Array of shape num_time_steps x num_dams x num_particles with
            the actual flows that exit the dams in the whole planning horizon
        """

        self.deep_update_river_basin(swarm=swarm, relvars=relvars)
        return self.river_basin.all_flows

    def objective_function_values(self, swarm: np.ndarray, relvars: bool) -> dict:

        """

        :param swarm:
            Array of shape num_particles x num_dimensions with
            candidate solutions (particles)
        :param relvars: Whether the particles represent relvars or flows
        :return:
            Dictionary formed by arrays of size num_particles with
            the objective function values reached by each particle
        """

        # Update river basin with current particles
        self.deep_update_river_basin(swarm=swarm, relvars=relvars)
        obj_values = dict()

        # Get accumulated income
        income = self.river_basin.accumulated_income
        obj_values.update({"income": income})

        # Get penalty or bonus because of volume shortage or exceedence
        for dam_index, dam in enumerate(self.river_basin.dams):
            obj_values.update(
                {
                    f"{dam.idx}_shortage": np.maximum(0, self.config.volume_objectives[dam_index] - dam.volume),
                    f"{dam.idx}_exceedance": np.maximum(0, dam.volume - self.config.volume_objectives[dam_index])
                }
            )

        return obj_values

    def objective_function(self, swarm: np.ndarray, relvars: bool) -> np.ndarray:

        """

        :param swarm:
            Array of shape num_particles x num_dimensions with
            candidate solutions (particles)
        :param relvars: Whether the particles represent relvars or flows
        :return:
            Array of size num_particles with
            the objective function reached by each particle
        """

        obj_values = self.objective_function_values(swarm=swarm, relvars=relvars)

        income = obj_values["income"]
        volume_shortage = np.array([obj_values[f"{dam_id}_shortage"] for dam_id in self.instance.get_ids_of_dams()]).sum(axis=0)
        volume_exceedance = np.array([obj_values[f"{dam_id}_exceedance"] for dam_id in self.instance.get_ids_of_dams()]).sum(axis=0)

        penalty = self.config.volume_shortage_penalty * volume_shortage
        bonus = self.config.volume_exceedance_bonus * volume_exceedance

        return -income - bonus + penalty

    def optimize(
        self, options: dict[str, float], num_particles: int, num_iters: int
    ) -> tuple[float, np.ndarray]:

        """
        :param options: Dictionary with options given to the PySwarms optimizer
         - "c1": cognitive coefficient.
         - "c2": social coefficient
         - "w": inertia weight
        :param num_particles: Number of particles of the swarm
        :param num_iters: Number of iterations with which to run the optimization algorithm
        :return: Best objective function value found
        """

        optimizer = ps.single.GlobalBestPSO(
            n_particles=num_particles,
            dimensions=self.num_dimensions,
            options=options,
            bounds=self.bounds,
        )

        kwargs = {"relvars": self.use_relvars}  # Argument of `self.objective_function`
        cost, position = optimizer.optimize(self.objective_function, iters=num_iters, **kwargs)

        self.objective_function_history = optimizer.cost_history

        return cost, position

    def solve(self, options: dict[str, float], num_particles: int = 200, num_iters: int = 100) -> dict:

        """
        Fill the 'solution' attribute of the object, with the optimal solution found by the PSO algorithm.

        :param options: Dictionary with options given to the PySwarms optimizer (see 'optimize' method for more info)
        :param num_particles: Number of particles of the swarm
        :param num_iters: Number of iterations with which to run the optimization algorithm
        :return: A dictionary with status codes and other information
        """

        self.metadata.update({"i": num_iters, "p": num_particles})
        self.metadata.update(options)

        start_time = time.time()
        cost, optimal_particle = self.optimize(options=options, num_particles=num_particles, num_iters=num_iters)
        end_time = time.time()
        execution_time = end_time - start_time

        optimal_flows = self.turn_into_flows(swarm=optimal_particle.reshape(1, -1), relvars=self.use_relvars)
        self.solution = Solution.from_flows(
            optimal_flows, dam_ids=self.instance.get_ids_of_dams()
        )

        return dict(
            status_sol=SOLUTION_STATUS_FEASIBLE,
            status=STATUS_UNDEFINED,
            execution_time=execution_time,
            objective=-cost,
        )

    def study_configuration(
        self,
        options: dict[str, float],
        num_iters_each_test: int = 100,
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

            info = self.solve(options=options, num_iters=num_iters_each_test)
            data["execution_times"].append(info["execution_time"])
            data["objectives"].append(info["objective"])

            obj_values = self.objective_function_values(swarm=self.reshape_as_swarm(self.solution.to_flows()), relvars=False)
            data["incomes"].append(obj_values["income"])
            for dam_id in self.instance.get_ids_of_dams():
                data[f"volume_exceedances_{dam_id}"].append(
                    obj_values[f"{dam_id}_exceedance"] - obj_values[f"{dam_id}_shortage"]
                )
        print(data)

        # Get mean and confidence interval of each variable (assuming it follows a normal distribution)
        alfa = 1 - confidence
        for variable, values in data.items():
            a = np.array(values)
            n = a.size
            mean, std_error = np.mean(a), scipy.stats.sem(a)
            h = std_error * scipy.stats.t.ppf(1 - alfa / 2., n - 1)
            results[variable] = [mean, mean - h, mean + h]

        return results

    def search_best_options(
        self,
        options: dict[str, list[float]],
        num_particles: int = 200,
        num_iters_selection: int = 100,
        num_iters_each_test: int = 10,
    ):

        """
        Use PySwarm's RandomSearch method to find the best options.
        """

        g = RandomSearch(
            ps.single.GlobalBestPSO,
            n_particles=num_particles,
            dimensions=self.num_dimensions,
            bounds=self.bounds,
            objective_func=self.objective_function,
            options=options,
            iters=num_iters_each_test,
            n_selection_iters=num_iters_selection,
        )
        best_score, best_options = g.search()

        return best_score, best_options

    def get_objective(self, solution: Solution = None) -> float:

        """
        :return: The full objective function value of the current solution
        """

        # Get objective function values of current/given solution
        if solution is None:
            assert self.solution is not None, (
                "Cannot plot solution history if no solution has been given and `solve` has not been called yet."
            )
            solution = self.solution

        swarm = self.reshape_as_swarm(solution.to_flows())
        obj = self.objective_function(swarm, relvars=False)

        return obj.item()

    def get_descriptive_filename(self, path: str) -> str:

        """
        Append useful information to the given path
        """

        filename, extension = os.path.splitext(path)
        filename += "_"
        filename += "_".join([f"{k}={round(v, 2)}" for k, v in self.metadata.items()])

        version = 0
        while os.path.exists(filename + f"_v{version}" + extension):
            version += 1

        return filename + f"_v{version}" + extension

    def save_solution(self, path: str):

        """
        Save the current solution using a descriptive filename
        """

        assert self.solution is not None, (
            "Cannot save solution if no solution has been given and `solve` has not been called yet."
        )

        self.solution.to_json(self.get_descriptive_filename(path))

    def plot_history(self) -> plt.Axes:

        """
        Plot the history of the river basin updated with the current solution
        """

        assert self.solution is not None, (
            "Cannot plot solution history if no solution has been given and `solve` has not been called yet."
        )

        self.river_basin.reset(num_scenarios=1)
        self.river_basin.deep_update_flows(self.solution.to_flows())
        axs = self.river_basin.plot_history()

        return axs

    def save_plot_history(self, path: str, show: bool = True):

        """
        Save the history plot using a descriptive filename
        """

        self.plot_history()
        plt.savefig(self.get_descriptive_filename(path))
        if show:
            plt.show()
        plt.close()

    def plot_objective_function_history(self) -> plt.Axes:

        """
        Plot the value of the best solution throughout every iteration of the PSO
        """

        assert self.objective_function_history is not None, (
            "Cannot plot objective function history if `solve` has not been called yet."
        )

        ax = plot_cost_history(cost_history=self.objective_function_history)
        return ax

    def save_plot_objective_function_history(self, path: str, show: bool = True):

        """
        Save the objective function's history plot using a descriptive filename
        """

        self.plot_objective_function_history()
        plt.savefig(self.get_descriptive_filename(path))
        if show:
            plt.show()
        plt.close()
