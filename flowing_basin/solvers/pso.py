from flowing_basin.core import Instance, Solution, Experiment
from flowing_basin.tools import RiverBasin
from cornflow_client.constants import SOLUTION_STATUS_FEASIBLE, STATUS_UNDEFINED
import numpy as np
import pyswarms as ps
import os
from dataclasses import dataclass


@dataclass
class PSOConfiguration:

    # Swarm size
    num_particles: int

    # Objective final volumes, the penalty for unfulfilling them and the bonus for exceeding them (in €/m3)
    volume_objectives: list[float]
    volume_shortage_penalty: float
    volume_exceedance_bonus: float

    # Conditions on the flow variations (PSOFlowVariations)
    # For PSOFlows, these are ignored
    keep_direction: int = 0
    max_relvar: float = 0.5


class PSO(Experiment):
    def __init__(
        self,
        instance: Instance,
        paths_power_models: dict[str, str],
        config: PSOConfiguration,
        solution: Solution = None,
    ):

        super().__init__(instance=instance, solution=solution)

        self.config = config
        self.num_particles = self.config.num_particles
        self.num_dimensions = (
            self.instance.get_num_dams() * self.instance.get_num_time_steps()
        )
        self.metadata = dict()

        # The upper and lower bounds for the position of the particles depend on what they represent,
        # so this attribute must be assigned in the child classes
        self.bounds = None

        self.river_basin = RiverBasin(
            instance=self.instance, paths_power_models=paths_power_models
        )

    def swarm_to_input(self, swarm: np.ndarray) -> np.ndarray:

        """
        Turn swarm into input appropriate for RiverBasin's deep update methods
        """

        # Assert  we are given an array of shape num_particles x num_dimensions (as required by PySwarms)
        assert swarm.shape == (
            self.num_particles,
            self.num_dimensions,
        ), f"{swarm.shape=} should actually be {(self.num_dimensions, self.num_particles)=}"

        # Transpose the array, turning its shape into num_dimensions x num_particles
        swarm_t = swarm.transpose()

        # Reshape the array into num_time_steps x num_dams x num_particles (as required by RiverBasin)
        input_env = swarm_t.reshape(
            (
                self.instance.get_num_time_steps(),
                self.instance.get_num_dams(),
                self.num_particles,
            )
        )
        return input_env

    def input_to_swarm(self, input_env: np.ndarray) -> np.ndarray:

        """
        Turn input for RiverBasin's deep update methods into swarm
        This method is not used inside this class, but may be useful to perform tests
        """

        # Assert we are given an array of shape num_time_steps x num_dams x num_particles (as required by RiverBasin)
        assert input_env.shape == (
            self.instance.get_num_time_steps(),
            self.instance.get_num_dams(),
            self.num_particles,
        ), f"{input_env.shape=} should actually be {(self.instance.get_num_time_steps(), self.instance.get_num_dams(), self.num_particles)=}"

        # Reshape the array into num_dimensions x num_particles
        swarm_t = input_env.reshape((self.num_dimensions, self.num_particles))

        # Transpose the array, turning its shape into num_particles x num_dimensions (as required by PySwarms)
        swarm = swarm_t.transpose()

        return swarm

    def particle_to_flows(self, particle: np.ndarray) -> list[list[float]]:

        """
        Turn particle (solution) into the corresponding nested list of flows
        """

        # This method depends on what the particles represent,
        # so it must be implemented in the child classes
        raise NotImplementedError()

    def get_income(self, swarm: np.ndarray) -> np.ndarray:

        """
        :param swarm: Array of shape num_particles x num_dimensions
        :return: Array of size num_particles with the total accumulated income obtained by each particle
        """

        # The way the income is calculated depends on what the particles represent,
        # so this method must be implemented in the child classes
        raise NotImplementedError()

    def objective_function(self, swarm: np.ndarray) -> np.ndarray:

        """
        :param swarm: Array of shape num_particles x num_dimensions
        :return: Array of size num_particles with the objective function reached by each particle
        """

        # Update river basin with current particles and get income
        income = self.get_income(swarm)

        # Calculate penalty or bonus because of volume shortage or exceedence
        volume_shortage = np.zeros(self.num_particles)
        volume_exceedance = np.zeros(self.num_particles)
        for dam_index, dam in enumerate(self.river_basin.dams):
            volume_shortage += np.maximum(0, self.config.volume_objectives[dam_index] - dam.volume)
            volume_exceedance += np.maximum(0, dam.volume - self.config.volume_objectives[dam_index])
        penalty = self.config.volume_shortage_penalty * volume_shortage
        bonus = self.config.volume_exceedance_bonus * volume_exceedance

        return - income - bonus + penalty

    def optimize(
        self, options: dict[str, float], num_iters: int = 100
    ) -> tuple[float, np.ndarray]:

        """
        :param options: Dictionary with options given to the PySwarms optimizer
         - "c1": cognitive coefficient.
         - "c2": social coefficient
         - "w": inertia weight
        :param num_iters: Number of iterations with which to run the optimization algorithm
        :return: Best objective function value found
        """

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.num_particles,
            dimensions=self.num_dimensions,
            options=options,
            bounds=self.bounds,
        )

        cost, position = optimizer.optimize(self.objective_function, iters=num_iters)

        return cost, position

    def solve(self, options: dict[str, float], num_iters: int = 100) -> dict:

        """
        Fill the 'solution' attribute of the object, with the optimal solution found by the PSO algorithm
        :param options: Dictionary with options given to the PySwarms optimizer (see 'optimize' method for more info)
        :param num_iters: Number of iterations with which to run the optimization algorithm
        :return: A dictionary with status codes
        """

        self.metadata.update({"i": num_iters, "p": self.num_particles})
        self.metadata.update(options)

        _, optimal_particle = self.optimize(options=options, num_iters=num_iters)
        optimal_flows = self.particle_to_flows(optimal_particle)

        self.solution = Solution.from_nestedlist(
            optimal_flows, dam_ids=self.instance.get_ids_of_dams()
        )

        return dict(status_sol=SOLUTION_STATUS_FEASIBLE, status=STATUS_UNDEFINED)

    def get_objective(self) -> float:

        """
        :return: The value of the current solution, given by the total income it provides
        """

        # Update river basin with a single scenario and get income
        self.river_basin.reset(num_scenarios=1)
        income = self.river_basin.deep_update_flows(self.solution.to_nestedlist())

        # Calculate penalty or bonus for volume shortage or excedence
        volume_shortage = 0
        volume_exceedance = 0
        for dam_index, dam in enumerate(self.river_basin.dams):
            volume_shortage += max(0, self.config.volume_objectives[dam_index] - dam.volume)
            volume_exceedance += max(0, dam.volume - self.config.volume_objectives[dam_index])
        penalty = self.config.volume_shortage_penalty * volume_shortage
        bonus = self.config.volume_exceedance_bonus * volume_exceedance

        return - income - bonus + penalty

    def get_descriptive_filename(self, path: str) -> str:

        """
        Append useful information to the given path
        """

        filename, extension = os.path.splitext(path)
        filename += "_"
        filename += "_".join([f"{k}={v}" for k, v in self.metadata.items()])

        version = 0
        while os.path.exists(filename + f"_v{version}" + extension):
            version += 1

        return filename + f"_v{version}" + extension

    def save_solution(self, path: str):

        """
        Save the current solution using a descriptive filename
        """

        self.solution.to_json(self.get_descriptive_filename(path))

    def save_history_plot(self, path: str):

        """
        Save the history plot of the river basin updated with the current solution
        using a descriptive filename
        """

        self.river_basin.reset(num_scenarios=1)
        self.river_basin.deep_update_flows(self.solution.to_nestedlist())
        self.river_basin.plot_history(path=self.get_descriptive_filename(path))


class PSOFlowVariations(PSO):
    def __init__(
        self,
        instance: Instance,
        paths_power_models: dict[str, str],
        config: PSOConfiguration,
        solution: Solution = None,
    ):

        super().__init__(
            instance=instance,
            paths_power_models=paths_power_models,
            config=config,
            solution=solution,
        )

        self.keep_direction = self.config.keep_direction

        max_bound = self.config.max_relvar * np.ones(self.num_dimensions)
        min_bound = -max_bound
        self.bounds = (min_bound, max_bound)

        self.metadata.update({"k": self.keep_direction, "m": self.config.max_relvar})

    def particle_to_relvar(self, particle: np.ndarray) -> list[list[float]]:

        """
        Turn particle (solution) into the relative flow variations it represents
        """

        # Assert we are given an array of shape num_dimensions (which is how PySwarms represents particles)
        assert particle.shape == (
            self.num_dimensions,
        ), f"{particle.shape=} should actually be {(self.num_dimensions,)=}"

        # Reshape the array into num_time_steps x num_dams
        relvar = particle.reshape(
            (self.instance.get_num_time_steps(), self.instance.get_num_dams())
        )

        # Turn the array into a list (as required by RiverBasin for a single scenario)
        relvar = relvar.tolist()

        return relvar

    def relvar_to_flows(self, relvar: list[list[float]]) -> list[list[float]]:

        """
        Turn relative flow variations into flows
        """

        self.river_basin.reset(num_scenarios=1)
        _, equivalent_flows = self.river_basin.deep_update_relvars(
            relvar, keep_direction=self.keep_direction, return_equivalent_flows=True
        )

        return equivalent_flows

    def particle_to_flows(self, particle: np.ndarray) -> list[list[float]]:

        """
        Turn particle (solution) into the corresponding nested list of flows
        """

        return self.relvar_to_flows(self.particle_to_relvar(particle))

    def get_income(self, swarm: np.ndarray) -> np.ndarray:

        """
        :param swarm: Array of shape num_particles x num_dimensions
        :return: Array of size num_particles with the total accumulated income obtained by each particle
        """

        self.river_basin.reset(num_scenarios=self.num_particles)
        relvars = self.swarm_to_input(swarm)
        accumulated_income = self.river_basin.deep_update_relvars(relvars, keep_direction=self.keep_direction)

        return accumulated_income


class PSOFlows(PSO):
    def __init__(
        self,
        instance: Instance,
        paths_power_models: dict[str, str],
        config: PSOConfiguration,
        solution: Solution = None,
    ):

        super().__init__(
            instance=instance,
            paths_power_models=paths_power_models,
            config=config,
            solution=solution,
        )

        max_bound = np.tile(
            [
                self.instance.get_max_flow_of_channel(dam_id)
                for dam_id in self.instance.get_ids_of_dams()
            ],
            self.instance.get_num_time_steps(),
        )
        min_bound = np.zeros(self.num_dimensions)
        self.bounds = (min_bound, max_bound)

    def particle_to_flows(self, particle: np.ndarray) -> list[list[float]]:

        """
        Turn particle (solution) into the corresponding nested list of flows
        """

        # Assert we are given an array of shape num_dimensions (which is how PySwarms represents particles)
        assert particle.shape == (
            self.num_dimensions,
        ), f"{particle.shape=} should actually be {(self.num_dimensions,)=}"

        # Reshape the array into num_time_steps x num_dams
        flows = particle.reshape(
            (self.instance.get_num_time_steps(), self.instance.get_num_dams())
        )

        # Turn the array into a nested list (as required by RiverBasin for a single scenario)
        flows = flows.tolist()

        return flows

    def get_income(self, swarm: np.ndarray) -> np.ndarray:

        """
        :param swarm: Array of shape num_particles x num_dimensions
        :return: Array of size num_particles with the total accumulated income obtained by each particle
        """

        self.river_basin.reset(num_scenarios=self.num_particles)
        flows = self.swarm_to_input(swarm)
        accumulated_income = self.river_basin.deep_update_flows(flows)

        return accumulated_income
