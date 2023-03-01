from flowing_basin.core import Instance, Solution, Experiment
from flowing_basin.tools import RiverBasin
from cornflow_client.constants import SOLUTION_STATUS_FEASIBLE, STATUS_UNDEFINED
import numpy as np
import pyswarms as ps


class PSO(Experiment):
    def __init__(
        self,
        instance: Instance,
        paths_power_models: dict[str, str],
        num_particles: int,
        solution: Solution = None,
    ):

        super().__init__(instance=instance, solution=solution)

        self.num_particles = num_particles
        self.num_dimensions = (
            self.instance.get_num_dams() * self.instance.get_num_time_steps()
        )

        max_bound = 0.5 * np.ones(self.num_dimensions)
        min_bound = -max_bound
        self.bounds = (min_bound, max_bound)

        self.river_basin = RiverBasin(
            instance=self.instance, paths_power_models=paths_power_models
        )

    def swarm_to_relvars(self, swarm: np.ndarray) -> np.ndarray:

        """
        Turn swarm into array of relative flow variations
        """

        # Assert  we are given an array of shape num_particles x num_dimensions (as required by PySwarms)
        assert swarm.shape == (
            self.num_particles,
            self.num_dimensions,
        ), f"{swarm.shape=} should actually be {(self.num_dimensions, self.num_particles)=}"

        # Transpose the array, turning its shape into num_dimensions x num_particles
        swarm_t = swarm.transpose()

        # Reshape the array into num_time_steps x num_dams x num_particles (as required by RiverBasin)
        relvars = swarm_t.reshape(
            (
                self.instance.get_num_time_steps(),
                self.instance.get_num_dams(),
                self.num_particles,
            )
        )
        return relvars

    def relvars_to_swarm(self, relvars: np.ndarray) -> np.ndarray:

        """
        Turn array of relative flow variations into swarm
        """

        # Assert we are given an array of shape num_time_steps x num_dams x num_particles (as required by RiverBasin)
        assert relvars.shape == (
            self.instance.get_num_time_steps(),
            self.instance.get_num_dams(),
            self.num_particles,
        ), f"{relvars.shape=} should actually be {(self.instance.get_num_time_steps(), self.instance.get_num_dams(), self.num_particles)=}"

        # Reshape the array into num_dimensions x num_particles
        swarm_t = relvars.reshape((self.num_dimensions, self.num_particles))

        # Transpose the array, turning its shape into num_particles x num_dimensions (as required by PySwarms)
        swarm = swarm_t.transpose()

        return swarm

    def particle_to_relvar(self, particle: np.ndarray) -> list[list[float]]:

        """
        Turn particle into the relative flow variations it represents
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
            relvar, return_equivalent_flows=True
        )

        return equivalent_flows

    def objective_function(self, swarm: np.ndarray) -> np.ndarray:

        """
        :param swarm: Array of shape num_particles x num_dimensions
        :return: Array of size num_particles with the total accumulated income obtained by each particle,
        in negative since the objective is minimization
        """

        self.river_basin.reset(num_scenarios=self.num_particles)
        relvars = self.swarm_to_relvars(swarm)
        accumulated_income = self.river_basin.deep_update_relvars(relvars)

        return -accumulated_income

    def optimize(self, options: dict[str, float], num_iters: int = 100) -> tuple[float, np.ndarray]:

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
        print("optimal solution (inside):", self.relvar_to_flows(self.particle_to_relvar(position)))
        print("optimal income (inside):", - cost)

        return cost, position

    def solve(self, options: dict[str, float], num_iters: int = 100) -> dict:

        """
        Fill the 'solution' attribute of the object, with the optimal solution found by the PSO algorithm
        :param options: Dictionary with options given to the PySwarms optimizer (see 'optimize' above for more info)
        :param num_iters: Number of iterations with which to run the optimization algorithm
        :return: A dictionary with status codes
        """

        _, optimal_particle = self.optimize(options=options, num_iters=num_iters)
        optimal_relvar = self.particle_to_relvar(optimal_particle)
        optimal_flows = self.relvar_to_flows(optimal_relvar)

        self.solution = Solution.from_nestedlist(optimal_flows, dam_ids=self.instance.get_ids_of_dams())

        return dict(status_sol=SOLUTION_STATUS_FEASIBLE, status=STATUS_UNDEFINED)

    def get_objective(self) -> float:

        """
        :return: The value of the current solution, given by the total income it provides
        """

        self.river_basin.reset(num_scenarios=1)
        total_income = self.river_basin.deep_update_flows(self.solution.to_nestedlist())

        return total_income
