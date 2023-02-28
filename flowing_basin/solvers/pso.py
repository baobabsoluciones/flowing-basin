from flowing_basin.core import Instance, Solution, Experiment
from flowing_basin.tools import RiverBasin
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

        self.options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

        max_bound = 0.5 * np.ones(self.num_dimensions)
        min_bound = -max_bound
        self.bounds = (min_bound, max_bound)

        self.river_basin = RiverBasin(
            instance=self.instance,
            paths_power_models=paths_power_models
        )

    def swarm_to_relvars(self, swarm: np.ndarray) -> np.ndarray:

        """
        Turn swarm into array of relative flow variations
        """

        assert swarm.shape == (
            self.num_particles,
            self.num_dimensions,
        ), f"{swarm.shape=} should actually be {(self.num_dimensions, self.num_particles)=}"

        # Turn array of shape num_particles x num_dimensions (as required by PySwarms)
        # into array of shape num_dimensions x num_particles
        swarm_t = swarm.transpose()

        # Turn array of shape num_dimensions x num_particles
        # into array of shape num_time_steps x num_dams x num_particles (as required by RiverBasin)
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

        assert relvars.shape == (
            self.instance.get_num_time_steps(),
            self.instance.get_num_dams(),
            self.num_particles,
        ), f"{relvars.shape=} should actually be {(self.instance.get_num_time_steps(), self.instance.get_num_dams(), self.num_particles)=}"

        # Turn array of shape num_time_steps x num_dams x num_particles (as required by RiverBasin)
        # into array of shape num_dimensions x num_particles
        swarm_t = relvars.reshape((self.num_dimensions, self.num_particles))

        # Turn array of shape num_dimensions x num_particles
        # into array of shape num_particles x num_dimensions (as required by PySwarms)
        swarm = swarm_t.transpose()

        return swarm

    def particle_to_relvar(self, particle: np.ndarray) -> list[list[float]]:

        """
        Turn particle into the relative flow variations it represents
        """

        assert particle.shape == (
            self.num_dimensions,
        ), f"{particle.shape=} should actually be {(self.num_dimensions,)=}"

        # Turn array of shape num_dimensions
        # into array of shape num_time_steps x num_dams
        relvar = particle.reshape((self.instance.get_num_time_steps(), self.instance.get_num_dams()))

        # Turn array of shape num_time_steps x num_dams
        # into list
        relvar = relvar.tolist()

        return relvar

    def relvar_to_flows(self, relvar: list[list[float]]) -> np.ndarray:

        """
        Turn relative flow variations into flows
        """

        self.river_basin.reset(num_scenarios=1)
        _, equivalent_flows = self.river_basin.deep_update_relvars(relvar, return_equivalent_flows=True)

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

        return - accumulated_income

    def optimize(self):

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.num_particles,
            dimensions=self.num_dimensions,
            options=self.options,
            bounds=self.bounds,
        )

        cost, position = optimizer.optimize(self.objective_function, iters=100)

        return cost, position
