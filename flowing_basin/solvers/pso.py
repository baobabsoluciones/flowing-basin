from flowing_basin.core import Instance, Solution, Experiment, Configuration
from flowing_basin.tools import RiverBasin
from cornflow_client.constants import SOLUTION_STATUS_FEASIBLE, STATUS_UNDEFINED
import numpy as np
from dataclasses import dataclass, asdict
import warnings
import time
from datetime import datetime
import pyswarms.backend as P
from pyswarms.backend.topology import Star


@dataclass(kw_only=True)
class PSOConfiguration(Configuration):  # noqa

    num_particles: int

    # PySwarms optimizer options
    cognitive_coefficient: float
    social_coefficient: float
    inertia_weight: float

    # Particles represent flows, or flow variations? In the second case, are they capped?
    use_relvars: bool
    max_relvar: float = 0.5  # Used only when use_relvars=True

    # Max iterations OR max time
    max_iterations: int = None
    max_time: float = None

    # RiverBasin simulator options
    flow_smoothing: int = 0
    mode: str = "nonlinear"

    def __post_init__(self):

        # Assert given mode is valid
        valid_modes = {"linear", "nonlinear"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid value for 'mode': {self.mode}. Allowed values are {valid_modes}")

        # Turn not given max iterations or max time into infinity
        if self.max_iterations is None and self.max_time is None:
            raise ValueError("You need to specify a maximum number of iterations or a time limit (or both).")
        if self.max_iterations is None:
            self.max_iterations = float('inf')  # noqa
        if self.max_time is None:
            self.max_time = float('inf')


class PSO(Experiment):
    def __init__(
        self,
        instance: Instance,
        config: PSOConfiguration,
        paths_power_models: dict[str, str] = None,
        solution: Solution = None,
        verbose: bool = True
    ):

        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        self.verbose = verbose
        self.config = config
        self.num_dimensions = (
            self.instance.get_num_dams() * self.instance.get_largest_impact_horizon()
        )

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
        ), (
            f"{flows_or_relvars.shape=} should actually be "
            f"{(self.instance.get_largest_impact_horizon(), self.instance.get_num_dams(), num_scenarios)=}"
        )

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

    def env_objective_function_details(self, dam_id: str) -> dict[str, np.ndarray]:

        """
        Objective function details for the given dam
        from the environment (river basin) in its current state (presumably updated).

        :return:
            Dictionary formed by arrays of size num_particles with
            the objective function values reached by each particle
        """

        self.check_env_updated()

        dam_index = self.instance.get_order_of_dam(dam_id) - 1
        dam = self.river_basin.dams[dam_index]

        income = dam.channel.power_group.acc_income
        startups = dam.channel.power_group.acc_num_startups
        limit_zones = dam.channel.power_group.acc_num_times_in_limit
        vol_shortage = np.maximum(0, self.config.volume_objectives[dam_id] - dam.final_volume)
        vol_exceedance = np.maximum(0, dam.final_volume - self.config.volume_objectives[dam_id])
        total_income = (
            income
            - startups * self.config.startups_penalty
            - limit_zones * self.config.limit_zones_penalty
            - vol_shortage * self.config.volume_shortage_penalty
            + vol_exceedance * self.config.volume_exceedance_bonus
        )

        obj_values = dict(
            total_income_eur=total_income,
            income_from_energy_eur=income,
            startups=startups,
            limit_zones=limit_zones,
            volume_shortage_m3=vol_shortage,
            volume_exceedance_m3=vol_exceedance
        )

        return obj_values

    def env_objective_function(self) -> np.ndarray:

        """
        Objective function of the environment (river basin) in its current state (presumably updated).
        This is the objective function to maximize.

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

        return income + bonus - penalty

    def calculate_cost(
        self, swarm: np.ndarray, is_relvars: bool
    ) -> np.ndarray:

        """
        Function that gives the cost of the given swarm, as required by PySwarms.
        This is the objective function with a NEGATIVE sign, as PySwarm's default behaviour is minimization.
        """

        self.river_basin.deep_update(
            self.reshape_as_flows_or_relvars(swarm), is_relvars=is_relvars, fast_mode=True
        )
        return - self.env_objective_function()

    def solve(self, options: dict = None) -> dict:

        """
        Fill the 'solution' attribute of the object, with the optimal solution found by the PSO algorithm.

        :param options: Unused argument, inherited from Experiment
        :return: A dictionary with status codes
        """

        # Create a Star topology (to implement Global Best PSO)
        topology = Star()

        # Create a swarm
        swarm_options = {
            "c1": self.config.cognitive_coefficient,
            "c2": self.config.social_coefficient,
            "w": self.config.inertia_weight
        }
        swarm = P.create_swarm(
            n_particles=self.config.num_particles,
            dimensions=self.num_dimensions,
            options=swarm_options,
            bounds=self.bounds,
            init_pos=None
        )

        start_time = time.perf_counter()
        current_time = 0.
        num_iters = 0
        obj_fun_history = []
        if self.verbose:
            print(f"{'Iteration':<15}{'Time (s)':<15}{'Objective (€)':<15}")

        # Optimization loop
        while current_time < self.config.max_time and num_iters < self.config.max_iterations:

            # Update personal best
            swarm.current_cost = self.calculate_cost(swarm.position, is_relvars=self.config.use_relvars)
            if num_iters == 0:
                swarm.pbest_cost = self.calculate_cost(swarm.pbest_pos, is_relvars=self.config.use_relvars)
            swarm.pbest_pos, swarm.pbest_cost = P.compute_pbest(swarm)

            # Update global best
            if np.min(swarm.pbest_cost) < swarm.best_cost:
                swarm.best_pos, swarm.best_cost = topology.compute_gbest(swarm)
            obj_fun_history.append((current_time, - swarm.best_cost))
            if self.verbose:
                print(f"{num_iters:<15}{current_time:<15.2f}{-swarm.best_cost:<15.2f}")

            # Update position and velocity matrices
            swarm.velocity = topology.compute_velocity(swarm)
            swarm.position = topology.compute_position(swarm)

            # Next iteration
            current_time = time.perf_counter() - start_time
            num_iters += 1

        if self.verbose:
            print(f"Optimization finished.\nBest position is {swarm.best_pos}\nBest cost is {swarm.best_cost}")

        # Get clipped optimal flows, and the corresponding volumes and powers of each dam
        self.river_basin.deep_update(
            self.reshape_as_flows_or_relvars(swarm=swarm.best_pos.reshape(1, -1)),
            is_relvars=self.config.use_relvars,
        )
        optimal_flows = self.river_basin.all_past_clipped_flows  # Array of shape num_time_steps x num_dams x 1
        optimal_flows = np.transpose(optimal_flows)[0]  # Array of shape num_dams x num_time_steps
        volumes = dict()
        powers = dict()
        for dam_id in self.instance.get_ids_of_dams():
            volumes[dam_id] = self.river_basin.all_past_volumes[dam_id]  # Array of shape num_time_steps x 1
            volumes[dam_id] = np.transpose(volumes[dam_id])[0]  # Array of shape num_time_steps
            powers[dam_id] = self.river_basin.all_past_powers[dam_id]  # Array of shape num_time_steps x 1
            powers[dam_id] = np.transpose(powers[dam_id])[0]  # Array of shape num_time_steps

        # Get objective function history
        time_stamps, obj_fun_values = zip(*obj_fun_history)
        time_stamps = list(time_stamps)
        obj_fun_values = list(obj_fun_values)

        # Get datetimes
        format_datetime = "%Y-%m-%d %H:%M"
        start_datetime, end_datetime = self.instance.get_start_end_datetimes()
        start_datetime = start_datetime.strftime(format_datetime)
        end_datetime = end_datetime.strftime(format_datetime)
        solution_datetime = datetime.now().strftime(format_datetime)

        # Store solution in attribute
        self.solution = Solution.from_dict(
            dict(
                instance_datetimes=dict(
                    start=start_datetime,
                    end_decisions=end_datetime
                ),
                solution_datetime=solution_datetime,
                solver="PSO",
                configuration=asdict(self.config),
                objective_function=self.env_objective_function().item(),
                objective_history=dict(
                    objective_values_eur=obj_fun_values,
                    time_stamps_s=time_stamps,
                ),
                dams=[
                    dict(
                        id=dam_id,
                        flows=optimal_flows[self.instance.get_order_of_dam(dam_id) - 1].tolist(),
                        power=powers[dam_id].tolist(),
                        volume=volumes[dam_id].tolist(),
                        objective_function_details={
                            detail_key: detail_value.item()
                            for detail_key, detail_value in self.env_objective_function_details(dam_id).items()
                        }
                    )
                    for dam_id in self.instance.get_ids_of_dams()
                ],
                price=self.instance.get_all_prices(),
            )
        )

        return dict(
            status_sol=SOLUTION_STATUS_FEASIBLE,
            status=STATUS_UNDEFINED,
        )

    def get_objective(self, solution: Solution = None) -> float:

        """

        :return: The full objective function value of the current or given solution
        """

        if solution is None:
            if self.solution is None:
                raise ValueError(
                    "Cannot get objective if no solution has been given and `solve` has not been called yet."
                )
            solution = self.solution

        self.river_basin.deep_update(solution.get_exiting_flows_array(), is_relvars=False)
        obj = self.env_objective_function()

        return obj.item()
