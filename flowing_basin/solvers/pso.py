from flowing_basin.core import Instance, Solution, Experiment, Configuration
from flowing_basin.tools import RiverBasin
from cornflow_client.constants import SOLUTION_STATUS_FEASIBLE, STATUS_UNDEFINED
import numpy as np
from dataclasses import dataclass
from bisect import bisect_left
import warnings
import time
import pyswarms.backend as P
from pyswarms.backend.handlers import BoundaryHandler
from pyswarms.backend.topology import Star, Ring, VonNeumann, Pyramid, Random


@dataclass(kw_only=True)
class PSOConfiguration(Configuration):  # noqa

    num_particles: int

    # Continuous PySwarms optimizer options
    cognitive_coefficient: float
    social_coefficient: float
    inertia_weight: float

    # Particles represent flows, or relative flow variations?
    use_relvars: bool

    # Discrete PySwarms optimizer options
    bounds_handling: str = "periodic"
    topology: str = "star"
    topology_num_neighbors: int = None
    topology_minkowski_p_norm: int = None
    # topology_range: int = None

    # Max iterations OR max time
    max_iterations: int = None
    max_time: float = None

    # Early stopping / convergence criteria
    # If the objective function has not improved
    # more than `convergence_tolerance * 100`% in the last  `convergence_time` seconds,
    # assume the PSO has converged and stop the execution
    convergence_tolarance: float = 0.005
    convergence_time: float = 150

    # RiverBasin simulator options
    mode: str = "nonlinear"

    def get_topology_kwargs(self):

        topology_kwargs = {
            "star": {},
            "ring": {"p": self.topology_minkowski_p_norm, "k": self.topology_num_neighbors},
            # "von_neumann": {"p": self.topology_minkowski_p_norm, "r": self.topology_range},
            "pyramid": {},
            "random": {"k": self.topology_num_neighbors}
        }
        return topology_kwargs[self.topology]

    def __post_init__(self):

        super(PSOConfiguration, self).__post_init__()

        if self.max_relvar < 1. and not self.use_relvars:
            raise ValueError(f"The maximum relative flow variation is {self.max_relvar} < 1., but {self.use_relvars=}.")

        # Assert given string values are valid
        # NOTE: the "von_neumann" topology is excluded from the list of valid topologies because,
        # with the >200 dimensions of our problem, the generated number of neighbours as a Delannoy number
        # is exceedingly high, even with a range of 1
        valid_attr_values = dict(
            mode={"linear", "nonlinear"},
            bounds_handling={"periodic", "nearest", "intermediate", "shrink", "reflective", "random"},
            topology={"star", "ring", "pyramid", "random"},
            topology_minkowski_p_norm={1, 2, None}
        )
        for attr_name, valid_values in valid_attr_values.items():
            if getattr(self, attr_name) not in valid_values:
                raise ValueError(
                    f"Invalid value for '{attr_name}': {getattr(self, attr_name)}. "
                    f"Allowed values are {valid_values}."
                )

        if self.topology_num_neighbors is not None and self.topology_num_neighbors > self.num_particles - 1:
            raise ValueError(
                f"The topology number of neighbours, {self.topology_num_neighbors}, "
                f"must be lower than the number of particles, {self.num_particles}."
            )

        # Assert the required topology kwargs are given
        for key, value in self.get_topology_kwargs().items():
            if value is None:
                raise ValueError(f"Topology {self.topology} requires a value for {key}.")

        # Assert a max iterations or max time is given
        if self.max_iterations is None and self.max_time is None:
            raise ValueError("You need to specify a maximum number of iterations or a time limit (or both).")


class PSO(Experiment):
    def __init__(
        self,
        instance: Instance,
        config: PSOConfiguration,
        paths_power_models: dict[str, str] = None,
        solution: Solution = None,
        verbose: bool = True
    ):

        super().__init__(instance=instance, config=config, solution=solution)
        self.config = config  # This is unnecessary but avoids error highlighting
        if solution is None:
            self.solution = None

        self.verbose = verbose
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
            max_relvar=self.config.max_relvar,
            mode=self.config.mode,
            do_history_updates=False
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
        return self.river_basin.get_objective_function_details(dam_id, config=self.config)

    def env_objective_function(self) -> np.ndarray:

        """
        Objective function of the environment (river basin) in its current state (presumably updated).
        This is the objective function to maximize.

        :return:
            Array of size num_particles with
            the objective function reached by each particle
        """

        self.check_env_updated()
        return self.river_basin.get_objective_function_value(config=self.config)

    def calculate_cost(
        self, swarm: np.ndarray, is_relvars: bool
    ) -> np.ndarray:

        """
        Function that gives the cost of the given swarm, as required by PySwarms.
        This is the objective function with a NEGATIVE sign, as PySwarm's default behaviour is minimization.
        """

        self.river_basin.deep_update(
            self.reshape_as_flows_or_relvars(swarm), is_relvars=is_relvars
        )
        return - self.env_objective_function()

    def solve(
        self, initial_solutions: np.ndarray = None, time_offset: float = 0., solver: str = "PSO", options: dict = None,
        epsilon: float = 1e-4
    ) -> dict:

        """
        Fill the 'solution' attribute of the object, with the optimal solution found by the PSO algorithm.

        :param initial_solutions: Array of shape num_time_steps x num_dams x num_particles with
            the initial solutions (flows or relvars)
        :param time_offset: Starting time of the algorithm in seconds
            (used if there is any preprocessing before PSO, e.g. RBO), defaults to 0
        :param solver: Solver to indicate in the stored solution (e.g. PSO or PSO-RBO), defaults to PSO
        :param options: Unused argument, inherited from Experiment
        :param epsilon: Tolerance for numerical imprecision when checking the results
        :return: A dictionary with status codes
        """

        # Choose a topology
        topology = {
            "star": Star(), "ring": Ring(), "von_neumann": VonNeumann(), "pyramid": Pyramid(), "random": Random()
        }[self.config.topology]

        # Create a swarm
        swarm_options = {
            "c1": self.config.cognitive_coefficient,
            "c2": self.config.social_coefficient,
            "w": self.config.inertia_weight
        }
        if initial_solutions is not None:
            initial_solutions = self.reshape_as_swarm(initial_solutions)
        swarm = P.create_swarm(
            n_particles=self.config.num_particles,
            dimensions=self.num_dimensions,
            options=swarm_options,
            bounds=self.bounds,
            init_pos=initial_solutions
        )

        # Choose a handler for out-of-bounds particles
        boundary_handler = BoundaryHandler(strategy=self.config.bounds_handling)
        boundary_handler.memory = swarm.position

        start_time = time.perf_counter()
        current_time = time_offset
        num_iters = 0
        obj_history_times = []
        obj_history_values = []
        if self.verbose:
            print(f"{'Iteration':<15}{'Time (s)':<15}{'Objective (€)':<15}")

        # Set maximum number of iterations or time limit
        max_iters = self.config.max_iterations
        if max_iters is None:
            max_iters = float('inf')
        max_time = self.config.max_time
        if max_time is None:
            max_time = float('inf')

        # Optimization loop
        # Based on https://github.com/ljvmiranda921/pyswarms/blob/master/pyswarms/single/general_optimizer.py
        # and https://github.com/ljvmiranda921/pyswarms/blob/master/pyswarms/base/base_single.py
        swarm.pbest_cost = np.full(self.config.num_particles, np.inf)
        while True:

            # Update personal best
            swarm.current_cost = self.calculate_cost(swarm.position, is_relvars=self.config.use_relvars)
            swarm.pbest_pos, swarm.pbest_cost = P.compute_pbest(swarm)

            # Update global best
            swarm.best_pos, swarm.best_cost = topology.compute_gbest(swarm, **self.config.get_topology_kwargs())
            current_value = - swarm.best_cost
            obj_history_times.append(current_time)
            obj_history_values.append(current_value)
            if self.verbose:
                print(f"{num_iters:<15}{current_time:<15.2f}{current_value:<15.2f}")

            # Early stopping / convergence criteria
            if current_time - time_offset > self.config.convergence_time:

                # Identify the objective function value from `convergence_time` seconds ago using binary search
                target_time = current_time - self.config.convergence_time
                target_time_index = bisect_left(a=obj_history_times, x=target_time)
                target_value = obj_history_values[target_time_index]

                # Check if the value has improved less than `convergence_tolerance * 100`%
                relative_improvement = (current_value - target_value) / target_value if target_value > 0 else float('inf')
                if relative_improvement < self.config.convergence_tolarance:
                    print(
                        f"The value {self.config.convergence_time}s ago was {target_value}; now it is {current_value}. "
                        f"Since the relative improvement is "
                        f"{relative_improvement*100:.2f}% < {self.config.convergence_tolarance*100:.2f}%, "
                        f"the execution will be stopped now."
                    )
                    break

            # Update position and velocity matrices
            swarm.velocity = topology.compute_velocity(swarm, bounds=self.bounds)
            swarm.position = topology.compute_position(swarm, bounds=self.bounds, bh=boundary_handler)

            # Next iteration
            current_time = time.perf_counter() - start_time + time_offset
            num_iters += 1
            if current_time > max_time or num_iters > max_iters:
                break

        # Obtain the final best_cost and the final best_position
        final_best_cost = swarm.best_cost.copy()
        final_best_pos = swarm.pbest_pos[swarm.pbest_cost.argmin()].copy()

        if self.verbose:
            print(f"Optimization finished.\nBest position is {final_best_pos}\nBest cost is {final_best_cost}")

        # Assert optimal party is between bounds
        assert (final_best_pos >= self.bounds[0] - epsilon).all() and (final_best_pos <= self.bounds[1] + epsilon).all()

        # Execute simulator with optimal particle
        self.river_basin.deep_update(
            self.reshape_as_flows_or_relvars(swarm=final_best_pos.reshape(1, -1)),
            is_relvars=self.config.use_relvars,
        )

        # Actual (clipped) exiting flows that result from the assigned flows
        clipped_flows = self.river_basin.all_past_clipped_flows  # Array of shape num_time_steps x num_dams x 1
        clipped_flows = np.transpose(clipped_flows)[0]  # Array of shape num_dams x num_time_steps

        # Volumes and powers of each dam
        volumes = dict()
        powers = dict()
        for dam_id in self.instance.get_ids_of_dams():
            volumes[dam_id] = self.river_basin.all_past_volumes[dam_id]  # Array of shape num_time_steps x 1
            volumes[dam_id] = np.transpose(volumes[dam_id])[0]  # Array of shape num_time_steps
            powers[dam_id] = self.river_basin.all_past_powers[dam_id]  # Array of shape num_time_steps x 1
            powers[dam_id] = np.transpose(powers[dam_id])[0]  # Array of shape num_time_steps

        # Get datetimes
        start_datetime, end_datetime, _, _, _, solution_datetime = self.get_instance_solution_datetimes()

        # Store solution in attribute
        self.solution = Solution.from_dict(
            dict(
                instance_datetimes=dict(
                    start=start_datetime,
                    end_decisions=end_datetime
                ),
                solution_datetime=solution_datetime,
                solver=solver,
                configuration=self.config.to_dict(),
                objective_function=self.env_objective_function().item(),
                objective_history=dict(
                    objective_values_eur=obj_history_values,
                    time_stamps_s=obj_history_times,
                ),
                dams=[
                    dict(
                        id=dam_id,
                        flows=clipped_flows[self.instance.get_order_of_dam(dam_id) - 1].tolist(),
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

        self.river_basin.deep_update(solution.get_flows_array(), is_relvars=False)
        obj = self.env_objective_function()

        return obj.item()
