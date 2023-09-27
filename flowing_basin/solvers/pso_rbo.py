from flowing_basin.core import Experiment, Instance, Solution
from flowing_basin.solvers import PSOConfiguration, HeuristicConfiguration, PSO, Heuristic
from cornflow_client.constants import SOLUTION_STATUS_FEASIBLE, STATUS_UNDEFINED
from dataclasses import dataclass
import numpy as np
from time import perf_counter


@dataclass(kw_only=True)
class PsoRboConfiguration(PSOConfiguration, HeuristicConfiguration):
    def __post_init__(self):
        if not self.random_biased_flows and not self.random_biased_sorting:
            raise ValueError(
                "Both random biased flows and sorting are False. "
                "At least one of them must be True to generate the initial solutions in PSO-RBO."
            )


class PsoRbo(Experiment):
    def __init__(
            self,
            instance: Instance,
            config: PsoRboConfiguration,
            paths_power_models: dict[str, str] = None,
            solution: Solution = None,
            verbose: bool = True
    ):

        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        self.config = config
        self.verbose = verbose

        self.pso = PSO(
            instance=self.instance,
            config=self.config,
            paths_power_models=paths_power_models,
            verbose=self.verbose
        )

    def relvars_from_flows(self, clipped_flows: np.ndarray) -> np.ndarray:

        """
        Turn the array of flows into an array of relvars (relative variations).
        For the flows and relvars to actually be equivalent, the flows must be the actual flows exiting the dams
        (i.e. they must be already smoothed and clipped).

        :param clipped_flows:
            Array of shape num_time_steps x num_dams x num_scenarios with
            the flows that go through each channel in every time step for every scenario (m3/s)
        :return:
            Array of shape num_time_steps x num_dams x num_scenarios with
            the variation of flow (as a fraction of flow max) through each channel in every time step
        """

        # Initialize relvars as an empty array of shape
        relvars = np.array([]).reshape(
            (0, self.instance.get_num_dams(), self.config.num_particles)
        )

        # Max flow through each channel, as an array of shape num_dams x num_scenarios
        max_flows = np.repeat(
            [
                self.instance.get_max_flow_of_channel(dam_id)
                for dam_id in self.instance.get_ids_of_dams()
            ],
            self.config.num_particles,
        ).reshape((self.instance.get_num_dams(), self.config.num_particles))

        # Initialize old flow
        # Flow that went through the channels in the previous time step, as an array of shape num_dams x num_scenarios
        old_flow = np.repeat(
            [
                self.instance.get_initial_lags_of_channel(dam_id)[0]
                for dam_id in self.instance.get_ids_of_dams()
            ],
            self.config.num_particles,
        ).reshape((self.instance.get_num_dams(), self.config.num_particles))

        # Get array num_time_steps x num_dams x num_time_steps with the relvars
        for flow in clipped_flows:
            relvar = (flow - old_flow) / max_flows
            relvars = np.vstack((relvars, [relvar]))
            old_flow = flow

        return relvars

    def generate_initial_solutions(self) -> np.ndarray:

        """

        :return:
            Array of shape num_time_steps x num_dams x num_scenarios with the initial solutions, represented as
            the flows that go through each channel in every time step for every scenario (m3/s)
        """

        flows = np.array([]).reshape(
            (self.instance.get_largest_impact_horizon(), self.instance.get_num_dams(), 0)
        )

        # First particle - greedy solution (heuristic)
        heuristic = Heuristic(config=self.config, instance=self.instance, greedy=True, do_tests=True)
        heuristic.solve()
        flows = np.concatenate(
            [
                flows,
                heuristic.solution.get_flows_array(),
            ],
            axis=2,
        )

        # Remaining particles - random biased solutions (RBO)
        for particle in range(1, self.config.num_particles):
            heuristic = Heuristic(config=self.config, instance=self.instance, greedy=False, do_tests=True)
            heuristic.solve()
            flows = np.concatenate(
                [
                    flows,
                    heuristic.solution.get_flows_array(),
                ],
                axis=2,
            )

        return flows

    def solve(self, options: dict = None) -> dict:

        # Generate initial solutions (RBO)
        rbo_start_time = perf_counter()
        initial_flows_or_relvars = self.generate_initial_solutions()
        if self.config.use_relvars:
            initial_flows_or_relvars = self.relvars_from_flows(initial_flows_or_relvars)
        rbo_exec_time = perf_counter() - rbo_start_time
        if self.verbose:
            print(
                f"Generated {self.config.num_particles} initial solutions in {rbo_exec_time}s "
                f"with {self.config.random_biased_sorting=} and {self.config.random_biased_flows=}."
            )

        # Execute PSO
        self.pso.solve(
            initial_solutions=initial_flows_or_relvars,
            time_offset=rbo_exec_time,
            solver="PSO-RBO"
        )
        self.solution = self.pso.solution

        return dict(
            status_sol=SOLUTION_STATUS_FEASIBLE,
            status=STATUS_UNDEFINED,
        )

