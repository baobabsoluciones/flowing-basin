from flowing_basin.core import Instance, Solution, Experiment, Configuration
from flowing_basin.tools import Dam
from dataclasses import dataclass
import numpy as np


@dataclass(kw_only=True)
class HeuristicConfiguration(Configuration):

    flow_smoothing: int = 0
    mode: str = "nonlinear"

    def __post_init__(self):
        valid_modes = {"linear", "nonlinear"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid value for 'mode': {self.mode}. Allowed values are {valid_modes}")


class HeuristicSingleDam:

    def __init__(
            self,
            dam_id: str,
            instance: Instance,
            sorted_time_steps: list[float],
            flow_contribution: list[float],
            config: HeuristicConfiguration,
    ):

        self.dam_id = dam_id
        self.instance = instance
        self.sorted_time_steps = sorted_time_steps
        self.flow_contribution = flow_contribution

        # Constant values
        self.added_volumes = self.calculate_added_volumes()

        # Values changed while finding the heuristic solution
        self.assigned_flows = [0 for _ in range(self.instance.get_largest_impact_horizon())]
        self.available_volumes = self.calculate_available_volumes()

    def volume_from_flow(self, flow: float) -> float:

        """
        Turn the given flow (m3/s) into volume (m3) taking into account the size of the time step (s)

        Note that
         - if the given flow is an incoming/unregulated flow, this method gives the volume the reservoir will gain
         - if the given flow is an exiting flow, this method gives the minimum volume required in the reservoir
        """

        volume = flow * self.instance.get_time_step_seconds()
        return volume

    def calculate_added_volumes(self) -> list[float]:

        """
        List with the volume (m3) added in every time step with the incoming/unregulated flows
        """

        unregulated_flows = self.instance.get_all_unregulated_flows_of_dam(self.dam_id)
        added_volumes = [
            self.volume_from_flow(contrib + unreg)
            for contrib, unreg in zip(self.flow_contribution, unregulated_flows)
        ]

        return added_volumes

    def calculate_available_volumes(self) -> list[float]:

        """
        List with the available volume (m3) in the dam at the end of every time step
        This is the extra volume (i.e., volume - minimum volume) the dam would have if no flow was taken out from it
        """

        # Calculate the initially available volume
        initial_available_volume = (
                self.instance.get_initial_vol_of_dam(self.dam_id) - self.instance.get_min_vol_of_dam(self.dam_id)
        )

        # Calculate the available volume at every time step
        current_volume = initial_available_volume
        available_volumes = []
        for added_volume in self.added_volumes:
            current_volume += added_volume
            # available_volume = min(current_volume, self.instance.get_max_vol_of_dam(self.dam_id))
            # available_volumes.append(available_volume)
            available_volumes.append(current_volume)

        return available_volumes

    def max_flow_from_available_volume(self, available_volume: float) -> float:

        """
        Compute the maximum flow (m3/s) you can take from the dam
        in a time step with the given available volume (m3)
        """

        max_flow = self.instance.get_max_flow_of_channel(self.dam_id)
        max_flow = min(max_flow, available_volume / self.instance.get_time_step_seconds())
        return max_flow

    def solve(self) -> tuple[list[float], list[float]]:

        """
        Get the exiting flows (m3/s) recommended by the heuristic,
        as well as the predicted volumes (m3)
        """

        verification_lags = self.instance.get_verification_lags_of_dam(self.dam_id)

        for time_step in self.sorted_time_steps:

            time_step_lags = [int(time_step - lag) for lag in verification_lags if time_step - lag >= 0]

            for time_step_lag in time_step_lags:

                # The actual available volume in this time step
                # is the minimum available volume of all time steps in the future,
                # since we cannot let any of them go below zero
                available_volume = min(self.available_volumes[time_step_lag:])

                # Assign the maximum possible flow given the actual available volume
                assigned_flow = self.max_flow_from_available_volume(available_volume)
                self.assigned_flows[time_step_lag] = assigned_flow  # noqa

                # Reduce the available volume in the current and next time steps
                required_volume = self.volume_from_flow(assigned_flow)
                self.available_volumes[time_step_lag:] = [
                    vol - required_volume
                    for vol in self.available_volumes[time_step_lag:]
                ]

        assert all([available_vol >= 0 for available_vol in self.available_volumes]), (
            "Remaining available volumes should be positive"
        )
        assert all([0 <= flow <= self.instance.get_max_flow_of_channel(self.dam_id) for flow in self.assigned_flows]), (
            "Assigned flows should be positive and lower than the maximum flow"
        )

        min_vol = self.instance.get_min_vol_of_dam(self.dam_id)
        predicted_volumes = [available_vol + min_vol for available_vol in self.available_volumes]

        return self.assigned_flows, predicted_volumes


class Heuristic(Experiment):

    """
    Heuristic solver
    This solver gives a good solution (not optimal) to the river basin problem.
    The solution is calculated by assigning the highest possible flows to the time steps with the highest energy prices,
    taking into account the lags of the dams.
    """

    def __init__(
        self,
        instance: Instance,
        config: HeuristicConfiguration,
        paths_power_models: dict[str, str] = None,
        solution: Solution = None,
    ):

        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        self.config = config
        self.paths_power_models = paths_power_models

        sorted_time_steps = self.sort_time_steps()
        self.single_dam_solvers = {
            dam_id: HeuristicSingleDam(
                dam_id=dam_id,
                instance=instance,
                sorted_time_steps=sorted_time_steps,
                flow_contribution=self.instance.get_all_incoming_flows(),
                # TODO: for dam2, this should actually be the turbined flows of the preceding dam
                config=config,
            )
            for dam_id in self.instance.get_ids_of_dams()
        }

    def sort_time_steps(self) -> list[int]:

        """
        List with the time steps sorted by their price
        """

        time_steps = list(range(self.instance.get_largest_impact_horizon()))
        prices = self.instance.get_all_prices()
        sorted_time_steps = sorted(time_steps, key=lambda x: prices[x], reverse=True)

        return sorted_time_steps

    def turbined_flows_from_assigned_flows(
            self, dam_id: str, assigned_flows: list[float]
    ) -> tuple[list[float], list[float], list[float]]:

        """
        Using the simulator, obtain the turbined flows (m3/s) produced by the dam with the given assigned flows,
        as well as the actual exiting flows (m3/s) and volumes (m3)
        """

        dam = Dam(
            idx=dam_id,
            instance=self.instance,
            paths_power_models=self.paths_power_models,
            flow_smoothing=self.config.flow_smoothing,
            num_scenarios=1,
            mode=self.config.mode,
        )

        turbined_flows = []
        actual_exiting_flows = []
        actual_volumes = []

        for time_step in range(self.instance.get_largest_impact_horizon()):

            turbined_flow = dam.update(
                price=self.instance.get_price(time_step),
                flow_out=np.array([assigned_flows[time_step]]),
                incoming_flow=self.instance.get_incoming_flow(time_step),
                unregulated_flow=self.instance.get_unregulated_flow_of_dam(time_step, dam.idx),
                turbined_flow_of_preceding_dam=np.array([0]),
            )

            turbined_flows.append(turbined_flow.item())
            actual_exiting_flows.append(dam.flow_out_clipped2.item())
            actual_volumes.append(dam.volume.item())

        return turbined_flows, actual_exiting_flows, actual_volumes

