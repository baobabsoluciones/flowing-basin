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

    def sort_time_steps(self) -> list[int]:

        """
        List with the time steps sorted by their price
        """

        time_steps = list(range(self.instance.get_largest_impact_horizon()))
        prices = self.instance.get_all_prices()
        sorted_time_steps = sorted(time_steps, key=lambda x: prices[x], reverse=True)

        return sorted_time_steps

    def volume_from_flow(self, flow: float) -> float:

        """
        Turn the given flow (m3/s) into volume (m3) taking into account the size of the time step (s)

        Note that
         - if the given flow is an incoming/unregulated flow, this method gives the volume the reservoir will gain
         - if the given flow is an exiting flow, this method gives the minimum volume required in the reservoir
        """

        volume = flow * self.instance.get_time_step_seconds()
        return volume

    def calculate_available_volumes(self, dam_id: str) -> list[float]:

        """
        List with the available volume (m3) in the indicated dam at the end of every time step
        This is the extra volume (i.e., volume - minimum volume) the dam would have if no flow was taken out from it
        """

        # Calculate the volume added every time step
        unregulated_flows = self.instance.get_all_unregulated_flows_of_dam(dam_id)
        if self.instance.get_order_of_dam(dam_id) == 1:
            incoming_flows = self.instance.get_all_incoming_flows()
            added_volumes = [
                self.volume_from_flow(incom + unreg)
                for incom, unreg in zip(incoming_flows, unregulated_flows)
            ]
        else:
            # added_volumes = volume_from_flow(unregulated_flows + turbined_flow_previous_dam)
            # TODO: implement this
            added_volumes = None

        # Calculate the initially available volume
        initial_available_volume = (
                self.instance.get_initial_vol_of_dam(dam_id) - self.instance.get_min_vol_of_dam(dam_id)
        )

        # Calculate the available volume at every time step
        current_volume = initial_available_volume
        available_volumes = []
        for added_volume in added_volumes:
            current_volume += added_volume
            available_volumes.append(current_volume)

        return available_volumes

    def max_flow_from_available_volume(self, dam_id: str, available_volume: float) -> float:

        """
        Compute the maximum flow (m3/s) you can take from the indicted dam
        in a time step with the given available volume (m3)
        """

        max_flow = self.instance.get_max_flow_of_channel(dam_id)
        max_flow = min(max_flow, available_volume / self.instance.get_time_step_seconds())
        return max_flow

    def solve_for_dam(self, dam_id: str) -> tuple[list[float], list[float]]:

        """
        Get the exiting flows (m3/s) recommended by the heuristic for the given dam,
        as well as the predicted volumes (m3)
        """

        assigned_flows = [0 for _ in range(self.instance.get_largest_impact_horizon())]
        verification_lags = self.instance.get_verification_lags_of_dam(dam_id)
        available_volumes = self.calculate_available_volumes(dam_id)
        time_steps = list(range(self.instance.get_largest_impact_horizon()))

        for time_step in self.sort_time_steps():

            time_step_lags = [time_step - lag for lag in verification_lags if time_step - lag >= 0]

            for time_step_lag in time_step_lags:

                # Available volume in this time step
                available_volume = available_volumes[time_step_lag]

                # The actual available volume is less, since
                # the volume left for the next time steps should be enough to keep their (previously assigned) flows
                for affected_time_step in time_steps[time_step_lag + 1:]:
                    volume_already_used = self.volume_from_flow(assigned_flows[affected_time_step])
                    remaining_volume = available_volumes[affected_time_step] - volume_already_used
                    available_volume = min(available_volume, remaining_volume)

                # Assign the maximum possible flow given the actual available volume
                assigned_flow = self.max_flow_from_available_volume(dam_id, available_volume)
                assigned_flows[time_step_lag] = assigned_flow  # noqa

                # Reduce the available volume in the next time steps
                required_volume = self.volume_from_flow(assigned_flow)
                available_volumes[time_step_lag + 1:] = [
                    vol - required_volume
                    for vol in available_volumes[time_step_lag + 1:]
                ]

        assert all([available_vol >= 0 for available_vol in available_volumes]), (
            "Remaining available volumes should be positive"
        )
        assert all([0 <= flow <= self.instance.get_max_flow_of_channel(dam_id) for flow in assigned_flows]), (
            "Assigned flows should be positive and lower than the maximum flow"
        )

        min_vol = self.instance.get_min_vol_of_dam(dam_id)
        predicted_volumes = [
            available_vol - self.volume_from_flow(assigned_flow) + min_vol
            for available_vol, assigned_flow in zip(available_volumes, assigned_flows)
        ]

        return assigned_flows, predicted_volumes

    def turbined_flows_from_assigned_flows(
            self, dam_id: str, assigned_flows: list[float]
    ) -> tuple[list[float], list[float], list[float]]:

        """
        Obtain the turbined flows (m3/s) produced by the dam for the given assigned flows,
        as well as the actual exiting flows (m3/s) and volume (m3) with these assigned flows
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

