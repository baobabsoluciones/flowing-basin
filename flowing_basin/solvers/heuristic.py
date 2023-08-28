from flowing_basin.core import Instance, Solution, Experiment, Configuration
from dataclasses import dataclass


@dataclass(kw_only=True)
class HeuristicConfiguration(Configuration):

    flow_smoothing: int = 0


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
        solution: Solution = None,
    ):

        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        time_steps = list(range(self.instance.get_largest_impact_horizon()))
        prices = self.instance.get_all_prices()
        self.time_steps_sorted = sorted(time_steps, key=lambda x: prices[x], reverse=True)

    def flow_to_volume(self, flow: float) -> float:

        """
        Turn the given flow (m3/s) into volume (m3) taking into account the size of the time step (s)
        """

        volume = flow * self.instance.get_time_step_seconds()
        return volume

    def calculate_available_volume_for_dam(self, dam_id: str) -> list[float]:

        """
        List with the available volume in the given dam at the end of every time step (m3)
        This is the extra volume (i.e., volume - minimum volume) the dam would have if no flow was taken out from it
        """

        # Calculate the volume added every time step
        unregulated_flows = self.instance.get_all_unregulated_flows_of_dam(dam_id)
        if self.instance.get_order_of_dam(dam_id) == 1:
            incoming_flows = self.instance.get_all_incoming_flows()
            added_volumes = [
                self.flow_to_volume(incom + unreg)
                for incom, unreg in zip(incoming_flows, unregulated_flows)
            ]
        else:
            # added_volumes = flow_to_volume(unregulated_flows + turbined_flow_previous_dam)
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
