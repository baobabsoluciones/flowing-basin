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
            config: HeuristicConfiguration,
            flow_contribution: list[float],
            paths_power_models: dict[str, str] = None,
    ):

        self.dam_id = dam_id
        self.instance = instance
        self.config = config
        self.flow_contribution = flow_contribution

        # Important constants
        self.time_steps = list(range(self.instance.get_largest_impact_horizon()))
        self.min_vol = self.instance.get_min_vol_of_dam(self.dam_id)
        self.max_available_vol = self.instance.get_max_vol_of_dam(self.dam_id) - self.min_vol

        # Dynamic values
        self.assigned_flows = [0 for _ in range(self.instance.get_largest_impact_horizon())]
        self.added_volumes = self.calculate_added_volumes()
        self.available_volumes = self.calculate_available_volumes()

        # Simulator
        self.dam = Dam(
            idx=self.dam_id,
            instance=self.instance,
            paths_power_models=paths_power_models,
            flow_smoothing=self.config.flow_smoothing,
            num_scenarios=1,
            mode=self.config.mode,
        )

    def sort_time_steps(self) -> list[int]:

        """
        List with the time steps sorted
        according to the price in their lags
        """

        time_steps = list(range(self.instance.get_largest_impact_horizon()))
        prices = self.instance.get_all_prices()

        verification_lags = self.instance.get_verification_lags_of_dam(self.dam_id)
        weights = [prices[time_step + verification_lags[0]] for time_step in time_steps if time_step + verification_lags[0] < len(prices)]
        weights[len(weights):] = [0 for _ in time_steps[len(weights):]]
        sorted_time_steps = sorted(time_steps, key=lambda x: weights[x], reverse=True)

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

    def calculate_added_volumes(self) -> list[float]:

        """
        List with the volume (m3) added in every time step with the incoming/unregulated flows
        """

        unregulated_flows = self.instance.get_all_unregulated_flows_of_dam(self.dam_id)
        added_volumes = [
            self.volume_from_flow(contrib + unreg - assigned)
            for contrib, unreg, assigned in zip(self.flow_contribution, unregulated_flows, self.assigned_flows)
        ]

        return added_volumes

    def calculate_available_volumes(self) -> list[float]:

        """
        Obtain a list with the available volume (m3) in the dam at the end of every time step.
        This is the extra volume (i.e., volume - minimum volume) the dam has
        with the currently assigned flows.
        """

        # Calculate the initially available volume
        initial_available_volume = self.instance.get_initial_vol_of_dam(self.dam_id) - self.min_vol

        # Calculate the available volume at every time step
        current_volume = initial_available_volume
        available_volumes = []
        for added_volume in self.added_volumes:
            current_volume += added_volume
            current_volume = min(current_volume, self.max_available_vol)
            available_volumes.append(current_volume)

        return available_volumes

    def calculate_max_vol_buffer(self, time_step: int) -> tuple[float, int]:

        """
        Calculate the amount of volume (m3)
        that can be taken out in the given time step
        without affecting the future time steps at all.
        This is given by the added volumes (m3) while on maximum volume.
        The method also returns the time step when the buffer has already ended.
        """

        total_unused_vol = 0.
        running_time_step = time_step
        while self.available_volumes[running_time_step] == self.max_available_vol:
            # We add the unused volume, which is the volume added when the reservoir was already full
            vol_used = self.max_available_vol - self.available_volumes[running_time_step - 1]
            vol_unused = self.added_volumes[running_time_step] - vol_used
            assert vol_unused >= - 1e-6, (
                f"Vol unused has a negative value {vol_unused} for time step {running_time_step}, with "
                f"previous volume {self.available_volumes[running_time_step - 1]} and "
                f"added volume {self.added_volumes[running_time_step]}."
            )
            total_unused_vol += vol_unused
            running_time_step += 1
            if running_time_step > self.time_steps[-1]:
                break

        return total_unused_vol, running_time_step

    def calculate_actual_available_volume(self, time_step: int) -> float:

        """
        Calculate the actual available volume in the given time step
        (i.e. the volume that, if consumed, will not leave negative volume in future time steps).

        This is, a priori, the minimum available volume of all future time steps;
        however, time steps with maximum volume provide a buffer
        that can be used without affecting the remaining volumes.

        As such, the actual available volume is the maximum of the following:
        - The minimum volume up to the FIRST max vol buffer (without exceeding this buffer);
        - The minimum volume up to the SECOND max vol buffer (without exceeding this buffer);
        - etc.
        Taking the maximum of these is necessary since the first buffer may be lower than the min vol up to the second
        (this rarely happens, though)
        """

        affected_volumes = []
        actual_available_volume = 0.

        running_time_step = time_step
        while running_time_step <= self.time_steps[-1]:
            affected_volumes.append(self.available_volumes[running_time_step])
            max_vol_buffer, time_step_after_buffer = self.calculate_max_vol_buffer(running_time_step)
            if max_vol_buffer > 0.:
                actual_available_volume = max(actual_available_volume, min(min(affected_volumes), max_vol_buffer))
                running_time_step = time_step_after_buffer
                # break  # <-- uncomment this to consider only the first buffer
            else:
                running_time_step += 1
        actual_available_volume = max(actual_available_volume, min(affected_volumes))  # <-- comment this if you break

        # Print in blue the affected volumes and in red the rest
        # final_running_time_step = running_time_step
        # red = "\033[31m"
        # blue = "\033[34m"
        # black = "\033[0m"
        # print(
        #     f"For time step {time_step:0>3} (max vol buffer {max_vol_buffer:0>8.2f}; "
        #     f"flow {self.max_flow_from_available_volume(actual_available_volume):0>5.2f}): "
        #     f"Available volumes: {', '.join([f'{i:0>8.2f}' for i in self.available_volumes[:time_step]])}, "
        #     f"{blue}{', '.join([f'{i:0>8.2f}' for i in self.available_volumes[time_step:final_running_time_step+1]])}{black}, "
        #     f"{red}{', '.join([f'{i:0>8.2f}' for i in self.available_volumes[final_running_time_step+1:]])}{black}"
        # )
        # print(
        #     f"For time step {time_step:0>3} (max vol buffer {max_vol_buffer:0>8.2f}; "
        #     f"flow {self.max_flow_from_available_volume(actual_available_volume):0>5.2f}): "
        #     f"Added     volumes: {', '.join([f'{i:0>8.2f}' for i in self.added_volumes])}"
        # )
        # self.clean_flows_and_volumes()

        return actual_available_volume

    def max_flow_from_available_volume(self, available_volume: float) -> float:

        """
        Compute the maximum flow (m3/s) you can take from the dam
        in a time step with the given available volume (m3)
        """

        max_flow = self.instance.get_max_flow_of_channel(self.dam_id)
        max_flow = min(max_flow, available_volume / self.instance.get_time_step_seconds())
        return max_flow

    @staticmethod
    def clean_list(lst: list, epsilon: float = 1e-6):

        """
        Turn the very small negative values (e.g. -3.8290358538183177e-16)
        of the list into 0.
        If these negative values are not small, raise an error.
        """

        i = 0
        while i < len(lst):
            if lst[i] < 0:
                if abs(lst[i]) < epsilon:
                    lst[i] = 0.
                else:
                    raise ValueError(
                        f"The given list has a negative value larger than {epsilon} "
                        f"in index {i}: {lst[i]}."
                    )
            i += 1

    def clean_flows_and_volumes(self):

        """
        Clean the assigned flows and available volumes lists
        """

        self.clean_list(self.assigned_flows)
        self.clean_list(self.available_volumes)

    def solve(self) -> tuple[list[float], list[float]]:

        """
        Get the exiting flows (m3/s) recommended by the heuristic,
        as well as the predicted volumes (m3) with these recommended flow assignments.
        """

        for time_step in self.sort_time_steps():
            available_volume = self.calculate_actual_available_volume(time_step)
            self.assigned_flows[time_step] = self.max_flow_from_available_volume(available_volume)  # noqa
            self.added_volumes = self.calculate_added_volumes()
            self.available_volumes = self.calculate_available_volumes()

        self.clean_flows_and_volumes()
        predicted_volumes = [available_vol + self.min_vol for available_vol in self.available_volumes]

        return self.assigned_flows, predicted_volumes

    def simulate(self) -> tuple[list[float], list[float], list[float], float, float]:

        """
        Use the river basin simulator to get the
        actual volumes (m3), turbined flows (m3/s), actual exiting flows (m3/s) and total income (€)
        with the current assigned flows
        """

        turbined_flows = []
        actual_exiting_flows = []
        actual_volumes = []

        # Calculate volumes with the current assigned flows
        self.dam.reset(instance=self.instance, flow_smoothing=self.config.flow_smoothing, num_scenarios=1)
        for time_step in self.time_steps:
            turbined_flow = self.dam.update(
                price=self.instance.get_price(time_step),
                flow_out=np.array([self.assigned_flows[time_step]]),
                incoming_flow=self.instance.get_incoming_flow(time_step),
                unregulated_flow=self.instance.get_unregulated_flow_of_dam(time_step, self.dam.idx),
                turbined_flow_of_preceding_dam=np.array([0]),
            )
            turbined_flows.append(turbined_flow.item())
            actual_exiting_flows.append(self.dam.flow_out_clipped2.item())
            actual_volumes.append(self.dam.volume.item())

        # Income (WITHOUT startup costs and objective final volumes)
        dam_income = self.dam.channel.power_group.acc_income.item()

        # Net income (WITH startup costs and objective final volumes)
        final_volume = self.dam.final_volume
        volume_shortage = max(0, self.config.volume_objectives[self.dam_id] - final_volume)
        volume_exceedance = max(0, final_volume - self.config.volume_objectives[self.dam_id])
        num_startups = self.dam.channel.power_group.acc_num_startups
        num_limit_zones = self.dam.channel.power_group.acc_num_times_in_limit
        penalty = (
                self.config.volume_shortage_penalty * volume_shortage
                + self.config.startups_penalty * num_startups
                + self.config.limit_zones_penalty * num_limit_zones
        )
        bonus = self.config.volume_exceedance_bonus * volume_exceedance
        dam_net_income = dam_income + bonus - penalty

        return actual_volumes, turbined_flows, actual_exiting_flows, dam_income, dam_net_income


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

        self.single_dam_solvers = {
            dam_id: HeuristicSingleDam(
                dam_id=dam_id,
                instance=instance,
                flow_contribution=self.instance.get_all_incoming_flows(),
                # TODO: for dam2, this should actually be the turbined flows of the preceding dam
                config=config,
            )
            for dam_id in self.instance.get_ids_of_dams()
        }

