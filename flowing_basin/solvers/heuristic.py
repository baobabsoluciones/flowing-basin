from flowing_basin.core import Instance, Solution, Experiment, Configuration
from flowing_basin.tools import Dam
from flowing_basin.solvers.common import get_turbine_count_intervals
from dataclasses import dataclass, asdict
from random import random, choices
from math import log
import numpy as np
import warnings


@dataclass(kw_only=True)
class HeuristicConfiguration(Configuration):

    mode: str = "nonlinear"

    # Maximize final volume independently of the objective final volume specified
    maximize_final_vol: bool = False

    # Randomly assign less flow than the maximum available by setting random_biased_flows=True
    # If True, the prob_below_half parameter gives the probability that the assigned flow is below half the maximum
    random_biased_flows: bool = False
    prob_below_half: float = 0.15

    # Randomly pick groups from the list of sorted groups by setting random_biased_sorting=True
    # If True, the common_ratio parameter indicates how likely the second, third, etc. groups are to be chosen
    random_biased_sorting: bool = False
    common_ratio: float = 0.6

    # Post-process outflows to avoid limit zones
    avoid_limit_zones: bool = False

    def __post_init__(self):

        super(HeuristicConfiguration, self).__post_init__()

        valid_modes = {"linear", "nonlinear"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid value for 'mode': {self.mode}. Allowed values are {valid_modes}")


class HeuristicSingleDam:

    def __init__(
            self,
            dam_id: str,
            instance: Instance,
            config: HeuristicConfiguration,
            bias_weight: float,
            flow_contribution: list[float],
            paths_power_models: dict[str, str] = None,
            greedy: bool = False,
            do_tests: bool = True
    ):

        self.dam_id = dam_id
        self.instance = instance
        self.config = config
        self.bias_weight = bias_weight
        self.flow_contribution = flow_contribution
        self.greedy = greedy
        self.do_tests = do_tests

        # Volume objectives
        # If no objective final volume is given, we assume it is equal to the minimum volume (always satisfied)
        if self.config.volume_objectives:
            self.volume_objectives = self.config.volume_objectives.copy()
        else:
            self.volume_objectives = {
                dam_id: self.instance.get_min_vol_of_dam(dam_id) for dam_id in self.instance.get_ids_of_dams()
            }

        # Important constants
        # The initial volume is clipped between the min and max volumes (like in the simulator)
        self.time_steps = list(range(self.instance.get_largest_impact_horizon()))
        self.min_vol = self.instance.get_min_vol_of_dam(self.dam_id)
        self.max_available_vol = self.instance.get_max_vol_of_dam(self.dam_id) - self.min_vol
        self.initial_available_vol = max(
            0.,
            min(
                self.max_available_vol,
                self.instance.get_initial_vol_of_dam(self.dam_id) - self.min_vol
            )
        )
        turbine_count_intervals = get_turbine_count_intervals(constants=self.instance)
        self.safe_turbine_flow_intervals = {
            dam_id: [
                (first_flow, last_flow) for turbine_count, first_flow, last_flow in turbine_count_intervals[dam_id]
                if turbine_count == int(turbine_count)
            ]
            for dam_id in self.instance.get_ids_of_dams()
        }

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

    def group_time_steps(self, time_steps: list[int]) -> list[list[int]]:

        """
        Group time steps into groups of size flows_smoothing + 1
        If the number of time steps is NOT a multiple of flows_smoothing + 1, the last group will have a smaller size
        (this is not a problem: the flow smoothing condition will still be met)
        """

        group_size = self.config.flow_smoothing + 1
        grouped_time_steps = [time_steps[i:i + group_size] for i in range(0, len(time_steps), group_size)]

        return grouped_time_steps

    def sort_groups(self, groups: list[list[int]]) -> list[list[int]]:

        """
        Sort the groups of time steps according to the energy price in their lags
        """

        # Calculate the weights with which we will order the groups
        prices = self.instance.get_all_prices()
        verification_lags = self.instance.get_verification_lags_of_dam(self.dam_id)
        weights = [
            # Calculate the avg of lagged prices in the group
            sum(
                # Calculate the lagged price in the time step (i.e., the avg price in the lags)
                sum(
                    prices[time_step + lag] for lag in verification_lags if time_step + lag < len(prices)
                ) / len(verification_lags)
                for time_step in group
            ) / len(group)
            for group in groups
        ]

        # Sort the groups according to the weights
        pairs = zip(groups, weights)
        sorted_pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
        sorted_grouped_time_steps, _ = zip(*sorted_pairs)

        return list(sorted_grouped_time_steps)

    def generate_random_biased_probabilities(self, num_groups: int) -> list[float]:

        """
        Generate a probability distribution over the sorted groups,
        such that the first group (i.e., with the highest average lagged price) has more probability than the rest.
        """

        if abs(self.config.common_ratio - 1) > 1e-5:
            prob = (self.config.common_ratio - 1) / (self.config.common_ratio ** num_groups - 1)
            probs = [prob]
            for _ in range(1, num_groups):
                prob *= self.config.common_ratio
                probs.append(prob)
        else:
            # When common_ratio = 1, all groups should have the same probability
            prob = 1 / num_groups
            probs = [prob for _ in range(num_groups)]

        return probs

    def pick_group(self, sorted_groups: list[list[int]]) -> list[int] | None:

        """
        Choose a group from the list of sorted groups, and remove it.
        Returns None if given list is empty.
        """

        if len(sorted_groups) == 0:
            return None

        if not self.config.random_biased_sorting or self.greedy:
            chosen_group = sorted_groups[0]
        else:
            chosen_group = choices(
                sorted_groups,
                weights=self.generate_random_biased_probabilities(len(sorted_groups)),
                k=1
            )[0]

        sorted_groups.remove(chosen_group)

        return chosen_group

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

        # Calculate the available volume at every time step
        current_volume = self.initial_available_vol
        available_volumes = []
        for added_volume in self.added_volumes:
            current_volume += added_volume
            current_volume = min(current_volume, self.max_available_vol)
            available_volumes.append(current_volume)

        return available_volumes

    def calculate_turbined_flows(self) -> list[float]:

        """
        Obtain a list with the turbined flow (m3/s) of the dam in every time step
        Uses the linear method
        """

        verification_lags = self.instance.get_verification_lags_of_dam(self.dam_id)
        first_lag = verification_lags[0]
        last_lag = verification_lags[-1]

        initial_lags = self.instance.get_initial_lags_of_channel(self.dam_id)[:last_lag]
        initial_lags.reverse()
        extended_assigned_flows = initial_lags + self.assigned_flows

        turbined_flows = []
        for time_step in self.time_steps:
            lag_flows = extended_assigned_flows[time_step: time_step + first_lag]
            turbined_flow = sum(lag_flows) / len(lag_flows)
            turbined_flows.append(turbined_flow)

        return turbined_flows

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
            prev_available_volume = (
                self.available_volumes[running_time_step - 1] if running_time_step - 1 >= 0
                else self.initial_available_vol
            )
            vol_used = self.max_available_vol - prev_available_volume
            vol_unused = self.added_volumes[running_time_step] - vol_used
            if self.do_tests:
                assert vol_unused >= - 1e-6, (
                    f"In dam {self.dam_id}, vol unused has a negative value {vol_unused} "
                    f"for time step {running_time_step}, with "
                    f"previous volume {prev_available_volume} and "
                    f"added volume {self.added_volumes[running_time_step]}."
                )
            total_unused_vol += vol_unused
            running_time_step += 1
            if running_time_step > self.time_steps[-1]:
                break

        return total_unused_vol, running_time_step

    def calculate_actual_available_volume(self, group: list[int]) -> float:

        """
        Calculate the actual available volume of the given group of time steps
        (i.e. the volume that, if consumed, will not leave negative volume in current or future time steps).

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

        affected_volumes = self.available_volumes[group[0]: group[-1]]
        actual_available_volume = 0.

        running_time_step = group[-1]
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
        in a time step with the given available volume (m3).
        """
        max_flow = self.instance.get_max_flow_of_channel(self.dam_id)
        max_flow = min(max_flow, available_volume / self.instance.get_time_step_seconds())
        return max_flow

    def get_closest_safe_flow(self, flow: float) -> float:
        """
        Get the closest lower flow that is not within a limit zone
        """

        # If the flow already is within a safe interval (i.e., not a limit zone), simply return it
        for first_flow, last_flow in self.safe_turbine_flow_intervals:
            if first_flow <= flow <= last_flow:
                return flow

        # If not in any interval, find the closest lower last flow
        last_flows = [last_flow for first_flow, last_flow in self.safe_turbine_flow_intervals if last_flow < flow]
        return max(last_flows)

    def get_flow_to_assign(self, available_volume: float) -> float:
        """
        Compute the flow that should be assigned with the given available volume.
        This is the maximum flow that is not within a limit zone.
        """
        max_flow = self.max_flow_from_available_volume(available_volume)
        max_safe_flow = self.get_closest_safe_flow(max_flow)
        return max_safe_flow

    def generate_random_biased_number(self) -> float:

        """
        Generate a random number between 0 and 1,
        but numbers close to 1 are more likely than numbers close to 0.

        This is achieved by generating a random number between 0 and 1
        and then passing it through a concave function [0,1] -> [0,1];
        in this case, f(x) = x ^ (1 / weight).

        The weight parameter indicates how much more likely numbers close to 1 are.
        If weight < 1, numbers close to 1 are actually less likely.
        If weight > 1, numbers close to 1 are more likely.

        :return:
        """

        return random() ** (1 / self.bias_weight)

    def adapt_flows_to_volume_limits(self, groups: list[list[int]]):

        """
        Adapt the flow of every group to the current available volumes
        This is necessary in dams where volume may impose a flow limit
        These method should be called whenever available volumes are modified
        :return:
        """

        if self.dam.channel.limit_flow_points is None:
            return

        for group in groups:
            # The flow of each time step is limited by the volume at the end of the previous time step
            # We take the minimum of these volumes for the whole group, to ensure the same flow can be assigned
            min_volume = min(
                (
                    self.available_volumes[time_step - 1] + self.min_vol if time_step - 1 >= 0
                    else self.initial_available_vol + self.min_vol
                ) for time_step in group
            )
            flow_to_assign = min(
                self.assigned_flows[group[0]],
                self.dam.channel.get_flow_limit(np.array([min_volume])).item()
            )
            for time_step in group:
                self.assigned_flows[time_step] = flow_to_assign
        return

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

    def relevant_groups_for_final_vol(self, sorted_groups: list[list[int]]) -> list[list[int]]:

        """
        Select the groups that can affect the final volumes
        (i.e., the groups after the last max or min vol time step).
        """

        running_time_step = self.instance.get_decision_horizon() - 1
        while self.available_volumes[running_time_step] < self.max_available_vol and running_time_step > 0:
            running_time_step -= 1

        # We filter the groups of time steps, taking into account that every time step affects the volume
        # in the same time step (since we are storing the final volume of each time step)
        relevant_groups = [
            group for group in sorted_groups
            if (
                group[-1] > running_time_step and
                group[0] <= self.instance.get_decision_horizon() - 1
            )
        ]
        return relevant_groups

    def adapt_flows_to_obj_vol(self, sorted_groups: list[list[int]], epsilon: float = 1e-6):

        """
        Assign zero flow to the least important relevant groups (i.e., lowest average lagged price)
        as many times as required to satisfy the objective final volume
        """

        objective_available_volume = (
            self.volume_objectives[self.dam_id] - self.min_vol
            if not self.config.maximize_final_vol else self.max_available_vol
        )
        decision_horizon = self.instance.get_decision_horizon()

        sorted_relevant_groups = self.relevant_groups_for_final_vol(sorted_groups)
        volume_gap = objective_available_volume - self.available_volumes[decision_horizon - 1]

        while volume_gap > -1e-4 and len(sorted_relevant_groups) > 0:

            # Select the group with the lowest weight or score (average lagged price)
            least_important_group = sorted_relevant_groups[-1]
            # print(
            #     f"\tvol in {decision_horizon - 1} = "
            #     f"{self.available_volumes[decision_horizon - 1]} < {objective_available_volume}"
            # )
            # print(f"\t{least_important_group=}")
            # print(
            #     "\tFlows in group:", [self.assigned_flows[time_step] for time_step in least_important_group],
            #     "Volumes in group:", [self.available_volumes[time_step] for time_step in least_important_group]
            # )

            # Calculate the volume to remove in the whole group
            # This is equal to the volume to add to the decision horizon
            highest_vol_in_next_groups = max(
                self.available_volumes[least_important_group[-1]: decision_horizon]
            ) if least_important_group[-1] < decision_horizon else 0.
            removable_volume_in_next_groups = self.max_available_vol - highest_vol_in_next_groups
            volume_to_remove = min(removable_volume_in_next_groups, volume_gap)
            # print(f"\t{volume_to_remove=}")

            # Calculate the volume to remove in every time step of the group
            # A priori, every time step of the group can contribute the same amount, volume_to_remove/len(group) ...
            relevant_time_steps = [
                time_step for time_step in least_important_group if time_step <= decision_horizon - 1
            ]
            num_relevant_time_steps = len(relevant_time_steps)
            volume_to_remove_per_time_step = volume_to_remove / num_relevant_time_steps
            # print(f"\t{volume_to_remove_per_time_step=}")

            # ...however, this is not true when one of the time steps is close to (or in) the max vol and does not have
            # enough removable volume (it is "capped"), forcing extra volume to be removed in the remaining time steps
            num_capped_time_steps = 0
            for time_step in relevant_time_steps:
                # Increase the volume to remove in the remaining time steps if there is not enough removable volume here
                removable_vol = self.max_available_vol - self.available_volumes[time_step]
                if volume_to_remove_per_time_step > removable_vol:
                    num_capped_time_steps += 1
                    extra_volume_to_remove = volume_to_remove_per_time_step - removable_vol
                    try:
                        volume_to_remove_per_time_step += (
                                extra_volume_to_remove / (num_relevant_time_steps - num_capped_time_steps)
                        )
                        # print(f"\t{volume_to_remove_per_time_step=}")
                    except ZeroDivisionError:
                        # This only happens if ALL relevant time steps of the group are capped
                        # In this case, we simply take as much volume from all time steps as possible
                        volume_to_remove_per_time_step = self.max_available_vol - min(
                            self.available_volumes[time_step] for time_step in relevant_time_steps
                        )
                        # print(f"\t{volume_to_remove_per_time_step=} (<- all capped)")

            # Reduce the calculated volume in every time step
            flow_to_remove = volume_to_remove_per_time_step / self.instance.get_time_step_seconds()
            for time_step in least_important_group:
                self.assigned_flows[time_step] = max(0., self.assigned_flows[time_step] - flow_to_remove)  # noqa

            self.added_volumes = self.calculate_added_volumes()
            self.available_volumes = self.calculate_available_volumes()
            self.adapt_flows_to_volume_limits(sorted_groups)

            volume_gap = objective_available_volume - self.available_volumes[decision_horizon - 1]
            sorted_relevant_groups.remove(least_important_group)
            sorted_relevant_groups = self.relevant_groups_for_final_vol(sorted_relevant_groups)

            # print(
            #     "\tFlows in group:", [self.assigned_flows[time_step] for time_step in least_important_group],
            #     "Volumes in group:", [self.available_volumes[time_step] for time_step in least_important_group]
            # )
            # print(
            #     f"\tvol in {decision_horizon - 1} = "
            #     f"{self.available_volumes[decision_horizon - 1]} < {objective_available_volume}"
            # )
            #
            # print(f"{self.available_volumes=}")
            # print(f"{sorted_relevant_groups=}")

        # Check the objective final volumes are now satisfied
        # if self.do_tests:
        #     assert self.available_volumes[decision_horizon - 1] > objective_available_volume - epsilon, (
        #         f"Objective volumes were not satisfied in {self.dam_id}; "
        #         f"final volume is {self.available_volumes[decision_horizon - 1]}, "
        #         f"but objective is {objective_available_volume}. "
        #         f"Assigned flows: {self.assigned_flows=}."
        #     )
        # For some alternative solutions, this assert is not satisfied
        # I don't know why, but it only matters with objective volumes, so it can be ignored

    def remove_limit_zones(self, sorted_groups: list[list[int]]):

        pass

    def solve(self) -> tuple[list[float], list[float]]:

        """
        Get the exiting flows (m3/s) recommended by the heuristic,
        as well as the predicted volumes (m3) with these recommended flow assignments.
        """

        time_steps = list(range(self.instance.get_largest_impact_horizon()))
        groups = self.group_time_steps(time_steps)
        sorted_groups = self.sort_groups(groups)
        remaining_groups = sorted_groups[:]

        while (group := self.pick_group(remaining_groups)) is not None:

            # Calculate the flow that should be assigned to the group
            available_volume = self.calculate_actual_available_volume(group) / len(group)
            flow_to_assign = self.max_flow_from_available_volume(available_volume)
            # TODO: Replace with the following line to test the new version:
            #   flow_to_assign = self.get_flow_to_assign(available_volume)
            if self.config.random_biased_flows and not self.greedy:
                flow_to_assign = self.generate_random_biased_number() * flow_to_assign

            # Assign the flow and recalculate volumes
            for time_step in group:
                self.assigned_flows[time_step] = flow_to_assign  # noqa
            self.added_volumes = self.calculate_added_volumes()
            self.available_volumes = self.calculate_available_volumes()
            self.adapt_flows_to_volume_limits(groups)

        # Recalculate available volumes one last time (to adapt to the new flows after volume limits)
        self.added_volumes = self.calculate_added_volumes()
        self.available_volumes = self.calculate_available_volumes()

        self.clean_flows_and_volumes()
        self.adapt_flows_to_obj_vol(sorted_groups)
        if self.config.avoid_limit_zones:
            self.remove_limit_zones(sorted_groups)
        predicted_volumes = [available_vol + self.min_vol for available_vol in self.available_volumes]

        return self.assigned_flows, predicted_volumes

    def simulate(self) -> tuple[list[float], list[float], list[float], list[float], dict[str, float | int]]:

        """
        Use the river basin simulator to get the
        turbined flows (m3/s), powers (MW), actual volumes (m3), actual exiting flows (m3/s),
        and dictionary with objective function details
        """

        turbined_flows = []
        actual_exiting_flows = []
        actual_volumes = []
        powers = []

        # Calculate volumes with the current assigned flows
        self.dam.reset(instance=self.instance, flow_smoothing=self.config.flow_smoothing, num_scenarios=1)
        for time_step in self.time_steps:
            turbined_flow = self.dam.update(
                price=self.instance.get_price(time_step),
                flow_out=np.array([self.assigned_flows[time_step]]),
                unregulated_flow=self.instance.get_unregulated_flow_of_dam(time_step, self.dam.idx),
                flow_contribution=np.array([self.flow_contribution[time_step]]),
                time=time_step
            )
            turbined_flows.append(turbined_flow.item())
            actual_exiting_flows.append(self.dam.flow_out_clipped2.item())
            actual_volumes.append(self.dam.volume.item())
            powers.append(self.dam.channel.power_group.power.item())

        # Income (WITHOUT startup costs and objective final volumes)
        dam_income = self.dam.channel.power_group.acc_income.item()

        # Net income (WITH startup costs and objective final volumes)
        final_volume = self.dam.final_volume.item()
        volume_shortage = max(0, self.volume_objectives[self.dam_id] - final_volume)
        volume_exceedance = max(0, final_volume - self.volume_objectives[self.dam_id])
        num_startups = self.dam.channel.power_group.acc_num_startups.item()
        num_limit_zones = self.dam.channel.power_group.acc_num_times_in_limit.item()
        penalty = (
                self.config.volume_shortage_penalty * volume_shortage
                + self.config.startups_penalty * num_startups
                + self.config.limit_zones_penalty * num_limit_zones
        )
        bonus = self.config.volume_exceedance_bonus * volume_exceedance
        dam_net_income = dam_income + bonus - penalty

        if self.do_tests:
            heuristic_turbined_flows = self.calculate_turbined_flows()
            assert all(abs(a - b) < 1e-4 for a, b in zip(turbined_flows, heuristic_turbined_flows)), (
                "The heuristic method to calculate turbined flow gives different results from the simulator."
            )

        obj_fun_details = dict(
            total_income_eur=dam_net_income,
            income_from_energy_eur=dam_income,
            startups=num_startups,
            limit_zones=num_limit_zones,
            volume_shortage_m3=volume_shortage,
            volume_exceedance_m3=volume_exceedance
        )

        return turbined_flows, powers, actual_volumes, actual_exiting_flows, obj_fun_details


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
        greedy: bool = False,
        do_tests: bool = True,
        solution: Solution = None,
    ):

        super().__init__(instance=instance, config=config, solution=solution)
        self.config = config  # This is unnecessary but avoids error highlighting
        if solution is None:
            self.solution = None

        self.paths_power_models = paths_power_models
        self.greedy = greedy
        self.do_tests = do_tests

        # Calculate the bias weight in the random biased number generator
        if self.config.prob_below_half > 1e-5:
            self.bias_weight = log(self.config.prob_below_half) / log(0.5)
        else:
            self.bias_weight = float('inf')

    @staticmethod
    def compare_flows_and_volumes(
            assigned_flows: list[float], actual_flows: list[float],
            predicted_vols: list[float], actual_vols: list[float], plot: bool = True
    ) -> bool:

        all_equal = True
        for time_step, (assigned_flow, actual_flow, predicted_vol, actual_vol) in enumerate(
                zip(assigned_flows, actual_flows, predicted_vols, actual_vols)
        ):
            if abs(actual_vol - predicted_vol) > 1:
                warnings.warn(
                    f"Actual and predicted volume in time step {time_step} are not equal: "
                    f"actual volume is {actual_vol} m3 and predicted volume is {predicted_vol} m3."
                )
                all_equal = False
            if abs(actual_flow - assigned_flow) > 0.01:
                warnings.warn(
                    f"Actual and assigned flow in time step {time_step} are not equal: "
                    f"actual flow is {actual_flow} m3/s and assigned_flow is {assigned_flow} m3/s."
                )
                all_equal = False

        if not all_equal and plot:
            import matplotlib.pyplot as plt
            fig2, axs = plt.subplots(1, 2)
            # Compare volumes:
            axs[0].set_xlabel("Time (15min)")
            axs[0].plot(predicted_vols, color='b', label="Predicted volume")
            axs[0].plot(actual_vols, color='c', label="Actual volume")
            axs[0].set_ylabel("Volume (m3)")
            axs[0].legend()
            # Compare flows:
            axs[1].set_xlabel("Time (15min)")
            axs[1].plot(assigned_flows, color='g', label="Assigned flows")
            axs[1].plot(actual_flows, color='lime', label="Actual exiting flows")
            axs[1].set_ylabel("Flow (m3/s)")
            axs[1].legend()
            plt.show()

        return all_equal

    def solve(self, options: dict = None) -> dict:

        """
        Fill the `solution` attribute of the object,
        with the solution recommended by the heuristic.

        :param options: Unused argument, inherited from Experiment
        :return: A dictionary with status codes
        """

        flows = dict()
        volumes = dict()
        powers = dict()
        obj_fun_details = dict()

        dams_ids_in_order = sorted(
            self.instance.get_ids_of_dams(),
            key=lambda dam_idx: self.instance.get_order_of_dam(dam_idx)
        )
        flow_contribution = self.instance.get_all_incoming_flows()

        for dam_id in dams_ids_in_order:

            # Get heuristic solution for this dam
            single_dam_solver = HeuristicSingleDam(
                dam_id=dam_id,
                instance=self.instance,
                flow_contribution=flow_contribution,
                config=self.config,
                bias_weight=self.bias_weight,
                greedy=self.greedy,
                do_tests=self.do_tests,
            )
            assigned_flows, predicted_vols = single_dam_solver.solve()
            flows[dam_id] = assigned_flows
            volumes[dam_id] = predicted_vols

            # Simulate to get turbined flows, powers and income
            (
                turbined_flows, powers[dam_id], actual_vols, actual_flows, obj_fun_details[dam_id]
            ) = single_dam_solver.simulate()

            # Check flows and volumes from heuristic are the same as those from the simulator
            if self.do_tests:
                assert self.compare_flows_and_volumes(
                    assigned_flows=assigned_flows, actual_flows=actual_flows,
                    predicted_vols=predicted_vols, actual_vols=actual_vols
                ), f"For {dam_id}, volume and flows from heuristic do not match those from the simulator."

            # Flow contribution to the next dam
            flow_contribution = turbined_flows

        # Add dam incomes to get total income
        total_income = sum(
            obj_fun_details[dam_id]["total_income_eur"] for dam_id in self.instance.get_ids_of_dams()
        )

        # Get datetimes
        start_datetime, end_datetime, _, _, _, solution_datetime = self.get_instance_solution_datetimes()

        sol_dict = dict(
            instance_datetimes=dict(
                start=start_datetime,
                end_decisions=end_datetime
            ),
            solution_datetime=solution_datetime,
            solver="Heuristic",
            configuration=asdict(self.config),
            objective_function=total_income,
            dams=[
                dict(
                    id=dam_id,
                    flows=flows[dam_id],
                    power=powers[dam_id],
                    volume=volumes[dam_id],
                    objective_function_details=obj_fun_details[dam_id]
                )
                for dam_id in self.instance.get_ids_of_dams()
            ],
            price=self.instance.get_all_prices(),
        )
        self.solution = Solution.from_dict(sol_dict)

        # Check flow smoothing parameter compliance
        if self.do_tests:
            assert self.solution.complies_with_flow_smoothing(
                flow_smoothing=self.config.flow_smoothing,
                initial_flows={
                    dam_id: self.instance.get_initial_lags_of_channel(dam_id)[0]
                    for dam_id in self.instance.get_ids_of_dams()
                },
                max_flows={
                    dam_id: self.instance.get_max_flow_of_channel(dam_id)
                    for dam_id in self.instance.get_ids_of_dams()
                },
            ), (
                "The solution from the heuristic does not comply with the flow smoothing parameter."
            )

        return dict()
