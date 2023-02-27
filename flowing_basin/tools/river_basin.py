from flowing_basin.core import Instance
from .dam import Dam
import numpy as np


class RiverBasin:

    """
    Class representing the river basin
    """

    def __init__(
        self,
        instance: Instance,
        paths_power_models: dict[str, str],
        num_scenarios: int = 1,
    ):

        # Number of scenarios (e.g. candidate solutions) for which to do calculations at the same time
        self.num_scenarios = num_scenarios

        self.instance = instance

        # Dams inside the flowing basin
        self.dams = []
        for dam_id in self.instance.get_ids_of_dams():
            dam = Dam(
                idx=dam_id,
                instance=self.instance,
                paths_power_models=paths_power_models,
                num_scenarios=self.num_scenarios,
            )
            self.dams.append(dam)

        # Identifier of the time step (increases with each update)
        self.time = 0

        # Create log
        self.log = self.create_log()

    def reset(self, instance: Instance = None):

        """
        Resets the river basin
        This method resets the instance (if given)
        and all attributes that represent time-dependent (non-constant) values
        """

        self.time = 0
        self.log = self.create_log()
        if instance is not None:
            self.instance = instance

        for dam in self.dams:
            dam.reset(self.instance)

    def create_log(self) -> str:

        """
        Create head for the table-like string in which we will be putting values
        """

        log = f"{'time': ^6}{'incoming': ^13}"
        log += "".join(
            [
                (
                    f"{f'{dam_id}_unreg': ^13}{f'{dam_id}_flow': ^13}"
                    f"{f'{dam_id}_clipped1': ^14}{f'{dam_id}_clipped2': ^14}"
                    f"{f'{dam_id}_netflow': ^14}{f'{dam_id}_volchange': ^15}{f'{dam_id}_vol': ^13}"
                    f"{f'{dam_id}_power': ^13}"
                    f"|\t"
                    f"{f'{dam_id}_turbined': ^15}"
                )
                for dam_id in self.instance.get_ids_of_dams()
            ]
        )
        log += f"{'price': ^13}{'income': ^13}"

        return log

    def update(self, flows: list[float] | np.ndarray) -> float | np.ndarray:

        """

        :param flows:
         - List of flows that should go through each channel in the current time step (m3/s)
         - OR Array of shape num_dams x num_scenarios with these flows for every scenario (m3/s)
        :return:
         - Income obtained with the indicated flows in this time step (€)
         - OR Array of size num_scenarios with the income obtained in every scenario (€)
        """

        if isinstance(flows, list):
            assert (
                len(flows) == self.instance.get_num_dams()
            ), f"{len(flows)=} should actually be {self.instance.get_num_dams()=}"
        if isinstance(flows, np.ndarray):
            assert flows.shape == (
                self.instance.get_num_dams(),
                self.num_scenarios,
            ), f"{flows.shape=} should actually be {(self.instance.get_num_dams(), self.num_scenarios)=}"

        # Get incoming flow
        incoming_flow = self.instance.get_incoming_flow(self.time)
        if self.num_scenarios == 1:
            self.log += f"\n{self.time: ^6}{round(incoming_flow, 2): ^13}"

        # Clip flows according to the flow limits of the channels
        clipped_flows = np.copy(flows)
        if self.num_scenarios == 1:
            clipped_flows = clipped_flows.tolist()
        for dam_index, dam in enumerate(self.dams):
            clipped_flows[dam_index] = np.clip(
                flows[dam_index], 0, dam.channel.flow_limit
            )

        # Update dams

        # The first dam has no preceding dam
        turbined_flow_of_preceding_dam = 0

        for dam_index, dam in enumerate(self.dams):

            flow_out = clipped_flows[dam_index]
            unregulated_flow = self.instance.get_unregulated_flow_of_dam(
                self.time, dam.idx
            )
            turbined_flow, flow_out_clipped = dam.update(
                flow_out=flow_out,
                incoming_flow=incoming_flow,
                unregulated_flow=unregulated_flow,
                turbined_flow_of_preceding_dam=turbined_flow_of_preceding_dam,
            )

            if self.num_scenarios == 1:
                net_flow = (
                    incoming_flow + unregulated_flow - flow_out
                    if dam.order == 1
                    else turbined_flow_of_preceding_dam + unregulated_flow - flow_out
                )
                self.log += (
                    f"{round(unregulated_flow, 4): ^13}{round(flows[dam_index], 4): ^13}"
                    f"{round(clipped_flows[dam_index], 4): ^14}{round(flow_out_clipped, 4): ^14}"
                    f"{round(net_flow, 4): ^14}{round(net_flow * self.instance.get_time_step(), 5): ^15}"
                    f"{round(dam.volume, 2): ^13}{round(dam.channel.power_group.power, 2): ^13}"
                    f"|\t"
                    f"{round(turbined_flow, 5): ^15}"
                )

            turbined_flow_of_preceding_dam = turbined_flow

        # Calculate income
        price = self.instance.get_price(self.time)
        power = sum(dam.channel.power_group.power for dam in self.dams)
        time_step_hours = self.instance.get_time_step() / 3600
        income = price * power * time_step_hours
        if self.num_scenarios == 1:
            self.log += f"{round(price, 2): ^13}{round(income, 2): ^13}"

        # Increase time step identifier to get the next price, incoming flow, and unregulated flows
        self.time = self.time + 1

        return income

    def sanitize_input(self, input_all_periods: list[list[float]] | np.ndarray) -> None:

        """
        Make sure the input is of the correct size for the deep update methods
        """

        if isinstance(input_all_periods, list):
            assert (
                len(input_all_periods) <= self.instance.get_num_time_steps()
            ), f"{len(input_all_periods)=} should be lower than {self.instance.get_num_time_steps()=}"
        if isinstance(input_all_periods, np.ndarray):
            assert (
                input_all_periods.shape[0] <= self.instance.get_num_time_steps()
            ), f"{input_all_periods.shape[0]=} should be lower than {self.instance.get_num_time_steps()=}"

    def deep_update(self, flows_all_periods: list[list[float]] | np.ndarray) -> float:

        """

        :param flows_all_periods:
         - Lists of lists with the flows that should go through each channel in every time step (m3/s)
         - OR Array of shape num_time_steps x num_dams x num_scenarios with these flows for every scenario (m3/s)
        :return:
         - Accumulated income obtained with the indicated flows in all time steps (€)
         - OR Array of size num_scenarios with the accumulated income obtained in every scenario (€)
        """

        self.sanitize_input(flows_all_periods)

        income = 0
        for flows in flows_all_periods:
            income += self.update(flows)

        return income

    def deep_update_relative_variations(
        self, rel_vars_all_periods: list[list[float]] | np.ndarray
    ) -> float:

        """

        :param rel_vars_all_periods:
         - Lists of lists with the variation of flow (as a fraction of flow max) through each channel in every time step (m3/s)
         - OR Array of shape num_time_steps x num_dams x num_scenarios with these relative variations for every scenario (m3/s)
        :return:
         - Accumulated income obtained with the indicated relative variations in all time steps (€)
         - OR Array of size num_scenarios with the accumulated income obtained in every scenario (€)
        """

        self.sanitize_input(rel_vars_all_periods)

        # Max flow through each channel
        max_flows = np.array(
            [
                self.instance.get_max_flow_of_channel(dam_id)
                for dam_id in self.instance.get_ids_of_dams()
            ]
        )
        if self.num_scenarios > 1:
            # Turn array of size num_dams into array of shape num_dams x num_scenarios
            max_flows = np.repeat(max_flows, self.num_scenarios).reshape(
                (self.instance.get_num_dams(), self.num_scenarios)
            )

        # Flows that went through the channels in the previous time step
        old_flows = [
            self.instance.get_initial_lags_of_channel(dam_id)[0]
            for dam_id in self.instance.get_ids_of_dams()
        ]
        if self.num_scenarios > 1:
            # Turn array of size num_dams into array of shape num_dams x num_scenarios
            old_flows = np.repeat(old_flows, self.num_scenarios).reshape(
                (self.instance.get_num_dams(), self.num_scenarios)
            )

        print(f"{max_flows=}")
        print(self.time, {dam.idx: dam.channel.past_flows for dam in self.dams})

        # Update river basin repeatedly
        income = 0
        for rel_vars in rel_vars_all_periods:

            new_flows = old_flows + rel_vars * max_flows
            if self.num_scenarios == 1:
                new_flows = new_flows.tolist()

            income += self.update(new_flows)
            old_flows = new_flows

            print(self.time, new_flows)

        return income

    def get_state(self) -> dict:

        """
        Returns the state of the river basin
        :return: Dictionary with:
         - the incoming flow (m3)
         - the price (EUR/MWh)
         - volume of each dam (m3)
         - flow limit of each dam (m3/s)
         - unregulated flow of each dam (m3/s)
         - current and past flows through each channel (m3/s)
         - power generated by each power group (MW)
         - turbined flow exiting each power group (m3/s)
        """

        state = {
            "next_incoming_flow": self.instance.get_incoming_flow(self.time),
            "next_price": self.instance.get_price(self.time),
        }
        for dam in self.dams:
            state[dam.idx] = {
                "vol": dam.volume,
                "flow_limit": dam.channel.flow_limit,
                "next_unregulated_flow": self.instance.get_unregulated_flow_of_dam(
                    self.time, dam.idx
                ),
                "lags": dam.channel.past_flows,
                "power": dam.channel.power_group.power,
                "turbined_flow": dam.channel.power_group.turbined_flow,
            }

        return state
