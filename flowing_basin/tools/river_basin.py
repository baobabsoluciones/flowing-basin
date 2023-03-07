from flowing_basin.core import Instance
from .dam import Dam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

        # Create history and log
        self.history = self.create_history()
        self.log = self.create_log()

    def reset(self, instance: Instance = None, num_scenarios: int = None):

        """
        Resets the river basin
        This method resets the instance and the number of scenarios (if given)
        and all attributes that represent time-dependent (non-constant) values
        """

        if instance is not None:
            self.instance = instance
        if num_scenarios is not None:
            self.num_scenarios = num_scenarios
        self.time = 0
        self.history = self.create_history()
        self.log = self.create_log()

        for dam in self.dams:
            dam.reset(instance=self.instance, num_scenarios=self.num_scenarios)

    def create_history(self) -> pd.DataFrame:

        """
        Create head for the data frame we will be concatenating rows
        This data frame will only be filled if there is a single scenario
        """

        column_list = []
        for dam_id in self.instance.get_ids_of_dams():
            column_list += [f"{dam_id}_vol", f"{dam_id}_flow", f"{dam_id}_power"]
        column_list += ["price"]

        df = pd.DataFrame(columns=column_list)

        return df

    def update_history(self):

        """
        Add a row to the data frame 'history' attribute
        with values from the current state of the river basin
        """

        # The history is only updated when a single scenario is considered
        if self.num_scenarios != 1:
            return

        new_row = dict()
        for dam_index, dam in enumerate(self.dams):
            new_row.update(
                {
                    f"{dam.idx}_vol": dam.volume,
                    f"{dam.idx}_flow": dam.flow_out_clipped_vol,
                    f"{dam.idx}_power": dam.channel.power_group.power,
                }
            )
        new_row.update({"price": self.instance.get_price(self.time)})
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])],
            ignore_index=True,
        )

    def plot_history(self, show: bool = True, path: str = None):

        """
        Plot the series saved in history
        :param show: Show plot
        :param path: Save plot in given path
        """

        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(18.5, 10.5)

        # Add X labels
        for i in range(3):
            axs[i].set_xlabel("Time (15min)")

        # Add Y labels
        axs[0].set_ylabel("Volume (m3)")
        axs[1].set_ylabel("Flow (m3/s)")
        axs[2].set_ylabel("Power (MW)")
        twinax = axs[2].twinx()
        twinax.set_ylabel("Price (€/MWh)")

        # Plot series
        for dam_id in self.instance.get_ids_of_dams():
            axs[0].plot(self.history[f"{dam_id}_vol"], label=f"{dam_id}_vol")
            axs[1].plot(self.history[f"{dam_id}_flow"], label=f"{dam_id}_flow")
            axs[2].plot(self.history[f"{dam_id}_power"], label=f"{dam_id}_power")
        twinax.plot(self.history["price"], color="green", label="price")

        # Add legends
        for i in range(3):
            axs[i].legend()
        twinax.legend()

        # Failed attempt using Pandas' plot method:
        # subplots = [
        #     [f"{dam_id}_vol" for dam_id in self.instance.get_ids_of_dams()],
        #     [f"{dam_id}_flow" for dam_id in self.instance.get_ids_of_dams()],
        #     ["price", *[f"{dam_id}_power" for dam_id in self.instance.get_ids_of_dams()]]
        # ]
        # self.history.plot(ax=axs, subplots=subplots, secondary_y=["price"])

        if path is not None:
            plt.savefig(path)

        # This instruction must be AFTER we save the plot, otherwise nothing will be saved
        if show:
            plt.show()

        plt.close()

    def create_log(self) -> str:

        """
        Create head for the table-like string in which we will be putting values
        This log will only be filled if there is a single scenario
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

    def update_log(self):

        """
        Add a row to the table-like string 'log' attribute
        with values from the current state of the river basin
        """

        # The log is only updated when a single scenario is considered
        if self.num_scenarios != 1:
            return

        # Add current time and incoming flow to the first dam
        self.log += f"\n{self.time: ^6}{round(self.instance.get_incoming_flow(self.time), 2): ^13}"

        # Add dam information
        for dam in self.dams:
            net_flow = dam.flow_contribution + dam.unregulated_flow - dam.flow_out_clipped_vol
            self.log += (
                f"{round(dam.unregulated_flow, 4): ^13}{round(dam.flow_out, 4): ^13}"
                f"{round(dam.flow_out_clipped_channel, 4): ^14}{round(dam.flow_out_clipped_vol, 4): ^14}"
                f"{round(net_flow, 4): ^14}{round(net_flow * self.instance.get_time_step(), 5): ^15}"
                f"{round(dam.volume, 2): ^13}{round(dam.channel.power_group.power, 2): ^13}"
                f"|\t"
                f"{round(dam.channel.power_group.turbined_flow, 5): ^15}"
            )

        # Add current price and income values
        self.log += f"{round(self.instance.get_price(self.time), 2): ^13}{round(self.calculate_income(), 2): ^13}"

    def calculate_income(self) -> float | np.ndarray:

        """

        :return:
         - Income obtained in this time step with the current power generated by the power groups (€)
         - OR Array of size num_scenarios with the income obtained in every scenario (€)
        """

        price = self.instance.get_price(self.time)
        power = sum(dam.channel.power_group.power for dam in self.dams)
        time_step_hours = self.instance.get_time_step() / 3600
        income = price * power * time_step_hours

        return income

    def update(
        self, flows: list[float] | np.ndarray, return_clipped_flows: bool = False
    ) -> (float | np.ndarray) | (tuple[float, float] | tuple[np.ndarray, np.ndarray]):

        """

        :param flows:
         - List of flows that should go through each channel in the current time step (m3/s)
         - OR Array of shape num_dams x num_scenarios with these flows for every scenario (m3/s)
        :param return_clipped_flows:
           If true, return the flows clipped because of the flow limits and minimum volumes
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

        # Given flows may be clipped because of flow and volume limits
        clipped_flows = np.copy(flows)
        if self.num_scenarios == 1:
            clipped_flows = clipped_flows.tolist()

        # Incoming flow to the first dam
        incoming_flow = self.instance.get_incoming_flow(self.time)

        # The first dam has no preceding dam
        turbined_flow_of_preceding_dam = 0

        for dam_index, dam in enumerate(self.dams):

            # Update dam with the flow we take from it, and the incoming and/or unregulated flow it receives
            unregulated_flow = self.instance.get_unregulated_flow_of_dam(
                self.time, dam.idx
            )
            turbined_flow, flow_out_clipped = dam.update(
                flow_out=flows[dam_index],
                incoming_flow=incoming_flow,
                unregulated_flow=unregulated_flow,
                turbined_flow_of_preceding_dam=turbined_flow_of_preceding_dam,
            )
            clipped_flows[dam_index] = flow_out_clipped

            turbined_flow_of_preceding_dam = turbined_flow

        # Calculate income obtained with the new power values
        income = self.calculate_income()

        # Update log and history
        self.update_log()
        self.update_history()

        # Increase time step identifier to get the next price, incoming flow, and unregulated flows
        self.time = self.time + 1

        if return_clipped_flows:
            return income, clipped_flows
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

    def deep_update_flows(self, flows: list[list[float]] | np.ndarray) -> float | np.ndarray:

        """

        :param flows:
         - Lists of lists with the flows that should go through each channel in every time step (m3/s)
         - OR Array of shape num_time_steps x num_dams x num_scenarios with these flows for every scenario (m3/s)
        :return:
         - Accumulated income obtained with the indicated flows in all time steps (€)
         - OR Array of size num_scenarios with the accumulated income obtained in every scenario (€)
        """

        self.sanitize_input(flows)

        income = 0
        for flow in flows:
            income += self.update(flow)

        return income

    def deep_update_relvars(
        self,
        relvars: list[list[float]] | np.ndarray,
        keep_direction: int = 0,
        return_equivalent_flows: bool = False,
    ) -> (float | np.ndarray) | (
        tuple[float, list[list[float]]] | tuple[np.ndarray, np.ndarray]
    ):

        """

        :param relvars: Relative variations
         - Lists of lists with the variation of flow (as a fraction of flow max) through each channel in every time step (m3/s)
         - OR Array of shape num_time_steps x num_dams x num_scenarios with these relative variations for every scenario (m3/s)
        :param keep_direction:
           Number of periods in which we force to keep the direction (flow increasing OR decreasing) in each dam
           Note that this rule may not apply if flows must be clipped
        :param return_equivalent_flows:
           If True, returns the flows equivalent to the given relative variations
        :return:
         - Accumulated income obtained with the indicated relative variations in all time steps (€)
         - OR Array of size num_scenarios with the accumulated income obtained in every scenario (€)
        """

        self.sanitize_input(relvars)

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

        # Equivalent flows to the given relvars
        equivalent_flows = []

        # Update river basin repeatedly ---- #

        # Initialize income
        income = 0

        # Initialize old relative variations,
        # which is used to know when the direction of variations changes from one period to the next
        old_relvars = np.zeros((1, self.instance.get_num_dams(), self.num_scenarios))
        if self.num_scenarios == 1:
            # Remove last dimension of array and turn it into list of lists
            old_relvars = old_relvars[:, :, 0].tolist()

        # Create copy of relvars to avoid modifying it
        # This also turns relvars into an array if it isn't already, which is convenient
        relvars = np.copy(relvars)

        for relvar in relvars:

            if keep_direction > 0:
                # Prevent change of direction among the last N = keep_distance periods:
                # Check all elements of the current relvar have the same sign as the last N relvars
                # Otherwise, set these elements to 0
                sign_changes_each_period = old_relvars[-keep_direction:] * relvar < 0
                sign_changes_any_period = np.sum(sign_changes_each_period, axis=0) > 0
                relvar[sign_changes_any_period] = 0

            # Calculate new flows by adding the current variation
            new_flows = old_flows + relvar * max_flows
            if self.num_scenarios == 1:
                new_flows = new_flows.tolist()

            # Update river basin with the new flows and get income
            new_income, clipped_flows = self.update(
                new_flows, return_clipped_flows=True
            )
            income += new_income
            equivalent_flows.append(clipped_flows)

            old_flows = clipped_flows
            old_relvars = np.append(old_relvars, [relvar], axis=0)

        if self.num_scenarios > 1:
            # Turn list into array
            equivalent_flows = np.array(equivalent_flows)

        if return_equivalent_flows:
            return income, equivalent_flows
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
