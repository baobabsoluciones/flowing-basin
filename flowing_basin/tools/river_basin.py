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
        flow_smoothing: int = 0,
        num_scenarios: int = 1,
    ):

        # Number of scenarios (e.g. candidate solutions) for which to do calculations at the same time
        self.num_scenarios = num_scenarios

        self.instance = instance

        # Number of periods in which we force to keep the direction (flow increasing OR decreasing) in each dam
        self.flow_smoothing = flow_smoothing

        # Dams inside the flowing basin
        self.dams = []
        for dam_id in self.instance.get_ids_of_dams():
            dam = Dam(
                idx=dam_id,
                instance=self.instance,
                paths_power_models=paths_power_models,
                flow_smoothing=self.flow_smoothing,
                num_scenarios=self.num_scenarios,
            )
            self.dams.append(dam)

        # Identifier of the time step (increases with each update)
        self.time = -1

        # Initialize accumulated income obtained with the generated energy throughout all time steps
        self.accumulated_income = np.zeros(self.num_scenarios)

        # Initialize the record of flows exiting the dams with an empty array of the correct shape
        self.all_flows = np.array([]).reshape((0, self.instance.get_num_dams(), self.num_scenarios))

        # Create data frame that will contain all states of the river basin throughout time
        self.history = self.create_history()

    def reset(self, instance: Instance = None, flow_smoothing: int = None, num_scenarios: int = None):

        """
        Resets the river basin.
        This method resets the instance, the flow smoothing parameter, and the number of scenarios (if given),
        and all attributes that represent time-dependent (non-constant) values.
        """

        if instance is not None:
            self.instance = instance
        if flow_smoothing is not None:
            self.flow_smoothing = flow_smoothing
        if num_scenarios is not None:
            self.num_scenarios = num_scenarios

        self.time = -1
        self.accumulated_income = 0
        self.all_flows = np.array([]).reshape((0, self.instance.get_num_dams(), self.num_scenarios))
        self.history = self.create_history()

        for dam in self.dams:
            dam.reset(instance=self.instance, flow_smoothing=self.flow_smoothing, num_scenarios=self.num_scenarios)

        return

    def create_history(self) -> pd.DataFrame:

        """
        Create head for the data frame we will be concatenating rows.
        """

        column_list = ["scenario", "time", "incoming"]
        for dam_id in self.instance.get_ids_of_dams():
            column_list += [
                f"{dam_id}_unreg",
                f"{dam_id}_flow_assigned",
                f"{dam_id}_flow_smoothed",
                f"{dam_id}_flow_clipped1",
                f"{dam_id}_flow_clipped2",
                f"{dam_id}_netflow",
                f"{dam_id}_volchange",
                f"{dam_id}_vol",
                f"{dam_id}_power",
                f"{dam_id}_turbined"
            ]
        column_list += ["price", "income"]

        df = pd.DataFrame(columns=column_list)

        return df

    def update_history(self):

        """
        Add a row to the data frame 'history' attribute
        with values from the current state of the river basin.
        """

        new_row = dict()
        income = self.get_income()
        for i in range(self.num_scenarios):
            new_row.update({"scenario": i, "time": self.time, "incoming": self.instance.get_incoming_flow(self.time)})
            for dam in self.dams:
                net_flow = (
                    dam.flow_contribution[i] + dam.unregulated_flow - dam.flow_out_clipped2[i]
                )
                new_row.update(
                    {
                        f"{dam.idx}_unreg": dam.unregulated_flow,
                        f"{dam.idx}_flow_assigned": dam.flow_out_assigned[i],
                        f"{dam.idx}_flow_smoothed": dam.flow_out_smoothed[i],
                        f"{dam.idx}_flow_clipped1": dam.flow_out_clipped1[i],
                        f"{dam.idx}_flow_clipped2": dam.flow_out_clipped2[i],
                        f"{dam.idx}_netflow": net_flow,
                        f"{dam.idx}_volchange": net_flow * self.instance.get_time_step(),
                        f"{dam.idx}_vol": dam.volume[i],
                        f"{dam.idx}_power": dam.channel.power_group.power[i],
                        f"{dam.idx}_turbined": dam.channel.power_group.turbined_flow[i],
                    }
                )
            new_row.update({"price": self.instance.get_price(self.time), "income": income[i]})

        self.history = pd.concat(
            [self.history, pd.DataFrame([new_row])],
            ignore_index=True,
        )

        return

    def plot_history(self) -> plt.Axes:

        """
        Plot the series saved in history.
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
            axs[1].plot(self.history[f"{dam_id}_flow_clipped2"], label=f"{dam_id}_flow")
            axs[2].plot(self.history[f"{dam_id}_power"], label=f"{dam_id}_power")
        twinax.plot(self.history["price"], color="green", label="price")

        # Add legends
        for i in range(3):
            axs[i].legend()
        twinax.legend()

        return axs

    def get_income(self) -> np.ndarray:

        """
        Get the income that is obtained with the power generated in this time step.

        :return:
             Array of size num_scenarios with
             the income obtained in this time step with the current generated power in every scenario (€)
        """

        price = self.instance.get_price(self.time)
        power = np.array([dam.channel.power_group.power for dam in self.dams]).sum(axis=0)
        time_step_hours = self.instance.get_time_step() / 3600
        income = price * power * time_step_hours

        return income

    def get_clipped_flows(self) -> np.ndarray:

        """
        Get the flows that are actually going out of the dams,
        which may be lower than the assigned flows because of channel flow limits and dam volume limits.

        :return:
            Array of shape num_dams x num_scenarios with
            the flows clipped because of the flow limits and minimum volumes (m3/s)
        """

        return np.array([dam.flow_out_clipped2 for dam in self.dams])

    def update(self, flows: np.ndarray):

        """
        Update the river basin for a single time step.

        :param flows:
            Array of shape num_dams x num_scenarios with
            the flows going through each channel for every scenario in the current time step (m3/s)
        """

        # Increase time step identifier (which will be used to get the next price, incoming flow, and unregulated flows)
        # This identifier was initialized as -1, and will go from 0 to num_time_steps - 1
        self.time = self.time + 1

        # Check instance is not finished already
        assert self.time < self.instance.get_num_time_steps(), (
            "The final time horizon has already been reached. "
            "You should reset the environment before doing another update."
        )

        # Check input is of the correct shape
        assert flows.shape == (
            self.instance.get_num_dams(),
            self.num_scenarios,
        ), f"{flows.shape=} should actually be {(self.instance.get_num_dams(), self.num_scenarios)=}"

        # Incoming flow to the first dam
        incoming_flow = self.instance.get_incoming_flow(self.time)

        # The first dam has no preceding dam
        turbined_flow_of_preceding_dam = np.zeros(self.num_scenarios)

        for dam_index, dam in enumerate(self.dams):

            # Update dam with the flow we take from it, and the incoming and/or unregulated flow it receives
            unregulated_flow = self.instance.get_unregulated_flow_of_dam(
                self.time, dam.idx
            )
            turbined_flow = dam.update(
                flow_out=flows[dam_index],
                incoming_flow=incoming_flow,
                unregulated_flow=unregulated_flow,
                turbined_flow_of_preceding_dam=turbined_flow_of_preceding_dam,
            )
            turbined_flow_of_preceding_dam = turbined_flow

        self.accumulated_income += self.get_income()
        self.all_flows = np.concatenate([self.all_flows, self.get_clipped_flows().reshape((1, self.instance.get_num_dams(), -1))], axis=0)
        self.update_history()

        return

    def deep_update_flows(self, flows: np.ndarray):

        """
        Update the river basin for the whole planning horizon.

        :param flows:
            Array of shape num_time_steps x num_dams x num_scenarios with
            the flows that should go through each channel in every time step for every scenario (m3/s)
        """

        for flow in flows:
            self.update(flow)

        return

    def deep_update_relvars(self, relvars: np.ndarray):

        """

        :param relvars: Relative variations
            Array of shape num_time_steps x num_dams x num_scenarios with
            the variation of flow (as a fraction of flow max) through each channel in every time step and scenario (m3/s)
        """

        # Max flow through each channel, as an array of shape num_dams x num_scenarios
        max_flows = np.repeat(
            [
                self.instance.get_max_flow_of_channel(dam_id)
                for dam_id in self.instance.get_ids_of_dams()
            ],
            self.num_scenarios,
        ).reshape((self.instance.get_num_dams(), self.num_scenarios))

        # Initialize old flows
        # Flows that went through the channels in the previous time step, as an array of shape num_dams x num_scenarios
        old_flows = np.repeat(
            [
                self.instance.get_initial_lags_of_channel(dam_id)[0]
                for dam_id in self.instance.get_ids_of_dams()
            ],
            self.num_scenarios,
        ).reshape((self.instance.get_num_dams(), self.num_scenarios))

        # Update river basin repeatedly
        for relvar in relvars:
            new_flows = old_flows + relvar * max_flows
            self.update(new_flows)
            old_flows = self.get_clipped_flows()

        return

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
            "next_incoming_flow": self.instance.get_incoming_flow(self.time + 1),
            "next_price": self.instance.get_price(self.time + 1),
        }
        for dam in self.dams:
            state[dam.idx] = {
                "vol": dam.volume,
                "flow_limit": dam.channel.flow_limit,
                "next_unregulated_flow": self.instance.get_unregulated_flow_of_dam(
                    self.time + 1, dam.idx
                ),
                "lags": dam.channel.past_flows,
                "power": dam.channel.power_group.power,
                "turbined_flow": dam.channel.power_group.turbined_flow,
            }

        return state
