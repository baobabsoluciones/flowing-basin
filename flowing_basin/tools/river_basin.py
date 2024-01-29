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
        flow_smoothing: int = 0,
        num_scenarios: int = 1,
        mode: str = "nonlinear",
        paths_power_models: dict[str, str] = None,
        do_history_updates: bool = True,
    ):

        # Number of scenarios (e.g. candidate solutions) for which to do calculations at the same time
        self.num_scenarios = num_scenarios

        self.instance = instance

        # Whether to update history (slow) or not (fast)
        self.do_history_updates = do_history_updates

        # Number of periods in which we force to keep the direction (flow increasing OR decreasing) in each dam
        self.flow_smoothing = flow_smoothing

        # Dams inside the flowing basin
        self.dams = [
            Dam(
                idx=dam_id,
                instance=self.instance,
                paths_power_models=paths_power_models,
                flow_smoothing=self.flow_smoothing,
                num_scenarios=self.num_scenarios,
                mode=mode,
            )
            for dam_id in self.instance.get_ids_of_dams()
        ]

        # Time-dependent attributes
        self.info_offset = None
        self.time = None
        self.all_past_flows = None
        self.all_past_clipped_flows = None
        self.all_past_volumes = None
        self.all_past_powers = None
        self.all_past_turbined = None
        self.all_past_groups = None
        self.history = None

        # Initialize the time-dependent attributes (variables)
        self._reset_variables()

        # Update river basin until the start of decisions
        if self.info_offset > 0:  # noqa
            self.update_until_decisions_start()

    def _reset_variables(self):

        """
        Reset all attributes that represent time-dependent (non-constant) values.
        """

        # Identifier of the time step (increases with each update)
        self.info_offset = self.instance.get_start_information_offset()
        self.time = -1 - self.info_offset

        # Record of flows exiting the dams,
        # initialized as an empty array of the correct shape (num_time_steps x num_dams x num_scenarios)
        self.all_past_flows = np.array([]).reshape(
            (0, self.instance.get_num_dams(), self.num_scenarios)
        )
        self.all_past_clipped_flows = np.array([]).reshape(
            (0, self.instance.get_num_dams(), self.num_scenarios)
        )

        # Record of volumes, powers, turbined flows and power group numbers of each dam,
        # initialized as empty arrays of the correct shape (num_time_steps x num_scenarios)
        self.all_past_volumes = {
            dam_id: np.array([]).reshape(
                (0, self.num_scenarios)
            )
            for dam_id in self.instance.get_ids_of_dams()
        }
        self.all_past_powers = {
            dam_id: np.array([]).reshape(
                (0, self.num_scenarios)
            )
            for dam_id in self.instance.get_ids_of_dams()
        }
        self.all_past_turbined = {
            dam_id: np.array([]).reshape(
                (0, self.num_scenarios)
            )
            for dam_id in self.instance.get_ids_of_dams()
        }
        self.all_past_groups = {
            dam_id: np.array([]).reshape(
                (0, self.num_scenarios)
            )
            for dam_id in self.instance.get_ids_of_dams()
        }

        # Data frame that will contain all states of the river basin throughout time
        self.history = self.create_history()

        return

    def reset(
        self,
        instance: Instance = None,
        flow_smoothing: int = None,
        num_scenarios: int = None,
        greedy_start: bool = False
    ):

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

        self._reset_variables()

        for dam in self.dams:
            dam.reset(
                instance=self.instance,
                flow_smoothing=self.flow_smoothing,
                num_scenarios=self.num_scenarios,
            )

        # Update river basin until the start of decisions
        if self.info_offset > 0:
            self.update_until_decisions_start(greedy_start)

        return

    def update_until_decisions_start(self, greedy: bool = False):

        """
        Update the river basin with the starting flows
        until decisions must start

        :param greedy: Use greedy flows (maximum flows) instead of starting flows
        """

        if greedy:
            starting_flows = [
                [
                    [
                        self.instance.get_max_flow_of_channel(dam_id)
                        for _ in range(self.instance.get_start_information_offset())
                    ] for dam_id in self.instance.get_ids_of_dams()
                ] for _ in range(self.num_scenarios)
            ]
        else:
            starting_flows = [
                [
                    self.instance.get_starting_flows(dam_id) for dam_id in self.instance.get_ids_of_dams()
                ] for _ in range(self.num_scenarios)
            ]

        starting_flows = np.array(starting_flows)  # Array of shape num_scenarios x num_dams x num_steps
        starting_flows = np.transpose(starting_flows)  # Array of shape num_steps x num_dams x num_scenarios
        self.deep_update_flows(starting_flows)
        assert self.time == -1

    def create_history(self) -> pd.DataFrame:

        """
        Create head for the data frame we will be concatenating rows.
        """

        column_list = ["scenario", "time", "price", "incoming"]
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
                f"{dam_id}_turbined",
                f"{dam_id}_groups",
                f"{dam_id}_startups",
                f"{dam_id}_limits",
                f"{dam_id}_income"
            ]

        df = pd.DataFrame(columns=column_list)

        return df

    def update_history(self):

        """
        Add a row to the data frame 'history' attribute
        with values from the current state of the river basin.
        """

        for i in range(self.num_scenarios):
            new_row = dict()
            new_row.update(
                {
                    "scenario": i,
                    "time": self.time,
                    "price": self.instance.get_price(self.time + self.info_offset),
                    "incoming": self.instance.get_incoming_flow(self.time + self.info_offset),
                }
            )
            for dam in self.dams:
                net_flow = (
                    dam.flow_contribution[i]
                    + dam.unregulated_flow
                    - dam.flow_out_clipped2[i]
                )
                new_row.update(
                    {
                        f"{dam.idx}_unreg": dam.unregulated_flow,
                        f"{dam.idx}_flow_assigned": dam.flow_out_assigned[i],
                        f"{dam.idx}_flow_smoothed": dam.flow_out_smoothed[i],
                        f"{dam.idx}_flow_clipped1": dam.flow_out_clipped1[i],
                        f"{dam.idx}_flow_clipped2": dam.flow_out_clipped2[i],
                        f"{dam.idx}_netflow": net_flow,
                        f"{dam.idx}_volchange": net_flow * self.instance.get_time_step_seconds(),
                        f"{dam.idx}_vol": dam.volume[i],
                        f"{dam.idx}_power": dam.channel.power_group.power[i],
                        f"{dam.idx}_turbined": dam.channel.power_group.turbined_flow[i],
                        f"{dam.idx}_groups": dam.channel.power_group.num_active_groups[i],
                        f"{dam.idx}_startups": int(dam.channel.power_group.num_startups[i]),
                        f"{dam.idx}_limits": int(dam.channel.power_group.num_times_in_limit[i]),
                        f"{dam.idx}_income": dam.channel.power_group.income[i],
                    }
                )
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
        fig.set_size_inches(10 * 3, 10)

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

        # Add vertical line indicating decision horizon
        decision_horizon = self.instance.get_decision_horizon()
        for ax in axs:
            ax.axvline(x=decision_horizon, color='gray')

        # Add legends
        for ax in axs:
            ax.legend()
        twinax.legend()

        return axs

    def get_income(self) -> np.ndarray:

        """
        Get the income that is obtained with the power generated in this time step.

        :return:
             Array of size num_scenarios with
             the income obtained in this time step with the current generated power in every scenario (€)
        """

        return np.array([dam.channel.power_group.income for dam in self.dams]).sum(axis=0)

    def get_acc_income(self) -> np.ndarray:

        """
        Get the accumulated income obtained with the generated energy
        throughout all time steps so far (up to the impact horizon of each dam).

        :return:
             Array of size num_scenarios with
             the accumulated income obtained in every scenario (€)
        """

        return np.array([dam.channel.power_group.acc_income for dam in self.dams]).sum(axis=0)

    def get_num_startups(self) -> np.ndarray:

        """
        Get the number of startups that happened in all dams in the current time step.

        :return:
             Array of size num_scenarios with
             the current number of startups in every scenario
        """

        return np.array([dam.channel.power_group.num_startups for dam in self.dams]).sum(axis=0)

    def get_acc_num_startups(self) -> np.ndarray:

        """
        Get the accumulated number of startups of all dams
        throughout all time steps so far (up to the decision horizon).

        :return:
             Array of size num_scenarios with
             the total number of startups in every scenario
        """

        return np.array([dam.channel.power_group.acc_num_startups for dam in self.dams]).sum(axis=0)

    def get_num_times_in_limit(self) -> np.ndarray:

        """
        Get the number of dams that have a turbined flow in a limit zone.

        :return:
             Array of size num_scenarios with
             the current number of turbined flows in limit in every scenario
        """

        return np.array([dam.channel.power_group.num_times_in_limit for dam in self.dams]).sum(axis=0)

    def get_acc_num_times_in_limit(self) -> np.ndarray:

        """
        Get the accumulated number of time steps with a turbined flow in a limit zone
        throughout all time steps so far (up to the decision horizon).

        :return:
             Array of size num_scenarios with
             the total number of times in limit zones in every scenario
        """

        return np.array([dam.channel.power_group.acc_num_times_in_limit for dam in self.dams]).sum(axis=0)

    def get_final_volume_of_dams(self) -> dict[str, np.ndarray]:

        """
        Get the volume of each dam at the end of the decision horizon.

        :return:
             Dictionary with (dam_id, final_volume) pairs, where each finaL_volume
             is an array of size num_scenarios with the final volume of the dam in every scenario (m3)
        """

        return {dam.idx: dam.final_volume for dam in self.dams}

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
        self.time += 1

        # Check instance is not finished already
        assert self.time < self.instance.get_largest_impact_horizon(), (
            f"The final time horizon, {self.instance.get_largest_impact_horizon()}, has already been reached. "
            f"You should reset the environment before doing another update."
        )

        # Check input is of the correct shape
        assert flows.shape == (
            self.instance.get_num_dams(),
            self.num_scenarios,
        ), f"{flows.shape=} should actually be {(self.instance.get_num_dams(), self.num_scenarios)=}"

        # The first dam has no preceding dam
        turbined_flow_of_preceding_dam = np.zeros(self.num_scenarios)
        for dam_index, dam in enumerate(self.dams):

            # Update dam with the flow we take from it, and the incoming and/or unregulated flow it receives
            flow_contribution = (
                np.repeat(self.instance.get_incoming_flow(self.time + self.info_offset), self.num_scenarios)
                if dam.order == 1
                else turbined_flow_of_preceding_dam
            )
            turbined_flow = dam.update(
                price=self.instance.get_price(self.time + self.info_offset),
                flow_out=flows[dam_index],
                unregulated_flow=self.instance.get_unregulated_flow_of_dam(
                    self.time + self.info_offset, dam.idx
                ),
                flow_contribution=flow_contribution,
                time=self.time
            )
            turbined_flow_of_preceding_dam = turbined_flow

            # Update all past volumes and powers of dam
            self.all_past_volumes[dam.idx] = np.vstack((self.all_past_volumes[dam.idx], dam.volume))
            self.all_past_powers[dam.idx] = np.vstack((self.all_past_powers[dam.idx], dam.channel.power_group.power))
            self.all_past_turbined[dam.idx] = np.vstack((self.all_past_turbined[dam.idx], dam.channel.power_group.turbined_flow))
            self.all_past_groups[dam.idx] = np.vstack((self.all_past_groups[dam.idx], dam.channel.power_group.num_active_groups))

        # Update all past flows
        self.all_past_flows = np.concatenate(
            [
                self.all_past_flows,
                flows.reshape((1, self.instance.get_num_dams(), -1)),
            ],
            axis=0,
        )
        self.all_past_clipped_flows = np.concatenate(
            [
                self.all_past_clipped_flows,
                self.get_clipped_flows().reshape((1, self.instance.get_num_dams(), -1)),
            ],
            axis=0,
        )

        if self.do_history_updates:
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

    def deep_update(self, flows_or_relvars: np.ndarray, is_relvars: bool):

        """
        Reset the river basin and update it for the whole planning horizon
        with the solutions represented by the given array.

        :param flows_or_relvars:
            Array of shape num_time_steps x num_dams x num_particles with
            the flows or relvars assigned for the whole planning horizon
        :param is_relvars: Whether the given array represents relvars or flows
        """

        num_scenarios = flows_or_relvars.shape[-1]
        self.reset(num_scenarios=num_scenarios)

        if is_relvars:
            self.deep_update_relvars(relvars=flows_or_relvars)
            return

        self.deep_update_flows(flows=flows_or_relvars)
        return

    def get_state(self) -> dict:

        """
        Returns the state of the river basin (DEPRECATED METHOD, USED ONLY IN TESTS).

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
            "next_incoming_flow": self.instance.get_incoming_flow(self.time + self.info_offset + 1),
            "next_price": self.instance.get_price(self.time + self.info_offset + 1),
        }
        for dam in self.dams:
            state[dam.idx] = {
                "vol": dam.volume,
                "flow_limit": dam.channel.flow_limit,
                "next_unregulated_flow": self.instance.get_unregulated_flow_of_dam(
                    self.time + self.info_offset + 1, dam.idx
                ),
                "lags": dam.channel.past_flows,
                "power": dam.channel.power_group.power,
                "turbined_flow": dam.channel.power_group.turbined_flow,
            }

        return state
