from flowing_basin.core import Instance
from .power_group import PowerGroup
import numpy as np


class Channel:
    def __init__(
        self,
        idx: str,
        dam_vol: np.ndarray,
        instance: Instance,
        paths_power_models: dict[str, str],
        num_scenarios: int,
    ):

        self.num_scenarios = num_scenarios

        self.idx = idx
        self.limit_flow_points = instance.get_flow_limit_obs_for_channel(self.idx)
        self.flow_max = instance.get_max_flow_of_channel(self.idx)

        # Time-dependent attributes
        self.past_flows = None
        self.flow_limit = None

        # Initialize the time-dependent attributes (variables)
        self._reset_variables(instance, dam_vol=dam_vol)

        self.power_group = PowerGroup(
            idx=self.idx,
            past_flows=self.past_flows,  # noqa
            instance=instance,
            paths_power_models=paths_power_models,
            num_scenarios=self.num_scenarios,
        )

    def _reset_variables(self, instance: Instance, dam_vol: np.ndarray):

        """
        Reset all time-varying attributes of the channel: past flows and flow limit.
        Flow limit observations are not reset as they are constant.
        """

        # Past flows (m3/s)
        # We save them as an array of shape num_scenarios x num_lags
        self.past_flows = np.repeat(
            [instance.get_initial_lags_of_channel(self.idx)],
            repeats=self.num_scenarios,
            axis=0,
        )

        # Initial flow limit (m3/s)
        self.flow_limit = self.get_flow_limit(dam_vol)

        return

    def reset(self, dam_vol: np.ndarray, instance: Instance, num_scenarios: int):

        """
        Reset the channel and the power group within.
        """

        self.num_scenarios = num_scenarios

        self._reset_variables(instance, dam_vol=dam_vol)

        self.power_group.reset(
            past_flows=self.past_flows, num_scenarios=self.num_scenarios
        )

        return

    def get_flow_limit(self, dam_vol: np.ndarray) -> np.ndarray:

        """
        The flow the channel can carry is limited by the volume of the dam.

        :param dam_vol:
            Array of shape num_scenarios with
            the volume of the dam connected to the channel in every scenario (m3)
        :return:
            Array of shape num_scenarios with
            the flow limit in every scenario (m3/s)
        """

        if self.limit_flow_points is None:
            return np.repeat(self.flow_max, self.num_scenarios)

        # Interpolate volume to get flow
        flow_limit = np.interp(
            dam_vol,
            self.limit_flow_points["observed_vols"],
            self.limit_flow_points["observed_flows"],
        )

        # Make sure limit is below maximum flow
        flow_limit = np.clip(flow_limit, 0, self.flow_max)

        return flow_limit

    def update(self, flow: np.ndarray, price: float, dam_vol: np.ndarray) -> np.ndarray:

        """
        Update the record of flows through the channel, its current maximum flow,
        and the state of the power group after it.

        :param flow:
            Array of shape num_scenarios with
            the flow going through the channel in the current time step in every scenario (m3/s)
        :param price: Price of energy in current time step (EUR/MWh)
        :param dam_vol:
            Array of shape num_scenario with
            the volume of the dam connected to the channel in every scenario (m3)
        :return:
            Array of shape num_scenarios with
            the turbined flow in the power group in every scenario (m3/s)
        """

        # Append a column to the left of the array with the assigned flow to the channel in every scenario
        self.past_flows = np.insert(self.past_flows, obj=0, values=flow, axis=1)
        self.past_flows = np.delete(
            self.past_flows, obj=self.past_flows.shape[1] - 1, axis=1
        )

        # Update flow limit to get the flow limit at the END of this time step
        # This is used in the next update() call of Dam to calculate flow_out_clipped1
        self.flow_limit = self.get_flow_limit(dam_vol)

        # Update power group and get turbined flow
        return self.power_group.update(price=price, past_flows=self.past_flows)
