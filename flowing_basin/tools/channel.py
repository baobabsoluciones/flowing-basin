from flowing_basin.core import Instance
from .power_group import PowerGroup
import numpy as np


class Channel:
    def __init__(
        self,
        index: int,
        idx: str,
        dam_vol: float,
        instance: Instance,
        paths_power_models: dict[str, str],
        num_scenarios: int,
    ):

        self.num_scenarios = num_scenarios

        self.index = index
        self.idx = idx
        self.limit_flow_points = instance.get_flow_limit_obs_for_channel(self.idx)

        # Past flows (m3/s)
        # We save them as an array of shape num_scenarios x num_lags
        initial_lags = instance.get_initial_lags_of_channel(self.idx)
        self.flows_over_time = np.repeat([initial_lags], repeats=self.num_scenarios, axis=0)

        # Maximum flow of channel (m3/s)
        self.flow_max = instance.get_max_flow_of_channel(self.idx)

        # Initial flow limit (m3/s)
        self.flow_limit = self.get_flow_limit(dam_vol)

        self.power_group = PowerGroup(
            idx=self.idx,
            flows_over_time=self.flows_over_time,
            instance=instance,
            num_scenarios=self.num_scenarios,
            paths_power_models=paths_power_models,
        )

    def reset(self, dam_vol: float | np.ndarray, instance: Instance):

        initial_lags = instance.get_initial_lags_of_channel(self.idx)
        self.flows_over_time = np.repeat([initial_lags], repeats=self.num_scenarios, axis=0)

        self.flow_limit = self.get_flow_limit(dam_vol)

        self.power_group.reset(flows_over_time=self.flows_over_time)

    def get_flow_limit(self, dam_vol: float | np.ndarray) -> float | np.ndarray:

        """
        The flow the channel can carry is limited by the volume of the dam
        :param dam_vol: Volume of preceding dam (m3)
        :return: Flow limit (maximum flow for given volume) (m3/s)
        """

        if self.limit_flow_points is None:
            flow_limit = self.flow_max
            if self.num_scenarios > 1:
                flow_limit = np.repeat(flow_limit, self.num_scenarios)

            return flow_limit

        # Interpolate volume to get flow
        flow_limit = np.interp(
            dam_vol,
            self.limit_flow_points["observed_vols"],
            self.limit_flow_points["observed_flows"]
        )
        # Make sure limit is below maximum flow
        flow_limit = np.clip(flow_limit, 0, self.flow_max)
        if self.num_scenarios == 1:
            flow_limit = flow_limit.item()

        return flow_limit

    def update(self, flows: list[float] | np.ndarray, dam_vol: float | np.ndarray) -> float | np.ndarray:

        """
        Update the record of flows through the channel, its current maximum flow,
        and the state of the power group after it
        :param flows:
         - List of flows that should go through each channel, in order (m3/s)
         - OR Array of shape num_dams x num_scenarios with these flows for every scenario (m3/s)
        :param dam_vol:
         - Volume of the dam connected to the channel (m3)
         - OR Array of shape num_scenarios containing the volume of every scenario (m3)
        :return:
         - Turbined flow in the power group (m3/s)
         - OR Array of shape num_scenarios containing the turbined flow of every scenario (m3/s)
        """

        # Append a column to the left of the array with the assigned flow to the channel in every scenario
        self.flows_over_time = np.insert(self.flows_over_time, obj=0, values=flows[self.index], axis=1)
        self.flows_over_time = np.delete(self.flows_over_time, obj=self.flows_over_time.shape[1] - 1, axis=1)

        # Update flow limit to get the flow limit at the END of this time step
        # This is used in the next update() call of RiverBasin
        self.flow_limit = self.get_flow_limit(dam_vol)

        # Update power group and get turbined flow
        return self.power_group.update(flows_over_time=self.flows_over_time)
