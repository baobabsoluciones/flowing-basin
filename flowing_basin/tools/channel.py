from flowing_basin.core import Instance
from .power_group import PowerGroup
from collections import deque
import numpy as np


class Channel:
    def __init__(
        self,
        idx: str,
        dam_vol: float,
        instance: Instance,
        paths_power_models: dict[str, str],
    ):

        self.idx = idx
        self.limit_flow_points = instance.get_flow_limit_obs_for_channel(self.idx)

        initial_lags = instance.get_initial_lags_of_channel(self.idx)
        num_lags = instance.get_relevant_lags_of_dam(self.idx)[-1]
        self.flows_over_time = deque(initial_lags, maxlen=num_lags)
        # We need to keep track of as many past flows as the last relevant lag

        # Maximum flow of channel
        self.flow_max = instance.get_max_flow_of_channel(self.idx)

        # Initial flow limit
        self.flow_limit = self.get_flow_limit(dam_vol)

        self.power_group = PowerGroup(
            idx=self.idx,
            flows_over_time=self.flows_over_time,
            instance=instance,
            paths_power_models=paths_power_models,
        )

    def get_flow_limit(self, dam_vol: float) -> float:

        """
        The flow the channel can carry is limited by the volume of the dam
        :param dam_vol: Volume of preceding dam
        :return: Flow limit (maximum flow for given volume)
        """

        if self.limit_flow_points is None:
            flow_limit = self.flow_max
        else:
            # Interpolate volume to get flow
            flow_limit = np.interp(
                dam_vol,
                self.limit_flow_points["observed_vols"],
                self.limit_flow_points["observed_flows"]
            )
            # Make sure limit is below maximum flow
            flow_limit = float(np.clip(flow_limit, 0, self.flow_max))

        return flow_limit

    def update(self, flows: dict[str, float], dam_vol: float) -> float:

        """
        Update the record of flows through the channel, its current maximum flow,
        and the state of the power group after it
        :param flows:
        :param dam_vol:
        :return: Turbined flow in the power group
        """

        self.flows_over_time.appendleft(flows[self.idx])

        # Update flow limit to get the flow limit at the END of this time step
        self.flow_limit = self.get_flow_limit(dam_vol)

        # Update power group and get turbined flow
        return self.power_group.update(flows_over_time=self.flows_over_time)
