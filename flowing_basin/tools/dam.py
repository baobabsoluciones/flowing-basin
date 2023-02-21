from flowing_basin.core import Instance
from .channel import Channel
import numpy as np


class Dam:
    def __init__(
        self, index: int, idx: str, instance: Instance, paths_power_models: dict[str, str], num_scenarios: int
    ):

        self.num_scenarios = num_scenarios

        # Index, identifier and order of the dam
        self.index = index
        self.idx = idx
        self.order = instance.get_order_of_dam(self.idx)

        # Constant values for the whole period - time step (seconds, s), min/max volumes (m3)
        self.time_step = instance.get_time_step()
        self.min_volume = instance.get_min_vol_of_dam(self.idx)
        self.max_volume = instance.get_max_vol_of_dam(self.idx)

        # Initial volume of dam (m3)
        self.volume = instance.get_initial_vol_of_dam(self.idx)
        if self.num_scenarios > 1:
            self.volume = np.repeat(self.volume, self.num_scenarios)

        # Clip initial volume of dam
        self.volume = np.clip(self.volume, self.min_volume, self.max_volume)
        if self.num_scenarios == 1:
            self.volume = self.volume.item()

        self.channel = Channel(
            index=self.index,
            idx=self.idx,
            dam_vol=self.volume,
            instance=instance,
            num_scenarios=self.num_scenarios,
            paths_power_models=paths_power_models,
        )

    def reset(self, instance: Instance):

        self.volume = instance.get_initial_vol_of_dam(self.idx)
        if self.num_scenarios > 1:
            self.volume = np.repeat(self.volume, self.num_scenarios)

        self.volume = np.clip(self.volume, self.min_volume, self.max_volume)
        if self.num_scenarios == 1:
            self.volume = self.volume.item()

        self.channel.reset(dam_vol=self.volume, instance=instance)

    def update(
        self,
        flows: list[float] | np.ndarray,
        incoming_flow: float,
        unregulated_flow: float,
        turbined_flow_of_preceding_dam: float,
    ) -> float | np.ndarray:

        """
        Update the volume of the dam, and the state of its connected channel
        :param flows:
         - List of flows that should go through each channel, in order (m3/s)
         - OR Array of shape num_dams x num_scenarios with these flows for every scenario (m3/s)
        :param incoming_flow: Incoming flow to the river basin (m3/s)
        :param unregulated_flow: Unregulated flow entering the dam (m3/s)
        :param turbined_flow_of_preceding_dam:
         - Turbined flow of the previous dam, that is entering this dam (m3/s)
         - OR Array of shape num_scenarios containing these flows (m3/s)
        :return:
         - Turbined flow in the power group (m3/s)
         - OR Array of shape num_scenarios containing the turbined flow of every scenario (m3/s)
        """

        # Obtain flow coming into the dam from the river or the previous dam
        if self.order == 1:
            flow_contribution = incoming_flow
        else:
            flow_contribution = turbined_flow_of_preceding_dam

        # Obtain flow coming out of the dam
        flow_out = flows[self.index]

        # Update volume to get the volume at the END of this time step
        old_volume = self.volume
        volume_increase = (unregulated_flow + flow_contribution) * self.time_step
        volume_decrease = flow_out * self.time_step
        self.volume = old_volume + volume_increase - volume_decrease

        # Clip volume to min value
        self.volume = np.clip(self.volume, self.min_volume, None)
        if self.num_scenarios == 1:
            self.volume = self.volume.item()

        # Limit the water that gets out of the dam if volume was below minimum
        # This only changes the value of the flow if the volume was increased to the minimum value
        flows[self.index] = (old_volume + volume_increase - self.volume) / self.time_step

        # Clip volume to max value
        self.volume = np.clip(self.volume, None, self.max_volume)
        if self.num_scenarios == 1:
            self.volume = self.volume.item()

        # print(
        #     self.idx + " calc",
        #     old_volume,
        #     incoming_flow,
        #     unregulated_flow,
        #     flows[self.index],
        #     self.volume,
        # )

        # We update the channel with the new volume (the FINAL volume in this time step),
        # because the channel stores the FINAL maximum flow, which is calculated with this volume
        return self.channel.update(flows=flows, dam_vol=self.volume)
