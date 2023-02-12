from flowing_basin.core import Instance
from .channel import Channel


class Dam:
    def __init__(
        self, idx: str, instance: Instance, paths_power_models: dict[str, str]
    ):

        # Identifier and order of the dam
        self.idx = idx
        self.order = instance.get_order_of_dam(self.idx)

        # Initial volume of dam
        self.volume = instance.get_initial_vol_of_dam(self.idx)

        # Constant values for the whole period
        self.time_step = instance.get_time_step()
        self.min_volume = instance.get_min_vol_of_dam(self.idx)
        self.max_volume = instance.get_max_vol_of_dam(self.idx)

        self.channel = Channel(
            idx=self.idx,
            dam_vol=self.volume,
            instance=instance,
            paths_power_models=paths_power_models,
        )

    def update(
        self,
        flows: dict[str, float],
        incoming_flow: float,
        unregulated_flow: float,
        turbined_flow_of_preceding_dam: float,
    ) -> float:

        """
        Update the volume of the dam, and the state of its connected channel
        :param flows: decision
        :param incoming_flow:
        :param unregulated_flow:
        :param turbined_flow_of_preceding_dam:
        :return: Turbined flow in the power group
        """

        # Obtain flow coming into the dam from the river or the previous dam
        if self.order == 1:
            flow_contribution = incoming_flow
        else:
            flow_contribution = turbined_flow_of_preceding_dam

        # Obtain flow coming out of the dam
        flow_out = flows[self.idx]

        # Update volume to get the volume at the END of this time step
        old_volume = self.volume
        volume_increase = (unregulated_flow + flow_contribution) * self.time_step
        volume_decrease = flow_out * self.time_step
        self.volume = old_volume + volume_increase - volume_decrease

        # Clip volume to min and max values
        flows_p = flows
        if self.volume > self.max_volume:
            self.volume = self.max_volume
        if self.volume < self.min_volume:
            # In this case, we must also clip the water that gets out of the dam
            flows_p[self.idx] = (old_volume + volume_increase - self.min_volume) / self.time_step
            self.volume = self.min_volume

        # We update the channel with the new volume (the FINAL volume in this time step),
        # because the channel stores the FINAL maximum flow, which is calculated with this volume
        return self.channel.update(flows=flows_p, dam_vol=self.volume)
