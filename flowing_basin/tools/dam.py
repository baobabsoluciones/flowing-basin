from flowing_basin.core import Instance
from .channel import Channel


class Dam:
    def __init__(self, idx: str, instance: Instance, path_power_model: str):

        # Identifier and order of the dam
        self.idx = idx
        self.order = instance.get_order_of_dam(self.idx)

        # Initial volume of dam
        self.volume = instance.get_initial_vol_of_dam(self.idx)

        # Constant values for the whole period
        self.time_step = instance.get_time_step()
        self.min_volume = instance.get_min_vol_of_dam(self.idx)
        self.max_volume = instance.get_max_vol_of_dam(self.idx)
        self.unregulated_flow = instance.get_unregulated_flow_of_dam(self.idx)

        self.channel = Channel(
            idx=self.idx,
            dam_vol=self.volume,
            instance=instance,
            path_power_model=path_power_model,
        )

    def update(
        self,
        flows: dict[str, float],
        incoming_flow: float,
        turbined_flow_of_preceding_dam: float,
    ) -> float:

        """
        Update the volume of the dam, and the state of its connected channel
        :param flows: decision
        :param incoming_flow:
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
        self.volume = (
            self.volume
            + (self.unregulated_flow + flow_contribution - flow_out) * self.time_step
        )

        # We update the channel with the new volume (the FINAL volume in this time step),
        # because the channel stores the FINAL maximum flow, which is calculated with this volume
        return self.channel.update(flows=flows, dam_vol=self.volume)
