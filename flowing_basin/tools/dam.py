from flowing_basin.core import Instance
from channel import Channel


class Dam:
    def __init__(self, ident: int, instance: Instance, path_power_model: str):

        # Identifier of dam
        self.ident = ident

        # Volume at the END of the current time step
        # TODO: confirm this is what we want
        self.volume = instance.get_initial_vol_of_dam(ident)

        # Constant values for the whole period
        self.time_step = instance.get_time_step()
        self.min_volume = instance.get_min_vol_of_dam(ident)
        self.max_volume = instance.get_max_vol_of_dam(ident)
        self.unregulated_flow = instance.get_unregulated_flow_of_dam(ident)

        # Save instance to get incoming flows in the update method
        self.instance = instance

        self.channel = Channel(
            ident=ident,
            dam_vol=self.volume,
            instance=instance,
            path_power_model=path_power_model,
        )

    def update(self, flows: list, time: int) -> None:

        """
        Update the volume of the dam, and the state of its connected channel
        :param flows: decision
        :param time:
        :return:
        """

        turbined_flow_preceding_dam = 0
        # TODO: this should be the turbined flow of the preceding dam in the CURRENT time step

        # Obtain flow coming into the dam from the river or the previous dam
        if self.ident == 1:
            flow_contribution = self.instance.get_incoming_flow(time)
        else:
            flow_contribution = turbined_flow_preceding_dam

        # Obtain flow coming out of the dam
        flow_out = flows[self.ident - 1]

        # Update volume
        self.volume = (
            self.volume
            + (self.unregulated_flow + flow_contribution - flow_out)
            * self.time_step
        )

        # We update the channel with the new volume (the FINAL volume in this time step),
        # because the channel stores the FINAL maximum flow, which is calculated with this volume
        self.channel.update(flows=flows, dam_vol=self.volume)
