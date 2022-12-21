from flowing_basin.core import Instance
from .channel import Channel


class Dam:

    def __init__(self, ident: int,
                 instance: Instance, path_flow_max_model: str, path_power_model: str):

        self.ident = ident
        self.time_step = instance.get_time_step()

        self.volume = instance.get_initial_vol_of_dam(ident)
        self.min_volume = instance.get_min_vol_of_dam(ident)
        self.max_volume = instance.get_max_vol_of_dam(ident)

        self.unregulated_flow = instance.get_unregulated_flow_of_dam(ident)
        # Assuming the unregulated flow is the same for the whole day
        # TODO: let there be a different unregulated flow for each time step

        self.channel = Channel(ident=ident,
                               dam_vol=self.volume,
                               instance=instance,
                               path_flow_max_model=path_flow_max_model,
                               path_power_model=path_power_model)
        # Assuming there is only one channel per dam
        # TODO: Let there be more than one channel per dam

    def update(self, flows: list) -> None:

        """
        Update the volume of the dam, and the state of its connected channel
        :param flows:
        :return:
        """

        # Obtain flow coming into the dam from the previous dam
        flow_contribution = 0
        if self.ident > 0:
            flow_contribution = flows[self.ident-1]

        # Obtain flow coming out of the dam
        flow_out = flows[self.ident]

        # Update volume
        self.volume = self.volume + (self.unregulated_flow + flow_contribution - flow_out) * self.time_step

        self.channel.update(flows=flows, dam_vol=self.volume)
