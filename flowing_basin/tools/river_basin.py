from flowing_basin.core import Instance
from .dam import Dam
import numpy as np


class RiverBasin:
    def __init__(self, instance: Instance, paths_power_models: dict[str, str]):

        # Dams inside the flowing basin
        self.dams = dict()
        dam_ids = instance.get_ids_of_dams()
        for dam_id in dam_ids:
            dam = Dam(
                idx=dam_id,
                instance=instance,
                paths_power_models=paths_power_models,
            )
            self.dams.update({dam_id: dam})

        # Identifier of the time step (increases with each update)
        self.time = 0

        # Save instance to get incoming and unregulated flows in the update method
        self.instance = instance

    def update(self, flows: dict[str, float]) -> None:

        """

        :param flows: Dictionary of flows that should go through each channel, indexed by dam
        """

        # The first dam has no preceding dam
        turbined_flow_of_preceding_dam = 0

        # Clip flows according to the flow limits of the channels
        flows_p = flows
        for dam_id, dam in self.dams.items():
            flows_p[dam_id] = float(np.clip(flows[dam_id], 0, dam.channel.flow_limit))

        # Update dams
        for dam_id, dam in self.dams.items():
            turbined_flow = dam.update(
                flows=flows_p,
                incoming_flow=self.instance.get_incoming_flow(self.time),
                unregulated_flow=self.instance.get_unregulated_flow_of_dam(self.time, dam_id),
                turbined_flow_of_preceding_dam=turbined_flow_of_preceding_dam,
            )
            turbined_flow_of_preceding_dam = turbined_flow

        # Increase time step identifier to get the next incoming and unregulated flows
        self.time = self.time + 1

    def get_state(self):

        """
        Returns the state of the river basin
        :return:
        """

        state = {
            "incoming_flow": self.instance.get_incoming_flow(self.time),
            "price": self.instance.get_price(self.time)
        }

        for dam_id, dam in self.dams.items():
            state[dam_id] = {
                "vol": dam.volume,
                "flow_limit": dam.channel.flow_limit,
                "unregulated_flow": self.instance.get_unregulated_flow_of_dam(self.time, dam_id),
                "lags": dam.channel.flows_over_time,
                "power": dam.channel.power_group.power,
                "turbined_flow": dam.channel.power_group.turbined_flow,
            }

        return state
