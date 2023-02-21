from flowing_basin.core import Instance
from .dam import Dam
import numpy as np


class RiverBasin:

    """
    Class representing the river basin
    """

    def __init__(
        self,
        instance: Instance,
        paths_power_models: dict[str, str],
        num_scenarios: int = 1,
    ):

        # Number of scenarios (e.g. candidate solutions) for which to do calculations at the same time
        self.num_scenarios = num_scenarios

        # Dams inside the flowing basin
        self.dams = dict()
        for dam_index, dam_id in enumerate(instance.get_ids_of_dams()):
            dam = Dam(
                index=dam_index,
                idx=dam_id,
                instance=instance,
                num_scenarios=self.num_scenarios,
                paths_power_models=paths_power_models,
            )
            self.dams.update({dam_id: dam})

        # Identifier of the time step (increases with each update)
        self.time = 0

        # Save instance to get incoming and unregulated flows in the update method
        self.instance = instance

    def reset(self, instance: Instance = None):

        """
        Resets the river basin
        This method resets the instance (if given)
        and all attributes that represent time-dependent values
        """

        self.time = 0
        if instance is not None:
            self.instance = instance

        for dam in self.dams.values():
            dam.reset(self.instance)

    def update(self, flows: list[float] | np.ndarray) -> float | np.ndarray:

        """

        :param flows:
         - List of flows that should go through each channel in the current time step (m3/s)
         - OR Array of shape num_dams x num_scenarios with these flows for every scenario (m3/s)
        :return:
         - Income obtained with the indicated flows in this time step (€)
         - OR Array of size num_scenarios with the income obtained in every scenario (€)
        """

        if isinstance(flows, list):
            assert len(flows) == self.instance.get_num_dams()
        if isinstance(flows, np.ndarray):
            assert flows.shape == (self.instance.get_num_dams(), self.num_scenarios)

        # The first dam has no preceding dam
        turbined_flow_of_preceding_dam = 0

        # Clip flows according to the flow limits of the channels
        for dam_index, dam in enumerate(self.dams.values()):
            flows[dam_index] = np.clip(flows[dam_index], 0, dam.channel.flow_limit)
            if self.num_scenarios == 1:
                flows[dam_index] = flows[dam_index].item()

        # Update dams
        for dam_id, dam in self.dams.items():
            turbined_flow = dam.update(
                flows=flows,
                incoming_flow=self.instance.get_incoming_flow(self.time),
                unregulated_flow=self.instance.get_unregulated_flow_of_dam(
                    self.time, dam_id
                ),
                turbined_flow_of_preceding_dam=turbined_flow_of_preceding_dam,
            )
            turbined_flow_of_preceding_dam = turbined_flow

        # Calculate income
        price = self.instance.get_price(self.time)
        power = sum(
            dam.channel.power_group.power for dam in self.dams.values()
        )
        time_step_hours = self.instance.get_time_step() / 3600
        income = price * power * time_step_hours

        # Increase time step identifier to get the next incoming and unregulated flows
        self.time = self.time + 1

        return income

    def deep_update(self, flows_all_periods: list[list[float]] | np.ndarray) -> float:

        """

        :param flows_all_periods:
         - Lists of lists with the flows that should go through each channel in every time step (m3/s)
         - OR Array of shape num_time_steps x num_dams x num_scenarios with these flows for every scenario (m3/s)
        :return:
         - Accumulated income obtained with the indicated flows in all time steps (€)
         - OR Array of size num_scenarios with the accumulated income obtained in every scenario (€)
        """

        income = 0
        for flows in flows_all_periods:
            income += self.update(flows)

        return income

    def get_state(self):

        """
        Returns the state of the river basin
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
            "incoming_flow": self.instance.get_incoming_flow(self.time),
            "price": self.instance.get_price(self.time),
        }

        for dam_id, dam in self.dams.items():
            state[dam_id] = {
                "vol": dam.volume,
                "flow_limit": dam.channel.flow_limit,
                "unregulated_flow": self.instance.get_unregulated_flow_of_dam(
                    self.time, dam_id
                ),
                "lags": dam.channel.flows_over_time,
                "power": dam.channel.power_group.power,
                "turbined_flow": dam.channel.power_group.turbined_flow,
            }

        return state
