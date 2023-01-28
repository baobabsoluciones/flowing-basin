from flowing_basin.core import Instance
from .power_group import PowerGroup
from collections import deque


class Channel:
    def __init__(
        self,
        ident: int,
        dam_vol: float,
        instance: Instance,
        path_power_model: str,
    ):

        self.ident = ident
        self.max_flow_points = instance.get_max_flow_points_of_channel(ident)

        initial_lags = instance.get_initial_lags_of_channel(ident)
        num_lags = instance.get_relevant_lags_of_dam(ident)[-1]
        self.flows_over_time = deque(initial_lags, maxlen=num_lags)

        # Inicial maximum flow of channel
        self.flow_max = self.get_max_flow(dam_vol)

        self.power_group = PowerGroup(
            flows_over_time=self.flows_over_time,
            path_power_model=path_power_model,
        )

    def get_max_flow(self, dam_vol: float) -> float:

        """
        Using the points saved in self.max_flow_points,
        returns the maximum flow that the channel can carry
        :param dam_vol: Volume of preceding dam
        :return:
        """

        # TODO: implement this function

        pass

    def update(self, flows: list, dam_vol: float) -> float:

        """
        Update the record of flows through the channel, its current maximum flow,
        and the state of the power group after it
        :param flows:
        :param dam_vol:
        :return: Turbined flow in the power group
        """

        self.flows_over_time.appendleft(flows[self.ident - 1])

        # Update maximum flow to get the maximum flow at the END of this time step
        self.flow_max = self.get_max_flow(dam_vol)

        # Update power group and get turbined flow
        return self.power_group.update(flows_over_time=self.flows_over_time)
