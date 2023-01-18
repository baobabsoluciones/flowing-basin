from flowing_basin.core import Instance
from power_group import PowerGroup


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

        self.flows_over_time = [instance.get_initial_flow_of_channel(ident)]

        # Maximum flow of channel at the END of the current time step
        # TODO: confirm this is what we want
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

        pass

    def update(self, flows: list, dam_vol: float) -> None:

        """
        Update the record of flows through the channel, its current maximum flow,
        and the state of the power group after it
        :param flows:
        :param dam_vol:
        :return:
        """

        self.flows_over_time.append(flows[self.ident])
        self.flow_max = self.get_max_flow(dam_vol)

        self.power_group.update(flows_over_time=self.flows_over_time)
