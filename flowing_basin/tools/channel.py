from flowing_basin.core import Instance
from power_group import PowerGroup


class Channel:

    def __init__(self, ident: int, dam_vol: float,
                 instance: Instance, path_flow_max_model: str, path_power_model: str):

        self.ident = ident
        self.path_flow_max_model = path_flow_max_model

        self.flows_over_time = [instance.get_initial_flow_of_channel(ident)]
        self.flow_max = self.get_max_flow(dam_vol)

        self.power_group = PowerGroup(flows_over_time=self.flows_over_time,
                                      path_power_model=path_power_model)
        # Assuming there is only one power group per channel
        # TODO: Let there be more than one channel per dam

    def get_max_flow(self, dam_vol: float) -> float:

        """
        Using the parameters of the model saved in self.path_flow_max_model,
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
