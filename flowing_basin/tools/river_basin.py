from flowing_basin.core import Instance
from .dam import Dam


class RiverBasin:

    def __init__(self, instance: Instance, path_flow_max_model: str, path_power_model: str):

        self.dams = []
        num_dams = instance.get_num_dams()
        for dam_ident in range(num_dams):
            dam = Dam(ident=dam_ident,
                      instance=instance,
                      path_flow_max_model=path_flow_max_model,
                      path_power_model=path_power_model)
            self.dams.append(dam)

    def update(self, flows: list) -> None:

        """

        :param flows: List of flows (the indices correspond to the IDs of the channels)
        """

        for dam in self.dams:
            dam.update(flows=flows)

    def get_state(self):

        pass
