import pickle
from cornflow_client import InstanceCore, get_empty_schema


class Instance(InstanceCore):
    schema = get_empty_schema()
    schema_checks = get_empty_schema()

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def to_dict(self):
        return pickle.loads(pickle.dumps(self.data, -1))

    def get_time_step(self) -> float:

        """

        :return: The time between updates in seconds
        """

        pass

    def get_num_dams(self) -> int:

        """

        :return: The number of dams of the river basin
        """

        pass

    def get_initial_vol_of_dam(self, ident: int) -> float:

        """

        :param ident: Identifier of the dam in the river basin
        :return: The volume of the dam in the beginning
        """

        pass

    def get_min_vol_of_dam(self, ident: int) -> float:

        """

        :param ident: Identifier of the dam in the river basin
        :return: Minimum volume of the dam
        """

        pass

    def get_max_vol_of_dam(self, ident: int) -> float:

        """

        :param ident: Identifier of the dam in the river basin
        :return: Maximum volume of the dam
        """

        pass

    def get_unregulated_flow_of_dam(self, ident: int) -> float:

        """

        :param ident: Identifier of the dam
        :return: Unregulated flow that enters the dam (flow that comes from the river)
        """

        pass

    def get_initial_flow_of_channel(self, ident: int) -> float:

        """

        :param ident: Identifier of the channel
        :return: Flow that goes through the channel in the beginning
        """

        pass
