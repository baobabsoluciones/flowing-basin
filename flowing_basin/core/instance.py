import pickle
from typing import List
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
        return self.data["time_step_minutes"] * 60

    def get_num_dams(self) -> int:

        """

        :return: The number of dams of the river basin
        """

        return len(self.data["dams"])

    def get_initial_vol_of_dam(self, ident: int) -> float:

        """

        :param ident: Identifier of the dam in the river basin, 1 .. num_dams
        :return: The volume of the dam in the beginning
        """

        return self.data["dams"][ident - 1]["initial_vol"]

    def get_min_vol_of_dam(self, ident: int) -> float:

        """

        :param ident: Identifier of the dam in the river basin, 1 .. num_dams
        :return: Minimum volume of the dam
        """

        return self.data["dams"][ident - 1]["vol_min"]

    def get_max_vol_of_dam(self, ident: int) -> float:

        """

        :param ident: Identifier of the dam in the river basin, 1 .. num_dams
        :return: Maximum volume of the dam
        """

        return self.data["dams"][ident - 1]["vol_max"]

    def get_unregulated_flow_of_dam(self, ident: int) -> float:

        """

        :param ident: Identifier of the dam, 1 .. num_dams
        :return: Unregulated flow that enters the dam (flow that comes from the river)
        """

        return self.data["dams"][ident - 1]["unreg_flow"]

    def get_initial_lags_of_channel(self, ident: int) -> List[float]:

        """

        :param ident: Identifier of the channel, 1 .. num_dams
        :return: Flow that goes through the channel in the beginning
        """

        return self.data["dams"][ident - 1]["initial_lags"]

    def get_relevant_lags_of_dam(self, ident: int) -> List[int]:

        """

        :param ident: Identifier of the dam, 1 .. num_dams
        :return: List of the relevant lags of the dam (1 lag = 15 minutes of time delay)
        """

        return self.data["dams"][ident - 1]["relevant_lags"]

    def get_max_flow_points_of_channel(self, ident: int) -> List[List[int]]:

        """

        :param ident: Identifier of the channel, 1 .. num_dams
        :return: List of the max flow points of the channel (vol_dam, max_flow)
        """

        return self.data["dams"][ident - 1]["max_flow_points"]

    def get_incoming_flow(self, time: int) -> float:

        """

        :param time: Identifier of the time step; if we consider steps of 15min for a whole day, 0 .. 95 (24*4)
        :return: FLow entering the first dam
        """
        return self.data["incoming_flow_in_day"][time]
