import pickle
from cornflow_client import InstanceCore, get_empty_schema


class Instance(InstanceCore):
    schema = get_empty_schema()
    schema_checks = get_empty_schema()

    @classmethod
    def from_dict(cls, data):

        # Change list of dams into dictionary indexed by ID
        # This also changes the parent's from_json method, as this method calls from_dict
        data_p = dict(data)
        data_p["dams"] = {el["id"]: el for el in data_p["dams"]}

        return cls(data_p)

    def to_dict(self):

        # Change dictionary of dams into list, to undo de changes of from_dict
        # Use pickle to work with a copy of the data and avoid changing the property of the class
        data_p = pickle.loads(pickle.dumps(self.data, -1))
        data_p["dams"] = list(data_p["dams"].values())

        return data_p

    def get_time_step(self) -> float:

        """

        :return: The time between updates in seconds
        """

        return self.data["time_step_minutes"] * 60

    def get_ids_of_dams(self) -> list[str]:

        """

        :return: The IDs of all dams in the river basin
        """

        return list(self.data["dams"].keys())

    def get_order_of_dam(self, idx: str) -> int:

        """

        :param idx: ID of the dam in the river basin
        :return: Order of the dam in the river basin,
        """

        return self.data["dams"][idx]["order"]

    def get_initial_vol_of_dam(self, idx: str) -> float:

        """

        :param idx: ID of the dam in the river basin
        :return: The volume of the dam in the beginning
        """

        return self.data["dams"][idx]["initial_vol"]

    def get_min_vol_of_dam(self, idx: str) -> float:

        """

        :param idx: ID of the dam in the river basin
        :return: Minimum volume of the dam
        """

        return self.data["dams"][idx]["vol_min"]

    def get_max_vol_of_dam(self, idx: str) -> float:

        """

        :param idx: ID of the dam in the river basin
        :return: Maximum volume of the dam
        """

        return self.data["dams"][idx]["vol_max"]

    def get_unregulated_flow_of_dam(self, idx: str) -> float:

        """

        :param idx: ID of the dam in the river basin
        :return: Unregulated flow that enters the dam (flow that comes from the river)
        """

        return self.data["dams"][idx]["unregulated_flow"]

    def get_initial_lags_of_channel(self, idx: str) -> list[float]:

        """

        :param idx: ID of the dam in the river basin
        :return: Flow that goes through the channel in the beginning
        """

        return self.data["dams"][idx]["initial_lags"]

    def get_relevant_lags_of_dam(self, idx: str) -> list[int]:

        """

        :param idx: ID of the dam in the river basin
        :return: List of the relevant lags of the dam (1 lag = 15 minutes of time delay)
        """

        return self.data["dams"][idx]["relevant_lags"]

    def get_max_flow_points_of_channel(self, idx: str) -> list[list[int]]:

        """

        :param idx: ID of the dam in the river basin
        :return: List of the max flow points of the channel (vol_dam, max_flow)
        """

        return self.data["dams"][idx]["max_flow_points"]

    def get_incoming_flow(self, time: int) -> float:

        """

        :param time: Identifier of the time step
        For example, if we consider steps of 15min for a whole day, this parameter will range from 0 to 95 (24*4)
        :return: FLow entering the first dam
        """

        return self.data["incoming_flows"][time]

    def get_price(self, time: int) -> float:

        """

        :param time: Identifier of the time step
        :return: Price of energy for the given time step
        """

        return self.data["energy_prices"][time]
