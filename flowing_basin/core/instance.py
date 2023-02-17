from cornflow_client import InstanceCore, get_empty_schema
import pickle
import json
from datetime import datetime


class Instance(InstanceCore):

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

    @property
    def schema(self) -> dict:

        """
        a dictionary representation of the json-schema for the object
        """

        path = "../schemas/instance.json"

        with open(path, "r") as f:
            schema = json.load(f)

        return schema

    def check_inconsistencies(self) -> dict:

        """
        Method that checks if there are inconsistencies in the data of the current instance.
        :return: A dictionary containing the inconsistencies found.
        """

        inconsistencies = dict()

        # Number of time steps ---- #

        # Calculate number of time steps
        start = datetime.strptime(self.data["datetime"]["start"], "%Y-%m-%d %H:%M")
        end = datetime.strptime(self.data["datetime"]["end"], "%Y-%m-%d %H:%M")
        difference = end - start
        num_time_steps = difference.total_seconds() // self.get_time_step() + 1

        # Calculate the number of each time-dependent variable
        num_time_dep_var_values = {
            "incoming flows": len(self.data["incoming_flows"]),
            "energy prices": len(self.data["energy_prices"]),
            **{
                "unregulated flows of "
                + dam_id: len(self.data["dams"][dam_id]["unregulated_flows"])
                for dam_id in self.get_ids_of_dams()
            },
        }

        # The number of each time-dependent variable must equal the number of time steps
        for concept, number in num_time_dep_var_values.items():
            if number != num_time_steps:
                inconsistencies.update(
                    {
                        "The number of "
                        + concept
                        + " is not the same as the number of time steps": f"{number} vs. {num_time_steps}"
                    }
                )

        # Number of observations of volumes and flows for the flow limit calculation ---- #

        # The number of given volumes must equal the number of given observed flows
        for dam_id in self.get_ids_of_dams():
            observations = self.get_flow_limit_obs_for_channel(dam_id)
            if observations is not None:
                num_observed_vols = len(observations["observed_vols"])
                num_observed_flows = len(observations["observed_vols"])
                if num_observed_vols != num_observed_flows:
                    inconsistencies.update(
                        {
                            "In the flow limit data of " + dam_id + ", "
                            "the number of given volumes is not the same as "
                            "the number of observed flows": f"{num_observed_vols} vs. {num_observed_flows}"
                        }
                    )

        # Number of initial lags ---- #

        # The number of initial lags must equal the last relevant lag
        for dam_id in self.get_ids_of_dams():
            num_initial_lags = len(self.get_initial_lags_of_channel(dam_id))
            last_relevant_lag = self.get_relevant_lags_of_dam(dam_id)[-1]
            if num_initial_lags != last_relevant_lag:
                inconsistencies.update(
                    {
                        "The number of initial lags given to "
                        + dam_id
                        + " does not equal the last relevant lag of the dam": f"{num_initial_lags} vs. {last_relevant_lag}"
                    }
                )

        return inconsistencies

    def check(self):

        """
        Method that checks if the data of the instance does not follow the schema or has inconsistencies
        :return: A dictionary containing the schema noncompliance problems and inconsistencies found
        """

        inconsistencies = self.check_inconsistencies()
        schema_errors = self.check_schema()
        if schema_errors:
            inconsistencies.update({"The given data does not follow the schema": schema_errors})

        return inconsistencies

    def get_time_step(self) -> float:

        """

        :return: The time between updates in seconds (s)
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
        :return: The volume of the dam in the beginning (m3)
        """

        return self.data["dams"][idx]["initial_vol"]

    def get_min_vol_of_dam(self, idx: str) -> float:

        """

        :param idx: ID of the dam in the river basin
        :return: Minimum volume of the dam (m3)
        """

        return self.data["dams"][idx]["vol_min"]

    def get_max_vol_of_dam(self, idx: str) -> float:

        """

        :param idx: ID of the dam in the river basin
        :return: Maximum volume of the dam (m3)
        """

        return self.data["dams"][idx]["vol_max"]

    def get_initial_lags_of_channel(self, idx: str) -> list[float]:

        """

        :param idx: ID of the dam in the river basin
        :return: Flow that goes through the channel in the beginning (m3/s)
        """

        return self.data["dams"][idx]["initial_lags"]

    def get_relevant_lags_of_dam(self, idx: str) -> list[int]:

        """

        :param idx: ID of the dam in the river basin
        :return: List of the relevant lags of the dam (1 lag = 15 minutes of time delay)
        """

        return self.data["dams"][idx]["relevant_lags"]

    def get_max_flow_of_channel(self, idx: str) -> float:

        """

        :param idx: ID of the dam in the river basin
        :return: Maximum flow the channel can carry (m3/s)
        """

        return self.data["dams"][idx]["flow_max"]

    def get_flow_limit_obs_for_channel(self, idx: str) -> dict[str, list]:

        """

        :param idx: ID of the dam in the river basin
        :return: Dictionary with a list of volumes and the corresponding maximum flow limits observed (m3 and m3/s)
        """

        if self.data["dams"][idx]["flow_limit"]["exists"]:
            points = {
                "observed_vols": self.data["dams"][idx]["flow_limit"]["observed_vols"],
                "observed_flows": self.data["dams"][idx]["flow_limit"][
                    "observed_flows"
                ],
            }
        else:
            points = None

        return points

    def get_unregulated_flow_of_dam(self, time: int, idx: str) -> float:

        """

        :param time: Identifier of the time step
        For example, if we consider steps of 15min for a whole day, this parameter will range from 0 to 95 (24*4)
        :param idx: ID of the dam in the river basin
        :return: Unregulated flow that enters the dam (flow that comes from the river) (m3/s)
        """

        return self.data["dams"][idx]["unregulated_flows"][time]

    def get_incoming_flow(self, time: int) -> float:

        """

        :param time: Identifier of the time step
        :return: FLow entering the first dam (m3/s)
        """

        return self.data["incoming_flows"][time]

    def get_price(self, time: int) -> float:

        """

        :param time: Identifier of the time step
        :return: Price of energy for the given time step (EUR/MWh)
        """

        return self.data["energy_prices"][time]
