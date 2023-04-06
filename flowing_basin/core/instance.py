from cornflow_client import InstanceCore, get_empty_schema
from cornflow_client.core.tools import load_json
import pickle
from datetime import datetime
import os
import warnings


class Instance(InstanceCore):

    schema = load_json(
        os.path.join(os.path.dirname(__file__), "../schemas/instance.json")
    )
    schema_checks = get_empty_schema()

    @classmethod
    def from_dict(cls, data) -> "Instance":

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

    def check_inconsistencies(self) -> dict:

        """
        Method that checks if there are inconsistencies in the data of the current instance.
        :return: A dictionary containing the inconsistencies found.
        """

        inconsistencies = dict()

        # Number of time steps ---- #

        # Get number of time steps
        num_time_steps = self.get_largest_impact_horizon()

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

        # Number of X and Y observations ---- #

        # The number of given volumes must equal the number of given observed flows
        for dam_id in self.get_ids_of_dams():
            obs_flow_limit = self.get_flow_limit_obs_for_channel(dam_id)
            if obs_flow_limit is not None:
                num_observed_vols = len(obs_flow_limit["observed_vols"])
                num_observed_flows = len(obs_flow_limit["observed_flows"])
                if num_observed_vols != num_observed_flows:
                    inconsistencies.update(
                        {
                            "In the flow limit data of " + dam_id + ", "
                            "the number of given volumes is not the same as "
                            "the number of observed flows": f"{num_observed_vols} vs. {num_observed_flows}"
                        }
                    )

        # The number of given flows must equal the number of given observed powers
        for dam_id in self.get_ids_of_dams():
            obs_turbined_flow = self.get_turbined_flow_obs_for_power_group(dam_id)
            num_observed_flows = len(obs_turbined_flow["observed_flows"])
            num_observed_powers = len(obs_turbined_flow["observed_powers"])
            if num_observed_flows != num_observed_powers:
                inconsistencies.update(
                    {
                        "In the turbined flow data of " + dam_id + ", "
                        "the number of given flows is not the same as "
                        "the number of observed powers": f"{num_observed_flows} vs. {num_observed_powers}"
                    }
                )

        # Number of startup and shutdown flows ---- #

        # The number of given startup flows must be equal to the number of given shutdown flows
        for dam_id in self.get_ids_of_dams():
            num_startup_flows = len(self.get_startup_flows_of_power_group(dam_id))
            num_shutdown_flows = len(self.get_shutdown_flows_of_power_group(dam_id))
            if num_startup_flows != num_shutdown_flows:
                inconsistencies.update(
                    {
                        "In the power group data of " + dam_id + ", "
                        "the number of startup flows is not the same as "
                        "the number of shutdown flows": f"{num_startup_flows} vs. {num_shutdown_flows}"
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
            inconsistencies.update(
                {"The given data does not follow the schema": schema_errors}
            )

        return inconsistencies

    def get_start_end_datetimes(self) -> tuple[datetime, datetime]:

        """

        :return: Starting datetime and final datetime (decision horizon)
        """

        start = datetime.strptime(self.data["datetime"]["start"], "%Y-%m-%d %H:%M")
        end_decisions = datetime.strptime(self.data["datetime"]["end_decisions"], "%Y-%m-%d %H:%M")

        return start, end_decisions

    def get_time_step_seconds(self) -> float:

        """

        :return: The time between updates in seconds (s)
        """

        return self.data["time_step_minutes"] * 60

    def get_decision_horizon(self) -> int:

        """
        Get the number of time steps up to the decision horizon
        (number of time steps in which we have to choose the flows).

        For example, if the instance spans one day, and we consider steps of 1/4 hour,
        this will be 24*4 = 96.

        :return: Number of time steps up to the decision horizon
        """

        start, end_decisions = self.get_start_end_datetimes()
        difference = end_decisions - start
        num_time_steps_decisions = difference.total_seconds() // self.get_time_step_seconds() + 1

        return int(num_time_steps_decisions)

    def get_largest_impact_horizon(self) -> int:

        """
        Get the number of time steps up to the largest impact horizon
        (maximum number of time steps in which the chosen flows have an impact in the income obtained).
        This should be equal to the total number of time steps of the instance
        (that is, the number of time steps for which we have data on the energy price, the unregulated flows, etc.).

        For example, if the instance spans one day, we consider steps of 1/4 hour, and
        the longest channel has a maximum delay of 3/4 hour, this will be (24 + 3/4) * 4 = 99.

        :return: Number of time steps up to the largest impact horizon
        """

        decision_horizon = self.get_decision_horizon()
        max_lag = max([self.get_relevant_lags_of_dam(dam_id)[0] for dam_id in self.get_ids_of_dams()])
        # TODO: Shouldn't we take self.get_relevant_lags_of_dam(dam_id)[-1],
        #  to get the max delay of each channel, and not the min?

        return decision_horizon + max_lag

    def get_num_dams(self) -> int:

        """

        :return: The number of dams in the river basin
        """

        return len(self.data["dams"])

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

    def get_verification_lags_of_dam(self, idx: str) -> list[int]:

        """

        :param idx: ID of the dam in the river basin
        :return: List of the verification lags of the dam (1 lag = 15 minutes of time delay)
        This must be a subset of the relevant lags, containing only the most important lags
        At each time step, the turbined flow should be roughly equal to the average of the verification lags
        """

        return self.data["dams"][idx]["verification_lags"]

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

    def get_turbined_flow_obs_for_power_group(self, idx: str) -> dict[str, list]:

        """

        :param idx: ID of the dam in the river basin
        :return: Dictionary with a list of turbined flows and the corresponding power observed (m3/s and MW)
        """

        points = {
            "observed_flows": self.data["dams"][idx]["turbined_flow"]["observed_flows"],
            "observed_powers": self.data["dams"][idx]["turbined_flow"][
                "observed_powers"
            ],
        }

        return points

    def get_startup_flows_of_power_group(self, idx: str) -> list[float]:

        """

        :param idx: ID of the dam in the river basin
        :return: List with the startup flows of the power group (m3/s)
        When the turbined flow exceeds one of these flows, an additional power group unit is activated
        """

        return self.data["dams"][idx]["startup_flows"]

    def get_shutdown_flows_of_power_group(self, idx: str) -> list[float]:

        """

        :param idx: ID of the dam in the river basin
        :return: List with the shutdown flows of the power group (m3/s)
        When the turbined flow falls behind one of these flows,one of the power group units is deactivated
        """

        return self.data["dams"][idx]["shutdown_flows"]

    def get_unregulated_flow_of_dam(
        self, time: int, idx: str, num_steps: int = 1
    ) -> float | list[float] | None:

        """

        :param time: Identifier of the current time step
        For example, if we consider steps of 15min for a whole day, this parameter will range from 0 to 95 (24*4)
        :param idx: ID of the dam in the river basin
        :param num_steps: Number of time steps to look ahead (by default, only the current time step is considered)
        :return: Unregulated flow that enters the dam (flow that comes from the river) in all of these time steps (m3/s)
        """

        if time >= self.get_largest_impact_horizon():
            warnings.warn(
                f"Tried to access unregulated flow for {time=}, "
                f"which is equal or greater than {self.get_largest_impact_horizon()=}. "
                f"None was returned"
            )
            return None

        unreg_flows = self.data["dams"][idx]["unregulated_flows"][
            time : time + num_steps
        ]
        if num_steps == 1:
            unreg_flows = unreg_flows[0]

        return unreg_flows

    def get_max_unregulated_flow_of_dam(self, idx: str) -> float:

        """

        :param idx: ID of the dam in the river basin
        :return: Maximum unregulated flow that can enter the dam (in any date) (m3/s)
        """

        return self.data["dams"][idx]["unregulated_flow_max"]

    def get_incoming_flow(
        self, time: int, num_steps: int = 1
    ) -> float | list[float] | None:

        """

        :param time: Identifier of the current time step
        :param num_steps: Number of time steps to look ahead (by default, only the current time step is considered)
        :return: FLow entering the first dam in all of these time steps (m3/s)
        """

        if time >= self.get_largest_impact_horizon():
            warnings.warn(
                f"Tried to access incoming flow for {time=}, "
                f"which is equal or greater than {self.get_largest_impact_horizon()=}. "
                f"None was returned"
            )
            return None

        incoming_flows = self.data["incoming_flows"][time : time + num_steps]
        if num_steps == 1:
            incoming_flows = incoming_flows[0]

        return incoming_flows

    def get_max_incoming_flow(self) -> float:

        """

        :return: Maximum possible value for the incoming flow (in any date) (m3/s)
        """

        return self.data["incoming_flow_max"]

    def get_price(self, time: int, num_steps: int = 1) -> float | list[float] | None:

        """

        :param time: Identifier of the current time step
        :param num_steps: Number of time steps to look ahead (by default, only the current time step is considered)
        :return: Price of energy in all of these time steps (EUR/MWh)
        """

        if time >= self.get_largest_impact_horizon():
            warnings.warn(
                f"Tried to access price for {time=}, "
                f"which is equal or greater than {self.get_largest_impact_horizon()=}. "
                f"None was returned"
            )
            return None

        prices = self.data["energy_prices"][time : time + num_steps]
        if num_steps == 1:
            prices = prices[0]

        return prices
