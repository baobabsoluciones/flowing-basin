from cornflow_client import InstanceCore, get_empty_schema
from cornflow_client.core.tools import load_json
import pickle
from datetime import datetime
import os
import warnings
from copy import copy


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

        # Starting flows ---- #

        # The starting exiting flows must be provided when the start of information if before the start of the decisions
        info_offset = self.get_start_information_offset()
        if info_offset > 0:
            dam_ids_no_startig_flows = [
                dam_id for dam_id in self.get_ids_of_dams()
                if self.get_starting_flows(dam_id) is None
            ]
            if dam_ids_no_startig_flows:
                inconsistencies.update(
                    {
                        "Information starts before decisions, but some dams do not have starting flows defined":
                            f"Information offset is {info_offset}, and "
                            f"dams {dam_ids_no_startig_flows} do not have starting flows"
                    }
                )

        # The number of starting flows must equal the information start offset
        if info_offset > 0:
            dam_ids_diff_num_starting_flows = [
                (dam_id, len(self.get_starting_flows(dam_id))) for dam_id in self.get_ids_of_dams()
                if self.get_starting_flows(dam_id) is not None and len(self.get_starting_flows(dam_id)) != info_offset
            ]
            if dam_ids_diff_num_starting_flows:
                inconsistencies.update(
                    {
                        "The number of time steps between the start of information and the start of decisions "
                        "is not the same as the number of starting flows in some dams":
                            f"Information offset is {info_offset}, but: " +
                            ','.join([
                                f"{dam_id} has {num_starting_flows} starting flows"
                                for dam_id, num_starting_flows in dam_ids_diff_num_starting_flows
                            ])
                    }
                )

        # Number of time steps ---- #

        # Get number of time steps for which we need data (information horizon and offset)
        num_time_steps = self.get_information_horizon() + self.get_start_information_offset()

        # The largest impact horizon must be lower than the information horizon
        largest_impact_horizon = self.get_largest_impact_horizon()
        if largest_impact_horizon > num_time_steps:
            inconsistencies.update(
                {
                    "The largest impact horizon is greater than the information horizon":
                        f"{largest_impact_horizon} vs. {num_time_steps}"
                }
            )

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
                        + " does not equal the last relevant lag of the dam":
                            f"{num_initial_lags} vs. {last_relevant_lag}"
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

    def get_start_information_datetime(self) -> datetime:

        """
        Get the datetime where the information of the instance starts.
        This is the same as the starting point of the decisions,
        except when information about the past is necessary before we begin making decisions
        (like in RL).
        """

        start_info = self.data["datetime"].get("start_information")
        if start_info is not None:
            start_info = datetime.strptime(start_info, "%Y-%m-%d %H:%M")
        else:
            start_info, _ = self.get_start_end_datetimes()
        return start_info

    def get_start_information_offset(self) -> int:

        """
        Get the number of time steps between the start of the information and the start of the decisions.
        This will be 0,
        except when information about the past is necessary before we begin making decisions
        (like in RL).
        """

        start_decisions, _ = self.get_start_end_datetimes()
        start_info = self.get_start_information_datetime()
        difference = start_decisions - start_info
        num_time_steps_offset = difference.total_seconds() // self.get_time_step_seconds()

        return int(num_time_steps_offset)

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

    def get_information_horizon(self) -> int:

        """
        Get the number of time steps up to the information horizon
        (number of time steps in which we need to know the price, incoming flow, and unregulated flow).

        This horizon is, by default, equal to the largest impact horizon,
        but may need to be larger with some solvers (RL).

        :return: Number of time steps up to the information horizon
        """

        start, _ = self.get_start_end_datetimes()
        try:
            end_info = datetime.strptime(self.data["datetime"]["end_information"], "%Y-%m-%d %H:%M")
            difference = end_info - start
            return int(difference.total_seconds() // self.get_time_step_seconds() + 1)
        except KeyError:
            return self.get_largest_impact_horizon()

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
        :return: Order of the dam in the river basin, from 1 to num_dams
        """

        return self.data["dams"][idx]["order"]

    def get_dam_id_from_order(self, order: int) -> str:

        """

        :param order: Order of the dam in the river basin, from 1 to num_dams
        :return: ID of the dam in the river basin
        """

        for dam_id in self.get_ids_of_dams():
            if self.get_order_of_dam(dam_id) == order:
                return dam_id

        raise ValueError(f"The given dam order, {order}, has no corresponding dam ID.")

    def get_initial_vol_of_dam(self, idx: str) -> float:

        """

        :param idx: ID of the dam in the river basin
        :return: The volume of the dam in the beginning (m3)
        """

        return self.data["dams"][idx]["initial_vol"]

    def get_historical_final_vol_of_dam(self, idx: str) -> float | None:

        """
        Get the historical final volume of the given dam.
        This method may return None as this is an optional field of the instance.

        :param idx: ID of the dam in the river basin
        :return: The previously observed volume of the dam in the decision horizon (m3)
        """

        return self.data["dams"][idx].get("final_vol")

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
        :return:
            Flow that went through the channel in the previous time steps,
            in decreasing order (i.e., flow in time steps -1, ..., -last_lag) (m3/s)
        """

        return self.data["dams"][idx]["initial_lags"]

    def get_starting_flows(self, idx: str) -> list[float] | None:

        """

        :param idx: ID of the dam in the river basin
        :return:
            Flow that went through the channel in the time steps prior to the start of decisions,
            in decreasing order (i.e., flow in time steps -1, ..., -info_offset) (m3/s)
        """

        starting_flows = self.data["dams"][idx].get("starting_flows")
        if starting_flows is not None:
            starting_flows = copy(starting_flows)

        return starting_flows

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

    def get_flow_limit_obs_for_channel(self, idx: str) -> dict[str, list[float]]:

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

    def get_turbined_flow_obs_for_power_group(self, idx: str) -> dict[str, list[float]]:

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

        information_horizon = self.get_information_horizon()
        if time + num_steps - 1 >= information_horizon:
            warnings.warn(
                f"Tried to access unregulated flow for "
                f"time + num_steps - 1 = {time} + {num_steps} - 1 = {time + num_steps - 1}, "
                f"which is equal or greater than {information_horizon=}. "
                f"None was returned"
            )
            return None

        unreg_flows = self.data["dams"][idx]["unregulated_flows"][
            time: time + num_steps
        ]
        if num_steps == 1:
            unreg_flows = unreg_flows[0]

        return unreg_flows
    
    def get_all_unregulated_flows_of_dam(self, idx: str) -> list[float]:
        
        """
        
        :return: All unregulated flow that enters the dam (flow that comes from the river)
        within the decision horizon (m3/s)
        """
        
        unreg_flows = self.data["dams"][idx]["unregulated_flows"]
        
        return unreg_flows
    
    def get_all_prices(self) -> list[float]:
        
        """
        
        :return: All the prices of energy (EUR/MWh) in the time bands within the decision horizon
        """
        
        prices = self.data["energy_prices"]
        
        return prices

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

        information_horizon = self.get_information_horizon()
        if time + num_steps - 1 >= information_horizon:
            warnings.warn(
                f"Tried to access incoming flow for "
                f"time + num_steps - 1 = {time} + {num_steps} - 1 = {time + num_steps - 1}, "
                f"which is equal or greater than {information_horizon=}. "
                f"None was returned"
            )
            return None

        incoming_flows = self.data["incoming_flows"][time: time + num_steps]
        if num_steps == 1:
            incoming_flows = incoming_flows[0]

        return incoming_flows
    
    def get_all_incoming_flows(self):
        
        """
        
        :return: All the flows (m3/s) entering the first dam in the time bands within the decision horizon
        """
        
        prices = self.data["incoming_flows"]
        
        return prices

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

        information_horizon = self.get_information_horizon()
        if time + num_steps - 1 >= information_horizon:
            warnings.warn(
                f"Tried to access price for time + num_steps - 1 = {time} + {num_steps} - 1 = {time + num_steps - 1}, "
                f"which is equal or greater than {information_horizon=}. "
                f"None was returned"
            )
            return None

        prices = self.data["energy_prices"][time: time + num_steps]
        if num_steps == 1:
            prices = prices[0]

        return prices

    def get_largest_price(self) -> float:

        """
        Get the largest price value for the instance.
        """

        return max(self.data["energy_prices"])

    def calculate_total_avg_inflow(self) -> float:

        """
        Calculate the total average inflow of the day.
        The total average inflow is calculated by adding the average incoming and unregulated flows
        up to the decision horizon.
        """

        incoming_flows = self.get_all_incoming_flows()[:self.get_decision_horizon()]
        # print("Incoming flows:", incoming_flows)
        # print("Incoming flow mean:", sum(incoming_flows)/len(incoming_flows))
        total_avg_inflow = sum(incoming_flows) / len(incoming_flows)
        for dam_id in self.get_ids_of_dams():
            unreg_flows = self.get_all_unregulated_flows_of_dam(dam_id)[:self.get_decision_horizon()]
            # print(dam_id, "unregulated flows:", unreg_flows)
            # print(dam_id, "unregulated flow mean:", sum(unreg_flows) / len(unreg_flows))
            total_avg_inflow += sum(unreg_flows) / len(unreg_flows)
        # print("Total avg inflow:", total_avg_inflow)

        return total_avg_inflow
