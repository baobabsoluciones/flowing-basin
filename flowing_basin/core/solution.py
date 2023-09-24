from cornflow_client import SolutionCore
from cornflow_client.core.tools import load_json
import os
import pickle
from datetime import datetime
import numpy as np


class Solution(SolutionCore):

    schema = load_json(
        os.path.join(os.path.dirname(__file__), "../schemas/solution.json")
    )

    @classmethod
    def from_dict(cls, data) -> "Solution":

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

    def check_inconsistencies(self, epsilon: float = 1e-6):

        inconsistencies = dict()

        # The number of flows given to all dams must be equal
        first_dam = list(self.data["dams"].values())[0]
        num_flows_first_dam = len(first_dam["flows"])
        if not all([len(dam["flows"]) == num_flows_first_dam for dam in self.data["dams"].values()]):
            inconsistencies.update(
                {
                    "The number of flows given to each dam is not equal":
                    [len(dam['flows']) for dam in self.data['dams'].values()]
                }
            )

        # The objective function history values must all have the same length
        if self.data.get("objective_history") is not None:

            # Check values and time stamps have the same length
            num_obj_values = len(self.data["objective_history"]["objective_values_eur"])
            num_time_stamps = len(self.data["objective_history"]["time_stamps_s"])
            if num_obj_values != num_time_stamps:
                inconsistencies.update(
                    {
                        "The number of objective values in history must equal the number of time stamps":
                            f"There are {num_obj_values} objective values but {num_time_stamps} time stamps."
                    }
                )

            # Check gap values (optional property) and time stamps have the same length
            if self.data["objective_history"].get("gap_values_pct") is not None:
                num_gap_values = len(self.data["objective_history"]["gap_values_pct"])
                if num_gap_values != num_time_stamps:
                    inconsistencies.update(
                        {
                            "The number of gap values in history must equal the number of time stamps":
                                f"There are {num_gap_values} gap values but {num_time_stamps} time stamps."
                        }
                    )

        # If details are provided for any dam, they should be provided for all dams
        has_details = False
        dam_ids_with_detail = [
            dam_id for dam_id in self.get_ids_of_dams()
            if self.data["dams"][dam_id].get("objective_function_details") is not None
        ]
        if len(dam_ids_with_detail) > 0:
            has_details = True
            if dam_ids_with_detail != self.get_ids_of_dams():
                inconsistencies.update(
                    {
                        "Only some of the dams have the objective function details stored":
                            f"All the dams are: {self.get_ids_of_dams()}; "
                            f"but only these dams have details: {dam_ids_with_detail}."
                    }
                )

        # The sum of dam incomes must equal the total objective function value
        total_obj_value = self.data.get("objective_function")
        if has_details and total_obj_value is not None:

            # Compute total income according to details
            total_income = 0.
            for dam_id in self.get_ids_of_dams():
                total_income += self.data["dams"][dam_id]["objective_function_details"]["total_income_eur"]

            # Compare with total objective function value
            if abs(total_income - total_obj_value) > epsilon:
                inconsistencies.update(
                    {
                        "The sum of dam incomes and the total objective function value are not the same":
                            f"The sum of dam incomes is {total_income}, "
                            f"but the total objective function value is {total_obj_value}."
                    }
                )

        # For evey dam, the total income must be equal to the computed income from configuration and details
        inconsistent_dams = []
        config = self.data.get("configuration")
        if has_details and config is not None:
            for dam_id in self.get_ids_of_dams():
                dam_details = self.data["dams"][dam_id]["objective_function_details"]
                total_income = dam_details["total_income_eur"]
                computed_income = (
                    dam_details["income_from_energy_eur"]
                    - dam_details["startups"] * config["startups_penalty"]
                    - dam_details["limit_zones"] * config["limit_zones_penalty"]
                    - dam_details["volume_shortage_m3"] * config["volume_shortage_penalty"]
                    + dam_details["volume_exceedance_m3"] * config["volume_exceedance_bonus"]
                )
                if abs(total_income - computed_income) > epsilon:
                    inconsistent_dams.append((dam_id, total_income, computed_income))
        if len(inconsistent_dams) > 0:
            inconsistencies.update(
                {
                    "The total income of some dams does not equal the computed income using the details and config":
                        [
                            f"The total income of dam {dam_id} is {total_income}, "
                            f"but the computed income from the details is {computed_income}."
                            for dam_id, total_income, computed_income in inconsistent_dams
                        ]
                }
            )

        return inconsistencies

    def check(self):

        """
        Method that checks if the data of the solution does not follow the schema or has inconsistencies
        :return: A dictionary containing the schema noncompliance problems and inconsistencies found
        """

        inconsistencies = self.check_inconsistencies()
        schema_errors = self.check_schema()
        if schema_errors:
            inconsistencies.update(
                {"The given data does not follow the schema": schema_errors}
            )

        return inconsistencies

    def complies_with_flow_smoothing(self, flow_smoothing: int, epsilon: float = 1e-6) -> bool:

        """
        Indicates whether the solution complies with the given flow smoothing parameter or not

        :param flow_smoothing:
        :param epsilon: Small tolerance for rounding errors
        :return:
        """

        compliance = True
        for dam in self.data["dams"].values():
            flows = dam["flows"]
            variations = []
            previous_flow = 0  # We do not have access to the instance, so we assume it is 0
            for flow in flows:
                current_variation = flow - previous_flow
                if any([current_variation * past_variation < - epsilon for past_variation in variations[-flow_smoothing:]]):
                    compliance = False
                variations.append(current_variation)
                previous_flow = flow

        return compliance

    def get_instance_start_end_datetimes(self) -> tuple[datetime, datetime] | None:

        """

        :return: Starting datetime and final datetime (decision horizon) of the solved instance
        """

        instance_datetimes = self.data.get("instance_datetimes")
        if instance_datetimes is not None:
            instance_datetimes = (
                datetime.strptime(instance_datetimes["start"], "%Y-%m-%d %H:%M"),
                datetime.strptime(instance_datetimes["end_decisions"], "%Y-%m-%d %H:%M")
            )

        return instance_datetimes

    def get_solution_datetime(self) -> datetime | None:

        """

        :return: Date and time of when the solution was created
        """

        solution_datetime = self.data.get("solution_datetime")
        if solution_datetime is not None:
            solution_datetime = datetime.strptime(solution_datetime, "%Y-%m-%d %H:%M")

        return solution_datetime

    def get_solver(self) -> str | None:

        """

        :return: Solver used to get the current solution
        """

        return self.data.get("solver")

    def get_ids_of_dams(self) -> list[str]:

        """

        :return: The IDs of all dams in the river basin
        """

        return list(self.data["dams"].keys())

    def get_exiting_flows_of_dam(self, idx: str) -> list[float]:

        """
        Get the assigned flows to the given dam.

        :param idx: ID of the dam in the river basin
        :return: List indicating the flow exiting the reservoir at each point in time (m3/s)
        """

        return self.data["dams"][idx]["flows"]

    def get_objective_function(self) -> float | None:

        """
        Get the objective function value recorded in the solution.

        :return:
        """

        return self.data.get("objective_function")

    def get_volumes_of_dam(self, idx: str) -> list[float] | None:

        """
        Get the predicted volumes of the given dam.

        :param idx: ID of the dam in the river basin
        :return: List indicating the volume of the reservoir at the end of every time step (m3)
        """

        return self.data["dams"][idx].get("volume")

    def get_powers_of_dam(self, idx: str) -> list[float] | None:

        """
        Get the predicted powers of the given dam.

        :param idx: ID of the dam in the river basin
        :return: List indicating the power generated by the power group in every time step (MW)
        """

        return self.data["dams"][idx].get("power")

    @classmethod
    def from_flows_array(cls, flows: np.ndarray, dam_ids: list[str]) -> "Solution":

        """
        Create solution from an array that represents
        the flows that should go through each channel in every time step.

        :param flows:
            Array of shape num_time_steps x num_dams x 1 with
            the flows that should go through each channel in every time step (m3/s)
        :param dam_ids: List with the IDs of the dams of the river basin (e.g. ["dam1", "dam2"])
        """

        # Transpose array, reshaping it from num_time_steps x num_dams x 1, to 1 x num_dams x num_time_steps
        flows_p = np.transpose(flows)

        # Remove first dimension
        flows_p = flows_p[0]

        return cls(
            dict(
                dams=[
                    dict(id=dam_id, flows=flows_p[dam_index].tolist())
                    for dam_index, dam_id in enumerate(dam_ids)
                ]
            )
        )

    def get_exiting_flows_array(self) -> np.ndarray:

        """
        Turn solution into an array containing the assigned flows.

        :return:
            Array of shape num_time_steps x num_dams x 1 with
            the flows that should go through each channel in every time step (m3/s)
        """

        flows_p = [dam["flows"] for dam in self.data["dams"].values()]

        # Transpose array, reshaping it from num_dams x num_time_steps, to num_time_steps x num_dams
        flows = np.transpose(flows_p)

        # Reshape array from num_time_steps x num_dams, to num_time_steps x num_dams x 1
        flows = flows.reshape((-1, len(self.data["dams"]), 1))

        return flows
