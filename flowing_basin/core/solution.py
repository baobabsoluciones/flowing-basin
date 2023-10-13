from cornflow_client import SolutionCore
from cornflow_client.core.tools import load_json
import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import warnings


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

        # Calculate number of decision time steps
        num_decision_time_steps = len(self.get_decisions_time_steps())

        # If the volumes and powers of the dams are provided, their length should be the same as the number of flows
        inconsistent_dams = dict(flows=[], volumes=[], powers=[])
        for dam_id in self.get_ids_of_dams():

            flows = self.get_exiting_flows_of_dam(dam_id)
            if len(flows) != num_decision_time_steps:
                inconsistent_dams["flows"].append((dam_id, len(flows)))

            vols = self.data["dams"][dam_id].get("volume")
            if vols is not None:
                if len(vols) != num_decision_time_steps:
                    inconsistent_dams["volumes"].append((dam_id, len(vols)))

            powers = self.data["dams"][dam_id].get("power")
            if powers is not None:
                if len(powers) != num_decision_time_steps:
                    inconsistent_dams["powers"].append((dam_id, len(powers)))

        for concept in inconsistent_dams.keys():
            if len(inconsistent_dams[concept]) != 0:
                inconsistencies.update(
                    {
                        f"The number of {concept} of some dams does not equal the number of decision time steps":
                            [
                                f"The number of {concept} in dam {dam_id} is {num_of_concept}, "
                                f"but the number of decision time steps is {num_decision_time_steps}."
                                for dam_id, num_of_concept in inconsistent_dams[concept]
                            ]
                    }
                )

        # The number of prices provided should be the same as the number of information time steps
        num_info_time_steps = len(self.get_information_time_steps())
        num_prices = len(self.data["price"])
        if num_prices != num_info_time_steps:
            inconsistencies.update(
                {
                    f"The number of prices is not equal to the number of information time steps":
                        f"The number of prices is {num_prices}, "
                        f"and the number of information time steps is {num_info_time_steps}."
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
        for dam_id in self.get_ids_of_dams():
            flows = self.get_exiting_flows_of_dam(dam_id)
            variations = []
            previous_flow = 0  # We do not have access to the instance, so we assume it is 0
            for flow in flows:
                current_variation = flow - previous_flow
                if any([current_variation * past_variation < - epsilon for past_variation in variations[-flow_smoothing:]]):
                    compliance = False
                variations.append(current_variation)
                previous_flow = flow

        return compliance

    def get_time_step_seconds(self) -> float | None:

        time_step_min = self.data.get("time_step_minutes")
        if time_step_min is None:
            return None

        return time_step_min * 60

    def get_start_decisions_datetime(self) -> datetime | None:

        # Get all datetimes
        datetimes = self.data.get("instance_datetimes")
        if datetimes is None:
            return None

        # Get start decisions datetime
        start_decisions = datetimes.get("start")
        if start_decisions is None:
            return None

        # Turn into string
        start_decisions = datetime.strptime(start_decisions, "%Y-%m-%d %H:%M")
        return start_decisions

    def get_end_decisions_datetime(self) -> datetime | None:

        # Get all datetimes
        datetimes = self.data.get("instance_datetimes")
        if datetimes is None:
            return None

        # Get end decisions datetime
        end_decisions = datetimes.get("end_decisions")
        if end_decisions is None:
            return None

        # Turn into string
        end_decisions = datetime.strptime(end_decisions, "%Y-%m-%d %H:%M")
        return end_decisions

    def get_start_information_datetime(self) -> datetime | None:

        # Get all datetimes
        datetimes = self.data.get("instance_datetimes")
        if datetimes is None:
            return None

        # Get start information datetime
        # If this information is not provided, assume information starts when decisions start
        start_info = datetimes.get("start_information")
        if start_info is None:
            return self.get_start_decisions_datetime()

        # Turn into string
        start_info = datetime.strptime(start_info, "%Y-%m-%d %H:%M")
        return start_info

    def get_end_impact_datetime(self) -> datetime | None:

        # Get all datetimes
        datetimes = self.data.get("instance_datetimes")
        if datetimes is None:
            return None

        # Get end impact datetime
        end_impact = datetimes.get("end_impact")
        if end_impact is None:
            return None

        # Turn into string
        end_impact = datetime.strptime(end_impact, "%Y-%m-%d %H:%M")
        return end_impact

    def get_end_information_datetime(self) -> datetime | None:

        # Get all datetimes
        datetimes = self.data.get("instance_datetimes")
        if datetimes is None:
            return None

        # Get end information datetime
        # If this information is not provided, assume information ends when impact ends
        end_info = datetimes.get("end_information")
        if end_info is None:
            return self.get_end_impact_datetime()

        # Turn into string
        end_info = datetime.strptime(end_info, "%Y-%m-%d %H:%M")
        return end_info

    def get_decisions_time_steps(self) -> list[int]:

        """
        Get the time steps corresponding to the decided flows
        For a one-day instance with a max lag of 4 and 15min long time steps, this will be 0, 1, ..., 98
        """

        # Calculate using datetimes, if provided
        start_decisions = self.get_start_decisions_datetime()
        end_impact = self.get_end_impact_datetime()
        time_step_seconds = self.get_time_step_seconds()
        if start_decisions is not None and end_impact is not None and time_step_seconds is not None:
            difference = end_impact - start_decisions
            num_time_steps = difference.total_seconds() // time_step_seconds + 1
            return list(range(int(num_time_steps)))

        # If datetimes are not provided, calculate using the number of flows of the first dam
        num_flows = len(self.get_exiting_flows_of_dam(self.get_ids_of_dams()[0]))
        return list(range(num_flows))

    def get_information_time_steps(self) -> list[int]:

        # Calculate using datetimes, if provided
        start_decisions = self.get_start_decisions_datetime()
        start_info = self.get_start_information_datetime()
        end_info = self.get_end_information_datetime()
        time_step_seconds = self.get_time_step_seconds()
        if start_decisions is not None and start_info is not None and end_info is not None and time_step_seconds is not None:
            start_info_offset = (start_decisions - start_info).total_seconds() // time_step_seconds
            num_time_steps = (end_info - start_decisions).total_seconds() // time_step_seconds
            return list(range(-int(start_info_offset), int(num_time_steps) + 1))

        # If datetimes are not provided, assume information time steps equal decision time steps
        return self.get_decisions_time_steps()

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

    def get_history_time_stamps(self) -> list[float] | None:

        """
        Get the time stamps for the history of objective function values (or gap values)
        """

        time_stamps = self.data.get("objective_history")
        if time_stamps is not None:
            time_stamps = time_stamps["time_stamps_s"]

        return time_stamps

    def get_history_values(self) -> list[float] | None:

        """
        Get the history of objective function values
        """

        values = self.data.get("objective_history")
        if values is not None:
            values = values["objective_values_eur"]

        return values

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

    def get_all_prices(self) -> list[float] | None:

        """

        :return: The price of energy (EUR/MWh) in all time steps of the solved instance
        """

        return self.data.get("price")

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
                dams={
                    dam_id: dict(id=dam_id, flows=flows_p[dam_index].tolist())
                    for dam_index, dam_id in enumerate(dam_ids)
                }
            )
        )

    def get_flows_array(self) -> np.ndarray:

        """
        Turn solution into an array containing the assigned flows.

        :return:
            Array of shape num_time_steps x num_dams x 1 with
            the flows that should go through each channel in every time step (m3/s)
        """

        flows_p = [self.get_exiting_flows_of_dam(dam_id) for dam_id in self.get_ids_of_dams()]

        # Transpose array, reshaping it from num_dams x num_time_steps, to num_time_steps x num_dams
        flows = np.transpose(flows_p)

        # Reshape array from num_time_steps x num_dams, to num_time_steps x num_dams x 1
        flows = flows.reshape((-1, self.get_num_dams(), 1))

        return flows

    def plot_solution_for_dam(self, dam_id: str, ax: plt.Axes):

        """
        Plot the exiting flow and volume of the dam at each time step,
        on top of the graph of the price of energy
        """

        decision_time_steps = self.get_decisions_time_steps()
        info_time_steps = self.get_information_time_steps()

        flows = self.get_exiting_flows_of_dam(dam_id)
        volumes = self.get_volumes_of_dam(dam_id)

        ax.plot(decision_time_steps, volumes, color='b', label="Predicted volume")
        ax.set_xlabel("Time (15min)")
        ax.set_ylabel("Volume (m3)")
        ax.set_title(f"Solution for {dam_id}")
        ax.legend()

        twinax = ax.twinx()
        twinax.plot(info_time_steps, self.get_all_prices(), color='r', label="Price")
        twinax.plot(decision_time_steps, flows, color='g', label="Flow")
        twinax.set_ylabel("Flow (m3/s), Price (€)")
        twinax.legend()

    def plot_objective_values(self, ax: plt.Axes, **kwargs):

        """
        Plot history's objective function values
        """

        time_stamps = self.get_history_time_stamps()
        values = self.get_history_values()
        if time_stamps is None or values is None:
            warnings.warn("This solution object does not have the time stamps or objective function values recorded.")
            return

        ax.plot(time_stamps, values, **kwargs)
