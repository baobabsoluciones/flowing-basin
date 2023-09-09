from cornflow_client import SolutionCore
from cornflow_client.core.tools import load_json
import os
import numpy as np


class Solution(SolutionCore):

    schema = load_json(
        os.path.join(os.path.dirname(__file__), "../schemas/solution.json")
    )

    def check_inconsistencies(self):

        inconsistencies = dict()

        # The number of flows given to all dams must be equal
        first_dam = self.data["dams"][0]
        num_flows_first_dam = len(first_dam["flows"])
        if not all([len(dam["flows"]) == num_flows_first_dam for dam in self.data["dams"]]):
            inconsistencies.update(
                {
                    "The number of flows given to each dam is not equal":
                    [len(dam['flows']) for dam in self.data['dams']]
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
        for el in self.data["dams"]:
            flows = el["flows"]
            variations = []
            previous_flow = 0  # We do not have access to the instance, so we assume it is 0
            for flow in flows:
                current_variation = flow - previous_flow
                if any([current_variation * past_variation < - epsilon for past_variation in variations[-flow_smoothing:]]):
                    compliance = False
                variations.append(current_variation)
                previous_flow = flow

        return compliance

    def get_exiting_flows_of_dam(self, idx: str) -> list[float]:

        """
        Get the assigned flows to the given dam.

        :param idx: ID of the dam in the river basin
        :return: List indicating the flow exiting the reservoir at each point in time (m3/s)
        """

        flows = None
        for el in self.data["dams"]:
            if el["id"] == idx:
                flows = el["flows"]

        return flows

    def get_objective_function(self) -> float:

        """
        Get the objective function value recorded in the solution.

        :return:
        """

        return self.data["objective_function"]

    def get_volumes_of_dam(self, idx: str) -> list[float]:

        """
        Get the predicted volumes of the given dam.

        :param idx: ID of the dam in the river basin
        :return: List indicating the volume of the reservoir at the end of every time step (m3)
        """

        vols = None
        for el in self.data["dams"]:
            if el["id"] == idx:
                try:
                    vols = el["volume"]
                except KeyError:
                    raise KeyError("The given Solution object does not have the predicted volumes stored.")

        return vols

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

        flows_p = [el["flows"] for el in self.data["dams"]]

        # Transpose array, reshaping it from num_dams x num_time_steps, to num_time_steps x num_dams
        flows = np.transpose(flows_p)

        # Reshape array from num_time_steps x num_dams, to num_time_steps x num_dams x 1
        flows = flows.reshape((-1, len(self.data["dams"]), 1))

        return flows
