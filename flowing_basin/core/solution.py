from cornflow_client import SolutionCore
from cornflow_client.core.tools import load_json
import os
import numpy as np


class Solution(SolutionCore):

    schema = load_json(
        os.path.join(os.path.dirname(__file__), "../schemas/solution.json")
    )

    @classmethod
    def from_flows(cls, flows: np.ndarray, dam_ids: list[str]) -> "Solution":

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

    def to_flows(self) -> np.ndarray:

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
