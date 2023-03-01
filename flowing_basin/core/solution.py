from cornflow_client import SolutionCore
from cornflow_client.core.tools import load_json
import os
import numpy as np


class Solution(SolutionCore):

    schema = load_json(
        os.path.join(os.path.dirname(__file__), "../schemas/solution.json")
    )

    @classmethod
    def from_nestedlist(cls, flows: list[list[float]], dam_ids: list[str]) -> "Solution":

        """
        Create solution from a nested list that represents
        the flows that should go through each channel in every time step
        """

        # Reshape nested list from num_time_steps x num_dams to num_dams x num_time_steps
        flows_p = np.transpose(flows)

        return cls(
            dict(
                dams=[
                    dict(id=dam_id, flows=flows_p[dam_index].tolist())
                    for dam_index, dam_id in enumerate(dam_ids)
                ]
            )
        )

    def to_nestedlist(self) -> list[list[float]]:

        flows_p = [el["flows"] for el in self.data["dams"]]
        flows = np.transpose(flows_p)

        return flows.tolist()  # noqa (suppres PyCharm inspection)
