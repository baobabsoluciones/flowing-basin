from cornflow_client import ApplicationCore, get_empty_schema
from flowing_basin.core import Instance, Solution


class RiverBasinProblem(ApplicationCore):
    name = "river-basin"
    description = "Problem for the river basin prblem with dams and reservoirs"
    instance = Instance
    solution = Solution
    solvers = dict()
    schema = get_empty_schema()

    @property
    def test_cases(self):
        return []
