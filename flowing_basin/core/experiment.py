from cornflow_client import get_empty_schema, ExperimentCore
from typing import Dict
from pytups import SuperDict
from .instance import Instance
from .solution import Solution
from dataclasses import dataclass


@dataclass(kw_only=True)
class Configuration:

    # Objective final volumes
    volume_objectives: dict[str, float]

    # Penalty for unfulfilling the objective volumes, and the bonus for exceeding them (in €/m3)
    volume_shortage_penalty: float
    volume_exceedance_bonus: float

    # Penalty for each power group startup, and
    # for each time step with the turbined flow in a limit zone (in €/occurrence)
    startups_penalty: float
    limit_zones_penalty: float


class Experiment(ExperimentCore):
    schema_checks = get_empty_schema()

    def __init__(self, instance: Instance, solution: Solution = None) -> None:
        ExperimentCore.__init__(self, instance=instance, solution=solution)
        if solution is None:
            self.solution = Solution(SuperDict())

    @property
    def instance(self) -> Instance:
        return super().instance

    @property
    def solution(self) -> Solution:
        return super().solution

    @solution.setter
    def solution(self, value):
        self._solution = value

    def solve(self, options: dict) -> dict:
        raise NotImplementedError()

    def get_objective(self) -> float:
        return 0

    def check_solution(self, *args, **kwargs) -> Dict[str, Dict]:
        return dict()
