from cornflow_client import get_empty_schema, ExperimentCore
from typing import Dict
from pytups import SuperDict
from .instance import Instance
from .solution import Solution
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(kw_only=True)
class Configuration:

    # Penalty for each power group startup, and
    # for each time step with the turbined flow in a limit zone (in €/occurrence)
    startups_penalty: float
    limit_zones_penalty: float

    # Objective final volumes
    volume_objectives: dict[str, float] = field(default_factory=lambda: dict())

    # Penalty for unfulfilling the objective volumes, and the bonus for exceeding them (in €/m3)
    volume_shortage_penalty: float = 0.
    volume_exceedance_bonus: float = 0.

    # This is an implementation of the "get" method of dictionaries
    def get(self, k, default=None):
        if hasattr(self, k):
            return getattr(self, k)
        else:
            return default


class Experiment(ExperimentCore):
    schema_checks = get_empty_schema()

    def __init__(self, instance: Instance, solution: Solution = None) -> None:
        ExperimentCore.__init__(self, instance=instance, solution=solution)
        if solution is None:
            self.solution = Solution(SuperDict())

    @property
    def instance(self) -> Instance:
        return super().instance  # noqa

    @property
    def solution(self) -> Solution:
        return super().solution  # noqa

    @solution.setter
    def solution(self, value):
        self._solution = value

    def solve(self, options: dict) -> dict:
        raise NotImplementedError()

    def get_objective(self) -> float:
        return 0

    def check_solution(self, *args, **kwargs) -> Dict[str, Dict]:
        return dict()

    def get_instance_solution_datetimes(self, format_datetime: str = "%Y-%m-%d %H:%M") -> tuple[str, str, str, str, str, str]:

        """
        Get the decision and information start end datetimes of the solved instance,
        as well as the datetime of when it was solved
        """

        start_decisions = self.instance.get_start_decisions_datetime().strftime(format_datetime)
        end_decisions = self.instance.get_end_decisions_datetime().strftime(format_datetime)

        end_impact = self.instance.get_end_impact_datetime().strftime(format_datetime)

        start_information = self.instance.get_start_information_datetime().strftime(format_datetime)
        end_information = self.instance.get_end_information_datetime().strftime(format_datetime)

        solution_datetime = datetime.now().strftime(format_datetime)

        return start_decisions, end_decisions, end_impact, start_information, end_information, solution_datetime


