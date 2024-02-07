from cornflow_client import get_empty_schema, ExperimentCore
from flowing_basin.core.instance import Instance
from flowing_basin.core.solution import Solution
from typing import Dict
from pytups import SuperDict
from datetime import datetime


class Experiment(ExperimentCore):
    schema_checks = get_empty_schema()

    def __init__(self, instance: Instance, solution: Solution = None, experiment_id: str = None) -> None:

        ExperimentCore.__init__(self, instance=instance, solution=solution)

        if solution is None:
            self.solution = Solution(SuperDict())

        self.experiment_id = experiment_id
        if self.experiment_id is None:
            self.experiment_id = datetime.now().strftime('%Y-%m-%d %H.%M')

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


