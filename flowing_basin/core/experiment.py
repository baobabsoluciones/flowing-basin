from cornflow_client import get_empty_schema, ExperimentCore
from flowing_basin.core.instance import Instance
from flowing_basin.core.solution import Solution
from flowing_basin.core.config import Configuration
from typing import Dict
from pytups import SuperDict
from datetime import datetime


class Experiment(ExperimentCore):
    schema_checks = get_empty_schema()

    def __init__(
            self, instance: Instance, config: Configuration, solution: Solution = None, experiment_id: str = None
    ) -> None:

        ExperimentCore.__init__(self, instance=instance, solution=solution)

        self.config = config

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

    def get_instance_solution_datetimes(
            self, format_datetime: str = "%Y-%m-%d %H:%M", instance: Instance = None
    ) -> tuple[str, str, str, str, str, str]:
        """
        Get the decision and information start end datetimes of the solved instance,
        as well as the datetime of when it was solved
        """
        if instance is None:
            instance = self.instance
        return instance.get_instance_current_datetimes(format_datetime=format_datetime)


