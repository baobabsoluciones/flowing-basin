from cornflow_client import get_empty_schema, ExperimentCore
from typing import Dict
from pytups import SuperDict
from .instance import Instance
from .solution import Solution
from dataclasses import dataclass, fields
from datetime import datetime


@dataclass(kw_only=True)
class Configuration:

    # Penalty for each power group startup, and
    # for each time step with the turbined flow in a limit zone (in €/occurrence)
    startups_penalty: float
    limit_zones_penalty: float

    # Objective final volumes
    volume_objectives: dict[str, float] = None

    # Penalty for unfulfilling the objective volumes, and the bonus for exceeding them (in €/m3)
    volume_shortage_penalty: float = None
    volume_exceedance_bonus: float = None

    @classmethod
    def from_dict(cls, data: dict):

        # We filter the data dictionary to include only the necessary keys/arguments
        necessary_attributes = {field.name for field in fields(cls) if field.init}
        filtered_data = {attr: val for attr, val in data.items() if attr in necessary_attributes}

        return cls(**filtered_data)


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

    def get_instance_solution_datetimes(self, format_datetime: str = "%Y-%m-%d %H:%M") -> tuple[str, str, str]:

        """
        Get the start and end datetimes of the solved instance,
        as well as the datetime of when it was solved
        """

        start_datetime, end_datetime = self.instance.get_start_end_datetimes()
        start_datetime = start_datetime.strftime(format_datetime)
        end_datetime = end_datetime.strftime(format_datetime)
        solution_datetime = datetime.now().strftime(format_datetime)

        return start_datetime, end_datetime, solution_datetime


