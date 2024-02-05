from cornflow_client import get_empty_schema, ExperimentCore
from typing import Dict
from pytups import SuperDict
from .instance import Instance
from .solution import Solution
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
import json
from copy import deepcopy


@dataclass(kw_only=True)
class BaseConfiguration:

    # This is an implementation of the "get" method of dictionaries
    def get(self, k, default=None):
        if hasattr(self, k):
            return getattr(self, k)
        else:
            return default

    def to_dict(self) -> dict:

        """
        Turn the original dataclass (before any post-processing) into a JSON-serializable dictionary
        """

        data_json = asdict(self.prior)
        return data_json

    @classmethod
    def from_dict(cls, data: dict):

        # We filter the data dictionary to include only the necessary keys/arguments
        necessary_attributes = {field.name for field in fields(cls) if field.init}
        filtered_data = {attr: val for attr, val in data.items() if attr in necessary_attributes}

        return cls(**filtered_data)

    def to_json(self, path: str):

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4, sort_keys=True)

    @classmethod
    def from_json(cls, path: str):

        with open(path, "r") as f:
            data_json = json.load(f)
        return cls(**data_json)  # noqa

    def __post_init__(self):

        self.check()
        self.prior = deepcopy(self)
        self.post_process()

    def check(self):

        pass

    def post_process(self):

        pass


@dataclass(kw_only=True)
class Configuration(BaseConfiguration):

    # Penalty for each power group startup, and
    # for each time step with the turbined flow in a limit zone (in €/occurrence)
    startups_penalty: float
    limit_zones_penalty: float

    # Objective final volumes
    volume_objectives: dict[str, float] = field(default_factory=lambda: dict())

    # Penalty for unfulfilling the objective volumes, and the bonus for exceeding them (in €/m3)
    volume_shortage_penalty: float = 0.
    volume_exceedance_bonus: float = 0.


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


