from dataclasses import dataclass, field, asdict, fields
import json


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
        Turn the dataclass into a JSON-serializable dictionary
        """

        data_json = asdict(self)
        return data_json

    @classmethod
    def from_dict(cls, data: dict):

        # We filter the data dictionary to include only the necessary keys/arguments
        necessary_attributes = {field.name for field in fields(cls) if field.init}
        filtered_data = {attr: val for attr, val in data.items() if attr in necessary_attributes}

        return cls(**filtered_data)  # noqa

    def to_json(self, path: str):

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4, sort_keys=True)

    @classmethod
    def from_json(cls, path: str):

        with open(path, "r") as f:
            data_json = json.load(f)
        return cls(**data_json)  # noqa

    def __post_init__(self):

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

    # Number of periods during which the flow through the channel may not vary
    # in order to change the sense of the flow's change
    flow_smoothing: int = 0