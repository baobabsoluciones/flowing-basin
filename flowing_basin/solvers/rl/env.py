from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
from dataclasses import dataclass, field
import torch


@dataclass
class DamObservation:

    """
    Class representing the observation of the agent for each dam
    """

    vol: float
    lags: list[float]
    unreg_flows: list[float]
    tensor: torch.Tensor = field(init=False)

    def __post_init__(self):

        self.tensor = torch.tensor([self.vol, *self.lags, *self.unreg_flows])


@dataclass
class Observation:

    """
    Class representing an observation of the agent (normalized state of the river basin)
    """

    prices: list[float] | float
    incoming_flows: list[float] | float
    dams: list[DamObservation]
    tensor: torch.Tensor = field(init=False)

    def __post_init__(self):

        dam_values = [value for dam in self.dams for value in dam.tensor.tolist()]
        self.tensor = torch.tensor([*self.prices, *self.incoming_flows, *dam_values])

    def normalize(self, low: "Observation", high: "Observation") -> torch.Tensor:

        """
        Brings the values of the tensor to the range [0, 1]
        """

        self.tensor = (self.tensor - low.tensor) / (high.tensor - low.tensor)
        return self.tensor


class Environment:

    """
    Class representing the environment for the RL agent
    The class acts as a wrapper of the river basin:
     - It normalizes the basin's states, turning them into observations
     - It computes the rewards for the agent (proportional to the generated energy and its price)
    """

    def __init__(self, instance: Instance, paths_power_models: dict[str, str], num_prices, num_incoming_flows, num_unreg_flows):

        self.instance = instance
        self.river_basin = RiverBasin(
            instance=self.instance, paths_power_models=paths_power_models
        )

        # Configuration
        self.num_prices = num_prices
        self.num_incoming_flows = num_incoming_flows
        self.num_unreg_flows = num_unreg_flows

        # Attributes for value normalization
        self._low = self._get_lower_limits()
        self._high = self._get_upper_limits()

    def _get_lower_limits(self) -> Observation:

        return Observation(
            prices=[0] * self.num_prices,
            incoming_flows=[0] * self.num_incoming_flows,
            dams=[
                DamObservation(
                    vol=self.instance.get_min_vol_of_dam(dam_id),
                    lags=[0] * self.instance.get_relevant_lags_of_dam(dam_id)[-1],
                    unreg_flows=[0] * self.num_unreg_flows,
                )
                for dam_id, dam in self.river_basin.dams.items()
            ],
        )

    def _get_upper_limits(self) -> Observation:

        return Observation(
            prices=[
                max(
                    self.instance.get_price(
                        0, num_steps=self.instance.get_total_num_time_steps()
                    )
                )
            ]
            * self.num_prices,
            incoming_flows=[self.instance.get_max_incoming_flow()]
            * self.num_incoming_flows,
            dams=[
                DamObservation(
                    vol=self.instance.get_max_vol_of_dam(dam_id),
                    lags=[self.instance.get_max_flow_of_channel(dam_id)]
                    * self.instance.get_relevant_lags_of_dam(dam_id)[-1],
                    unreg_flows=[self.instance.get_max_unregulated_flow_of_dam(dam_id)]
                    * self.num_unreg_flows,
                )
                for dam_id, dam in self.river_basin.dams.items()
            ],
        )

    def get_observation(self, normalize: bool = True) -> torch.Tensor:

        obs = Observation(
            prices=self.instance.get_price(
                self.river_basin.time, num_steps=self.num_prices
            ),
            incoming_flows=self.instance.get_incoming_flow(
                self.river_basin.time, num_steps=self.num_incoming_flows
            ),
            dams=[
                DamObservation(
                    vol=dam.volume,
                    lags=dam.channel.flows_over_time,
                    unreg_flows=self.instance.get_unregulated_flow_of_dam(
                        self.river_basin.time, dam_id, num_steps=self.num_unreg_flows
                    ),
                )
                for dam_id, dam in self.river_basin.dams.items()
            ],
        )

        if normalize:
            obs.normalize(self._low, self._high)

        return obs.tensor
