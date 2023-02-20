from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
from dataclasses import dataclass, field
import torch


@dataclass
class DamObservation:

    """
    Class representing the observation of the agent for each dam
    After initialization, the values are flattened into a `list` attribute
    """

    vol: float
    lags: list[float]
    unreg_flows: list[float]
    list: list = field(init=False)

    def __post_init__(self):

        self.list = [self.vol, *self.lags, *self.unreg_flows]


@dataclass
class Observation:

    """
    Class representing an observation of the agent
    After initialization, the values are flattened into a `tensor` attribute
    After calling the normalization method, the normalized tensor is saved in a `normalized` attribute
    """

    prices: list[float]
    incoming_flows: list[float]
    dams: list[DamObservation]
    tensor: torch.Tensor = field(init=False)
    normalized: None | torch.Tensor = field(init=False)

    def __post_init__(self):

        dam_values = [value for dam in self.dams for value in dam.list]
        self.tensor = torch.tensor([*self.prices, *self.incoming_flows, *dam_values])
        self.normalized = None

    def normalize(self, low: "Observation", high: "Observation"):

        """
        Brings the values of the tensor to the range [0, 1]
        """

        self.normalized = (self.tensor - low.tensor) / (high.tensor - low.tensor)


class Environment:

    """
    Class representing the environment for the RL agent
    The class acts as a wrapper of the river basin:
     - It filters and normalizes the river basin's states, turning them into observations
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
        self.obs_low = self.get_obs_lower_limits()
        self.obs_high = self.get_obs_upper_limits()

    def get_obs_lower_limits(self) -> Observation:

        """
        Returns the lower limits (as an Observation object) of the values of the agent's observations
        All attributes are 0, except the volume (for which the minimum volume is given)
        """

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

    def get_obs_upper_limits(self) -> Observation:

        """
        Returns the upper limits (as an Observation object) of the values of the agent's observations
        """

        return Observation(
            prices=[
                max(
                    self.instance.get_price(
                        0, num_steps=self.instance.get_num_time_steps()
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

    def get_observation(self) -> Observation:

        """
        Returns the observation of the agent for the current state of the river basin
        """

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

        obs.normalize(self.obs_low, self.obs_high)

        return obs

    def get_reward(self) -> float:

        """
        Returns the reward that should be given to the agent for the energy generated in the current time step

        The reward is proportional to the income obtained, income (EUR) = price (EUR/MWh) * power (MW) * time_step (h)
        However, the normalized energy price is used, to avoid inconsistencies throughout several episodes
        (in which energy prices are normalized differently)
        """

        price_normalized = self.instance.get_price(self.river_basin.time) / self.obs_high.prices[0]
        power = sum(dam.channel.power_group.power for dam in self.river_basin.dams.values())
        time_step_hours = self.instance.get_time_step() / 3600

        return price_normalized * power * time_step_hours

    def step(self, action: torch.Tensor) -> tuple[float, Observation, bool]:

        """
        Updates the river basin with the given action
        Returns the reward obtained, the next observation, and whether the episode is finished or not
        """

        assert list(action.size()) == [self.instance.get_num_dams()]

        flows = {dam_id: flow for dam_id, flow in zip(self.instance.get_ids_of_dams(), action.tolist())}
        self.river_basin.update(flows)

        reward = self.get_reward()
        next_obs = self.get_observation()
        done = self.river_basin.time >= self.instance.get_num_time_steps()

        return reward, next_obs, done
