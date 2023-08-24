from flowing_basin.core import Instance, Solution, Experiment
from .rl_env import RLConfiguration, RLEnvironment
from stable_baselines3 import SAC
import numpy as np


class RLRun(Experiment):

    def __init__(
            self,
            instance: Instance,
            config: RLConfiguration,
            paths_power_models: dict[str, str] = None,
            solution: Solution = None,
    ):
        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        self.env = RLEnvironment(
            instance=instance,
            config=config,
            paths_power_models=paths_power_models,
        )

    def solve(self, path_agent: str, options: dict = None) -> dict:

        """
        Load the given model and use it to solve the instance given in the initialization.
        """

        model = SAC.load(path_agent)
        obs = self.env.get_observation()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = self.env.step(action)

        self.solution = Solution.from_flows(
            self.env.river_basin.all_past_clipped_flows, dam_ids=self.instance.get_ids_of_dams()
        )

        return dict()
