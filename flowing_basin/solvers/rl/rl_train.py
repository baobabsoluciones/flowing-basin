from flowing_basin.core import Instance, Solution, Experiment
from .rl_env import RLEnvironment, RLConfiguration
from stable_baselines3 import SAC


class RLTrain(Experiment):

    def __init__(
            self,
            config: RLConfiguration,
            path_constants: str,
            path_training_data: str,
            paths_power_models: dict[str, str] = None,
            instance: Instance = None,
            solution: Solution = None,
    ):

        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        self.env = RLEnvironment(
            config=config,
            path_constants=path_constants,
            path_training_data=path_training_data,
            paths_power_models=paths_power_models,
            instance=instance,
        )
        self.model = SAC("MlpPolicy", self.env, verbose=1)

    def solve(self, num_episodes: int, path_agent: str, options: dict = None) -> dict:

        """
        Train the model and save it in the given path.
        """

        total_timesteps = num_episodes * self.env.instance.get_largest_impact_horizon()
        self.model.learn(total_timesteps=total_timesteps, log_interval=5)
        self.model.save(path_agent)

        return dict()


