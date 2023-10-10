from flowing_basin.core import Instance, Solution, Experiment
from .rl_env import RLConfiguration, RLEnvironment
from stable_baselines3 import SAC
from dataclasses import asdict


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

        self.config = config
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

        income = self.env.river_basin.get_acc_income()
        num_startups = self.env.river_basin.get_acc_num_startups()
        num_limit_zones = self.env.river_basin.get_acc_num_times_in_limit()
        obj_fun = income - num_startups * self.config.startups_penalty - num_limit_zones * self.config.limit_zones_penalty

        start_datetime, end_datetime, solution_datetime = self.get_instance_solution_datetimes()

        self.solution = Solution.from_dict(
            dict(
                instance_datetimes=dict(
                    start=start_datetime,
                    end_decisions=end_datetime
                ),
                solution_datetime=solution_datetime,
                solver="RL",
                configuration=asdict(self.config),
                objective_function=obj_fun.item(),
                dams=[
                    dict(
                        id=dam_id,
                        flows=self.env.river_basin.all_past_flows.squeeze()[:, self.instance.get_order_of_dam(dam_id) - 1].tolist(),
                    )
                    for dam_id in self.instance.get_ids_of_dams()
                ],
                price=self.instance.get_all_prices(),
            )
        )

        return dict()
