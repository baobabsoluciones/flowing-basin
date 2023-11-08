from flowing_basin.core import Instance, Solution, Experiment
from .rl_env import RLConfiguration, RLEnvironment
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BaseModel


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

    def solve(self, model: BaseModel | str, options: dict = None) -> dict:

        """
        Load the given model and use it to solve the instance given in the initialization.

        :param model: The StableBaselines3 model, or a path to it
        :param options: Unused parameter
        """

        if isinstance(model, str):
            model = SAC.load(model)

        # Reset the environment (this allows the `solve` method to be called more than once)
        obs = self.env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = self.env.step(action)

        clipped_flows = {
            dam_id: self.env.river_basin.all_past_clipped_flows.squeeze()[
                self.instance.get_start_information_offset():, self.instance.get_order_of_dam(dam_id) - 1
            ].tolist()
            for dam_id in self.instance.get_ids_of_dams()
        }
        volume = {
            dam_id: self.env.river_basin.all_past_volumes[dam_id].squeeze()[
                self.instance.get_start_information_offset():
            ].tolist()
            for dam_id in self.instance.get_ids_of_dams()
        }

        income = self.env.river_basin.get_acc_income()
        num_startups = self.env.river_basin.get_acc_num_startups()
        num_limit_zones = self.env.river_basin.get_acc_num_times_in_limit()
        obj_fun = income - num_startups * self.config.startups_penalty - num_limit_zones * self.config.limit_zones_penalty

        start_decisions, end_decisions, end_impact, start_info, end_info, solution_datetime = self.get_instance_solution_datetimes()

        self.solution = Solution.from_dict(
            dict(
                instance_datetimes=dict(
                    start=start_decisions,
                    end_decisions=end_decisions,
                    end_impact=end_impact,
                    start_information=start_info,
                    end_information=end_info,
                ),
                solution_datetime=solution_datetime,
                solver="RL",
                time_step_minutes=self.instance.get_time_step_seconds() // 60,
                configuration=self.config.to_dict(),
                objective_function=obj_fun.item(),
                dams=[
                    dict(
                        id=dam_id,
                        flows=clipped_flows[dam_id],
                        volume=volume[dam_id]
                    )
                    for dam_id in self.instance.get_ids_of_dams()
                ],
                price=self.instance.get_all_prices(),
            )
        )

        return dict()
