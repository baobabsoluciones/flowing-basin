from flowing_basin.core import Instance, Solution, Experiment
from flowing_basin.solvers.rl import RLConfiguration, RLEnvironment
from flowing_basin.solvers.rl.feature_extractors import Projector
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy


class RLRun(Experiment):

    named_policies = ["random", "greedy"]

    def __init__(
            self,
            instance: Instance,
            config: RLConfiguration,
            projector: Projector,
            solver_name: str = "rl",
            paths_power_models: dict[str, str] = None,
            solution: Solution = None,
            experiment_id: str = None
    ):
        super().__init__(instance=instance, solution=solution, experiment_id=experiment_id)
        if solution is None:
            self.solution = None

        self.solver_name = solver_name
        self.config = config
        self.env = RLEnvironment(
            instance=self.instance,
            projector=projector,
            config=config,
            paths_power_models=paths_power_models,
        )

    def solve(self, policy: BasePolicy | str, options: dict = None) -> dict:

        """
        Load the given model and use it to solve the instance given in the initialization.

        :param policy: A StableBaselines3 policy, a path to a model, or one of the named policies ("random" or "greedy")
        :param options: Unused parameter
        :return: Dictionary with additional information
        """

        if isinstance(policy, str) and policy not in RLRun.named_policies:
            # To avoid a KeyError, you must indicate the env and its observation_space and action_space
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/1682#issuecomment-1813338493
            policy = SAC.load(
                policy,
                env=self.env,
                custom_objects={
                    'observation_space': self.env.observation_space,
                    'action_space': self.env.action_space
                }
            ).policy

        # Reset the environment (this allows the `solve` method to be called more than once)
        obs, _ = self.env.reset(self.instance)
        done = False
        rewards = []
        while not done:
            if policy == "random":
                action = self.env.action_space.sample()
            elif policy == "greedy":
                action = self.env.action_space.high  # noqa
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, _, _ = self.env.step(action)
            rewards.append(reward)

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
                instance_name=self.instance.get_instance_name(),
                instance_datetimes=dict(
                    start=start_decisions,
                    end_decisions=end_decisions,
                    end_impact=end_impact,
                    start_information=start_info,
                    end_information=end_info,
                ),
                solution_datetime=solution_datetime,
                solver=self.solver_name,
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

        return dict(rewards=rewards)
