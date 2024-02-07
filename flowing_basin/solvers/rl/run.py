from flowing_basin.core import Instance, Solution, Experiment
from flowing_basin.solvers.rl import RLConfiguration, RLEnvironment
from flowing_basin.solvers.rl.feature_extractors import Projector
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
import os


class RLRun(Experiment):

    named_policies = ["random", "greedy"]

    constants_path = os.path.join(os.path.dirname(__file__), "../../data/constants/constants_2dams.json")
    historical_data_path = os.path.join(os.path.dirname(__file__), "../../data/history/historical_data_clean.pickle")

    def __init__(
            self,
            instance: Instance,
            config: RLConfiguration,
            projector: Projector,
            solver_name: str = "rl",
            update_to_decisions: bool = False,
            paths_power_models: dict[str, str] = None,
            solution: Solution = None,
            experiment_id: str = None
    ):
        super().__init__(instance=instance, solution=solution, experiment_id=experiment_id)
        if solution is None:
            self.solution = None

        self.solver_name = solver_name
        self.config = config
        self.rewards = None
        if self.config.action_type == "adjustments":
            self.solutions = None
            self.total_rewards = None

        # Do not use the instance directly; instead,
        # create a fresh instance from the initial datetime of the given instance
        # This guarantees that the information buffer is adapted to the configuration
        self.env = RLEnvironment(
            instance=None,
            initial_row=self.instance.get_start_decisions_datetime(),
            projector=projector,
            config=config,
            paths_power_models=paths_power_models,
            path_constants=RLRun.constants_path,
            path_historical_data=RLRun.historical_data_path,
            update_to_decisions=update_to_decisions,
        )

    def solve(self, policy: BasePolicy | str, options: dict = None) -> dict:

        """
        Load the given model and use it to solve the instance given in the initialization.

        :param policy: A StableBaselines3 policy, a path to a model, or one of the named policies
            ("random" or "greedy"). You can also give "greedy_0.7" to indicate the degree of greediness,
            which the percentage of flow that the greedy agent attempts to assign at each period.
        :param options: Unused parameter
        :return: Dictionary with additional information
        """

        # Define the policy if input is a string
        policy_name = None
        greediness = 1.
        if isinstance(policy, str):
            policy_parts = policy.split("_")
            policy_name = policy_parts[0]
            if policy_name not in RLRun.named_policies:
                # Not a named policy, but a path.
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
            else:
                # A named policy.
                if len(policy_parts) > 1:
                    greediness = float(policy_parts[1])
                    assert 0. <= greediness <= 1., f"Greediness must be a number between 0 and 1, not {greediness}."

        # Reset the environment (this allows the `solve` method to be called more than once)
        # Remember we must not give the instance directly, but rather create a fresh new one for the same day
        obs, _ = self.env.reset(
            instance=None,
            initial_row=self.instance.get_start_decisions_datetime()
        )
        done = False
        self.rewards = []
        if self.config.action_type == "adjustments":
            self.solutions = []

        while not done:

            if self.config.action_type == "adjustments":
                solution = self.get_solution()
                self.solutions.append(solution)

            if policy_name == "random":
                action = self.env.action_space.sample()
            elif policy_name == "greedy":
                action = (greediness * 2 - 1) * self.env.action_space.high  # noqa
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, _, _ = self.env.step(action)
            self.rewards.append(reward)

        self.solution = self.get_solution()
        if self.config.action_type == "adjustments":
            self.solutions.append(self.solution)
            self.total_rewards = self.env.total_rewards

        return dict()

    def get_obj_fun(self) -> float:

        """
        Get the objective function from the current state of the environment
        """

        income = self.env.river_basin.get_acc_income()
        num_startups = self.env.river_basin.get_acc_num_startups()
        num_limit_zones = self.env.river_basin.get_acc_num_times_in_limit()
        obj_fun = income - num_startups * self.config.startups_penalty - num_limit_zones * self.config.limit_zones_penalty
        return obj_fun.item()

    def get_solution(self) -> Solution:

        """
        Get the solution from the current state of the environment
        """

        # We must not include the starting flows read from the data in the solution; only those generated by the agent
        # Note that the correct information offsets are not in self.instance, but in self.env.instance
        start_info_offset = self.env.instance.get_start_information_offset()
        clipped_flows = {
            dam_id: self.env.river_basin.all_past_clipped_flows.squeeze()[
                start_info_offset:, self.instance.get_order_of_dam(dam_id) - 1
            ].tolist()
            for dam_id in self.instance.get_ids_of_dams()
        }
        volume = {
            dam_id: self.env.river_basin.all_past_volumes[dam_id].squeeze()[
                start_info_offset:
            ].tolist()
            for dam_id in self.instance.get_ids_of_dams()
        }

        start_decisions, end_decisions, end_impact, start_info, end_info, solution_datetime = self.get_instance_solution_datetimes()

        solution = Solution.from_dict(
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
                objective_function=self.get_obj_fun(),
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

        return solution
