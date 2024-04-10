import numpy as np

from flowing_basin.core import Instance, Solution, Experiment
from flowing_basin.solvers.rl import RLConfiguration, RLEnvironment
from flowing_basin.solvers.rl.feature_extractors import Projector
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
        self.rewards_per_step = None
        self.rewards_per_period = None
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
            config=self.config,
            paths_power_models=paths_power_models,
            path_constants=RLRun.constants_path,
            path_historical_data=RLRun.historical_data_path,
            update_to_decisions=update_to_decisions,
        )

    def solve(self, policy: BasePolicy | str | Solution, skip_rewards: bool = False, options: dict = None) -> dict:

        """
        Load the given model and use it to solve the instance given in the initialization.

        :param policy: A StableBaselines3 policy, one of the named policies ("random" or "greedy"),
            or a solution to imitate.
            If you use "greedy", you can also give "greedy_0.7" to indicate the degree of greediness,
            which the percentage of flow that the greedy agent attempts to assign at each period.
        :param skip_rewards: Whether to skip reward calculation to accelerate inference
        :param options: Unused parameter
        :return: Dictionary with additional information
        """

        # Define the policy if input is a string
        policy_name = None
        greediness = 1.
        if isinstance(policy, str):
            # A named policy
            policy_parts = policy.split("_")
            policy_name = policy_parts[0]
            if policy_name not in RLRun.named_policies:
                raise ValueError(
                    f"The given policy, '{policy_name}', is not a named policy (it is not in {RLRun.named_policies}). "
                    f"Please give a SB3 policy or a valid named policy."
                )
            if len(policy_parts) > 1:
                greediness = float(policy_parts[1])
                assert 0. <= greediness <= 1., f"Greediness must be a number between 0 and 1, not {greediness}."

        # Define the policy if input is a Solution
        update_as_flows = False
        i = None
        actions = None
        if isinstance(policy, Solution):
            update_as_flows = True
            actions = np.array([
                np.array(policy.get_exiting_flows_of_dam(dam_id)) / self.instance.get_max_flow_of_channel(dam_id)
                for dam_id in self.instance.get_ids_of_dams()
            ])
            actions = 2. * actions - 1.  # from the range [0, 1] to the range [-1, 1]
            actions = np.transpose(actions, (1, 0))  # from (dam, period) to (period, dam)
            actions = actions.reshape((-1, self.config.num_actions_block * self.instance.get_num_dams()))
            i = 0

        # Reset the environment (this allows the `solve` method to be called more than once)
        # Remember we must not give the instance directly, but rather create a fresh new one for the same day
        obs, _ = self.env.reset(
            instance=None,
            initial_row=self.instance.get_start_decisions_datetime()
        )
        done = False
        self.rewards_per_step = []
        self.rewards_per_period = []
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
            elif isinstance(policy, Solution):
                action = actions[i]
                i += 1
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.env.step(
                action, update_as_flows=update_as_flows, skip_rewards=skip_rewards
            )
            self.rewards_per_step.append(reward)
            self.rewards_per_period.extend(info['rewards'])

        self.solution = self.get_solution()
        if self.config.action_type == "adjustments":
            self.solutions.append(self.solution)
            self.total_rewards = self.env.total_rewards

        return dict()

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
        power = {
            dam_id: self.env.river_basin.all_past_powers[dam_id].squeeze()[
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
                objective_function=self.env.river_basin.get_objective_function_value(config=self.config).item(),
                dams=[
                    dict(
                        id=dam_id,
                        flows=clipped_flows[dam_id],
                        volume=volume[dam_id],
                        power=power[dam_id],
                        objective_function_details={
                            detail_key: detail_value.item()
                            for detail_key, detail_value in self.env.river_basin.get_objective_function_details(
                                dam_id, config=self.config
                            ).items()
                        }
                    )
                    for dam_id in self.instance.get_ids_of_dams()
                ],
                price=self.instance.get_all_prices(),
            )
        )

        return solution
