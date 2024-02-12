from flowing_basin.solvers.rl import ReinforcementLearning
from flowing_basin.core import Instance, Solution
from flowing_basin.tools import RiverBasin
from unittest import TestCase
import warnings


class TestConsistency(TestCase):

    """
    Template for checking the consistency of solutions
    """

    def check_consistency(self, solution: Solution, epsilon: float = 0.005):

        """
        Check that the objective function values of each baseline
        are the same as those predicted by the simulator for the solutions
        """

        inconsistencies = solution.check()
        self.assertEqual(
            len(inconsistencies),
            0,
            msg=f"The solution of {solution.get_solver()} with config {solution.get_configuration().to_dict()} "
                f"for instance {solution.get_instance_name()} "
                f"has inconsistencies: {inconsistencies}."
        )

        config = solution.get_configuration()
        instance = Instance.from_name(solution.get_instance_name())
        mode = getattr(config, 'mode', 'linear')
        detail_penalty = dict(
            startups=config.startups_penalty,
            limit_zones=config.limit_zones_penalty,
            volume_shortage_m3=config.volume_shortage_penalty,
            volume_exceedance_m3=config.volume_exceedance_bonus
        )

        # By default, force the objective function to match 2 decimal places (see the end of this function)
        delta_obj_fun = None
        places_obj_fun = 2

        # Do not smooth flows. Flow smoothing compliance should be checked separately
        river_basin = RiverBasin(
            instance=instance,
            flow_smoothing=0,
            num_scenarios=1,
            mode=mode,
            paths_power_models=None,
            do_history_updates=True,
            update_to_decisions=False
        )
        river_basin.deep_update_flows(solution.get_flows_array())

        # Check that the details from the simulator are the same as the details from the solution
        for dam_id in instance.get_ids_of_dams():

            sol_details = solution.get_objective_details(dam_id)
            self.assertIsNotNone(
                sol_details,
                msg=f"The solution of {solution.get_solver()} with config {solution.get_configuration().to_dict()} "
                    f"for instance {solution.get_instance_name()} "
                    f"does not have objective function details."
            )

            sim_details = river_basin.get_objective_function_details(dam_id, config=config)
            sim_details = {detail_key: detail_value.item() for detail_key, detail_value in sim_details.items()}

            for detail, value in sim_details.items():

                # Skip those details with no penalty; they are not required to be consistent
                if detail in detail_penalty:
                    if detail_penalty[detail] == 0.:
                        continue

                # Exceptionally, allow a variation of 1 unit of the startups of dam1
                # This is an edge case: since there is a startup flow that matches a shutdown flow in dam1,
                # the MILP can _choose_ whether it is in a limit zone or in a startup depending on what it is better
                # In this case, the objective function can also differ in the value of the penalty
                if dam_id == "dam1" and detail == "startups":
                    if abs(sol_details[detail] - value) == 1:
                        warnings.warn(
                            f"Edge case with {solution.get_solver()} in {solution.get_instance_name()}: "
                            f"The solution of {solution.get_solver()} with config {solution.get_configuration().to_dict()} "
                            f"for instance {solution.get_instance_name()} "
                            f"has a {detail} with value {sol_details[detail]} for dam {dam_id}, "
                            f"but the simulator predicts the value {value}. "
                        )
                        delta_obj_fun = config.startups_penalty + epsilon
                        places_obj_fun = None
                        continue

                self.assertAlmostEqual(
                    sol_details[detail],
                    value,
                    delta=delta_obj_fun if detail == "total_income_eur" else None,
                    places=places_obj_fun if detail == "total_income_eur" else 2,
                    msg=f"The solution of {solution.get_solver()} with config {solution.get_configuration().to_dict()} "
                        f"for instance {solution.get_instance_name()} "
                        f"has a {detail} with value {sol_details[detail]} for dam {dam_id}, "
                        f"but the simulator predicts the value {value}."
                )

        # Check global objective function value
        # This test is redundant if details are provided
        sol_obj_fun = solution.get_objective_function()
        sim_obj_fun = river_basin.get_objective_function_value(config=config).item()
        self.assertAlmostEqual(
            sol_obj_fun,
            sim_obj_fun,
            places=places_obj_fun,
            delta=delta_obj_fun,
            msg=f"The solution of {solution.get_solver()} with config {solution.get_configuration().to_dict()} "
                f"for instance {solution.get_instance_name()} "
                f"has an objective function value of {sol_obj_fun}, "
                f"but the simulator predicts the value {sim_obj_fun}."
        )


class TestBaselinesConsistency(TestConsistency):

    """
    Check the consistency of saved RL baselines
    """

    def setUp(self) -> None:

        solutions = {
            sol for general_config in ["G0", "G1"] for sol in ReinforcementLearning.get_all_baselines(general_config)
        }
        solvers = {sol.get_solver() for sol in solutions}
        self.solutions = {solver: [sol for sol in solutions if sol.get_solver() == solver] for solver in solvers}

    def test_milp_consistency(self):

        self.assertNotEqual(len(self.solutions["MILP"]), 0, msg="No MILP solutions.")
        for sol in self.solutions["MILP"]:
            self.check_consistency(sol)

    def test_pso_consistency(self):

        self.assertNotEqual(len(self.solutions["PSO"]), 0, msg="No PSO solutions.")
        for sol in self.solutions["PSO"]:
            self.check_consistency(sol)

    def test_heuristic_consistency(self):

        self.assertNotEqual(len(self.solutions["Heuristic"]), 0, msg="No Heuristic solutions.")
        for sol in self.solutions["Heuristic"]:
            self.check_consistency(sol)

    def test_rl_greedy_consistency(self):

        self.assertNotEqual(len(self.solutions["rl-greedy"]), 0, msg="No rl-greedy solutions.")
        for sol in self.solutions["rl-greedy"]:
            self.check_consistency(sol)

    def test_rl_random_consistency(self):

        self.assertNotEqual(len(self.solutions["rl-random"]), 0, msg="No rl-random solutions.")
        for sol in self.solutions["rl-random"]:
            self.check_consistency(sol)


class TestAgentsConsistency(TestConsistency):

    """
    Check the consistency of trained RL agents (for different General and Action configurations)
    """

    def setUp(self) -> None:

        # Only different Action and General configurations are susceptible of being inconsistent
        # In addition, take only the Action configs with the first digit different to avoid the test taking too long
        action_configs = ReinforcementLearning.get_all_configs("A", relevant_digits=1)
        general_configs = ReinforcementLearning.get_all_configs("G")

        self.agents = []
        for action_config in action_configs:
            for general_config in general_configs:
                # Get the first agent matching the current Action and General configurations
                agents = ReinforcementLearning.get_all_agents(f".*{action_config}.*{general_config}.*")
                if agents:
                    self.agents.append(agents[0])
        print("Agents to test:", self.agents)

    def test_agents_consistency(self):

        for agent in self.agents:
            rl = ReinforcementLearning(agent)
            for instance in ReinforcementLearning.get_all_fixed_instances():
                run = rl.run_agent(instance)
                self.check_consistency(run.solution)
            print(f"Test passed by agent {agent}.")
