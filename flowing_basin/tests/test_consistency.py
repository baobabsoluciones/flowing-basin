from flowing_basin.solvers.rl import ReinforcementLearning
from flowing_basin.core import Instance, Solution
from flowing_basin.tools import RiverBasin
from unittest import TestCase
import warnings


class TestBaselinesConsistency(TestCase):

    def setUp(self) -> None:

        solutions = {
            sol for general_config in ["G0", "G1"] for sol in ReinforcementLearning.get_all_baselines(general_config)
        }
        solvers = {sol.get_solver() for sol in solutions}
        self.solutions = {solver: [sol for sol in solutions if sol.get_solver() == solver] for solver in solvers}

    def check_consistency(self, solution: Solution):

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
            if sol_details is None:
                warnings.warn(
                    f"The solution of {solution.get_solver()} with config {solution.get_configuration().to_dict()} "
                    f"for instance {solution.get_instance_name()} "
                    f"does not have objective function details."
                )
            else:
                sim_details = river_basin.get_objective_function_details(dam_id, config=config)
                sim_details = {detail_key: detail_value.item() for detail_key, detail_value in sim_details.items()}
                for detail, value in sim_details.items():
                    self.assertAlmostEqual(
                        sol_details[detail],
                        value,
                        places=2,
                        msg=f"The solution of {solution.get_solver()} with config {solution.get_configuration().to_dict()} "
                            f"for instance {solution.get_instance_name()} "
                            f"has a {detail} with value {sol_details[detail]}, "
                            f"but the simulator predicts the value {value}."
                    )

        # Check global objective function value
        # This test is redundant if details are provided
        # TODO: enforce details in all solvers and remove these lines of code
        sol_obj_fun = solution.get_objective_function()
        sim_obj_fun = river_basin.get_objective_function_value(config=config).item()
        self.assertAlmostEqual(
            sol_obj_fun,
            sim_obj_fun,
            places=2,
            msg=f"The solution of {solution.get_solver()} with config {solution.get_configuration().to_dict()} "
                f"for instance {solution.get_instance_name()} "
                f"has an objective function value of {sol_obj_fun}, "
                f"but the simulator predicts the value {sim_obj_fun}."
        )

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

