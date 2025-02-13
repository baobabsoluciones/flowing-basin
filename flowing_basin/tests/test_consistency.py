from flowing_basin.solvers.rl import ReinforcementLearning
from flowing_basin.core import Instance, Solution, Configuration
from flowing_basin.tools import RiverBasin
from unittest import TestCase
import warnings
import random


# General configurations to test
GENERAL_CONFIGS = ['G0', 'G01', 'G1', 'G2', 'G21', 'G3', 'G9']


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
        instance = Instance.from_name(solution.get_instance_name(), num_dams=solution.get_num_dams())
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
            max_relvar=1.,
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
            (general_config, sol) for general_config in GENERAL_CONFIGS
            for sol in ReinforcementLearning.get_all_baselines(general_config)
        }
        solvers = {sol[1].get_solver() for sol in solutions}
        self.solutions = {solver: [sol for sol in solutions if sol[1].get_solver() == solver] for solver in solvers}

    def test_solver_consistency(self, solver: str):
        """Test the consistency of the given solver."""
        self.assertNotEqual(len(self.solutions[solver]), 0, msg=f"No {solver} solutions.")
        for general_config, sol in self.solutions[solver]:
            self.check_consistency(sol)
            print(
                f"[Solver {solver}] [Configuration {general_config}] [Instance {sol.get_instance_name()}] "
                f"Successfully checked the consistency of the solution."
            )

    def test_milp_consistency(self):
        self.test_solver_consistency('MILP')

    def test_pso_consistency(self):
        self.test_solver_consistency('PSO')

    def test_heuristic_consistency(self):
        self.test_solver_consistency('Heuristic')

    def test_rl_greedy_consistency(self):
        self.test_solver_consistency('rl-greedy')

    def test_rl_random_consistency(self):
        self.test_solver_consistency('rl-random')

    def test_config_concordance(self):

        """
        Test that all general configurations in each environment (real G0 or simplified G1)
        is the same.
        :return:
        """

        def flatten_dict(d: dict):
            """Flattens a dictionary for one layer."""
            d_flattened = dict()
            for k_outer, v_outer in d.items():
                if isinstance(v_outer, dict):
                    for k_inner, v_inner in v_outer.items():
                        d_flattened[str(k_outer) + "_" + str(k_inner)] = v_inner
                else:
                    d_flattened[k_outer] = v_outer
            return d_flattened

        for general_config in GENERAL_CONFIGS:

            sols = ReinforcementLearning.get_all_baselines(general_config)
            configs = set()

            for sol in sols:

                config = sol.get_configuration().to_dict()

                # Remove any solver-specific configs by creating a Configuration object
                config_filtered = Configuration.from_dict(config).to_dict()

                # If there is no exceedance bonus or shortage penalty, volume objectives do not matter
                if config_filtered['volume_exceedance_bonus'] == 0. and config_filtered['volume_shortage_penalty'] == 0.:
                    del config_filtered['volume_objectives']

                # If volume objectives do matter, this inner dictionary must be flattened
                config_flattened = flatten_dict(config_filtered)

                config_hashable = tuple(sorted(config_flattened.items()))
                configs.add(config_hashable)

            configs_str = '\n'.join([str(config) for config in configs])
            self.assertEqual(
                len(configs), 1,
                msg=f"The configurations under {general_config} are not unique: {configs_str}."
            )


class TestAgentsConsistency(TestConsistency):

    """
    Check the consistency of trained RL agents (for different General and Action configurations)
    """

    def setUp(self) -> None:

        # Only different Action and General configurations are susceptible of being inconsistent
        # In addition, take only the Action configs with the first digit different to avoid the test taking too long
        # action_configs = ReinforcementLearning.get_all_configs("A", relevant_digits=1)
        action_configs = ['A1', 'A113', 'A21']
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
            runs = rl.run_agent(ReinforcementLearning.get_all_fixed_instances(rl.config.num_dams))
            for run in runs:
                self.check_consistency(run.solution)
            print(f"Test passed by agent {agent}.")

    def test_imitator_consistency(self, epsilon: float = 0.005):

        """
        When the agent imitates another solution, it should give the same objetive function as that solution
        """

        for agent in self.agents:

            rl = ReinforcementLearning(agent)
            solution = random.choice(rl.get_all_baselines(rl.config_names['G']))
            instance = Instance.from_name(solution.get_instance_name(), num_dams=rl.config.num_dams)
            run = rl.run_imitator(solution=solution, instance=instance)

            # Detect the edge case
            delta_obj_fun = None
            places_obj_fun = 4
            if solution.get_solver() == "MILP":
                sol_value = solution.get_objective_details("dam1")["startups"]
                run_value = run.solution.get_objective_details("dam1")["startups"]
                if abs(sol_value - run_value) == 1:
                    delta_obj_fun = solution.get_configuration().startups_penalty + epsilon
                    places_obj_fun = None
                    warnings.warn(
                        f"Allowing a difference of {delta_obj_fun} in the objective function value due to edge case."
                    )

            self.assertAlmostEqual(
                run.solution.get_objective_function(), solution.get_objective_function(),
                delta=delta_obj_fun, places=places_obj_fun
            )
            print(f"Test passed by agent {agent} "
                  f"imitating {solution.get_solver()} in instance {instance.get_instance_name()}.")
