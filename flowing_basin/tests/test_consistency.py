from flowing_basin.solvers.rl import ReinforcementLearning
from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
from unittest import TestCase
import warnings


class TestBaselinesConsistency(TestCase):

    def test_baselines_consistency(self):

        """
        Check that the objective function values of each baseline
        are the same as those predicted by the simulator for the solutions
        """

        for general_config in ["G0", "G1"]:

            for solution in ReinforcementLearning.get_all_baselines(general_config):

                inconsistencies = solution.check()
                self.assertEqual(
                    len(inconsistencies),
                    0,
                    msg=f"In {general_config}, the solution of {solution.get_solver()} "
                        f"for instance {solution.get_instance_name()} "
                        f"has inconsistencies: {inconsistencies}."
                )

                config = solution.get_configuration()
                instance = Instance.from_name(solution.get_instance_name())
                mode = getattr(config, 'mode', 'linear')

                river_basin = RiverBasin(
                    instance=instance,
                    flow_smoothing=config.flow_smoothing,
                    num_scenarios=1,
                    mode=mode,
                    paths_power_models=None,
                    do_history_updates=True,
                    update_to_decisions=False
                )
                river_basin.deep_update_flows(solution.get_flows_array())

                volume_shortage = 0.
                volume_exceedance = 0.

                for dam_id in instance.get_ids_of_dams():

                    dam_index = instance.get_order_of_dam(dam_id) - 1
                    details = {
                        "income_from_energy_eur": river_basin.dams[dam_index].channel.power_group.acc_income.item(),
                        "startups": river_basin.dams[dam_index].channel.power_group.acc_num_startups.item(),
                        "limit_zones": river_basin.dams[dam_index].channel.power_group.acc_num_times_in_limit.item(),
                        "volume_shortage_m3": max(
                            0., config.volume_objectives[dam_id] - river_basin.dams[dam_index].final_volume.item()
                        ),
                        "volume_exceedance_m3": max(
                            0., river_basin.dams[dam_index].final_volume.item() - config.volume_objectives[dam_id]
                        ),
                    }
                    volume_shortage += details["volume_shortage_m3"]
                    volume_exceedance += details["volume_exceedance_m3"]

                    # Check details
                    obj_fun_details = solution.get_objective_details(dam_id)
                    if obj_fun_details is None:
                        warnings.warn(
                            f"In {general_config}, the solution of {solution.get_solver()} "
                            f"for instance {solution.get_instance_name()} "
                            f"does not have objective function details."
                        )
                    else:
                        for detail, value in details.items():
                            self.assertAlmostEqual(
                                obj_fun_details[detail],
                                value,
                                msg=f"In {general_config}, the solution of {solution.get_solver()} "
                                    f"for instance {solution.get_instance_name()} "
                                    f"has a {detail} with value {obj_fun_details[detail]}, "
                                    f"but the simulator predicts the value {value}."
                            )

                # Check global objective function value
                # This test is redundant if details are provided
                # TODO: enforce details in all solvers and remove these lines of code
                income = river_basin.get_acc_income().item()
                penalty = (
                        config.volume_shortage_penalty * volume_shortage
                        + config.startups_penalty * river_basin.get_acc_num_startups()
                        + config.limit_zones_penalty * river_basin.get_acc_num_times_in_limit()
                )
                bonus = config.volume_exceedance_bonus * volume_exceedance
                obj_fun_value = income + bonus - penalty
                self.assertAlmostEqual(
                    solution.get_objective_function(),
                    obj_fun_value,
                    msg=f"In {general_config}, the solution of {solution.get_solver()} "
                        f"for instance {solution.get_instance_name()} "
                        f"has an objective function value of {solution.get_objective_function()}, "
                        f"but the simulator predicts the value {obj_fun_value}."
                )

