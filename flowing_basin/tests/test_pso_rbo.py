from flowing_basin.tools import RiverBasin
from flowing_basin.core import Instance
from flowing_basin.solvers import Baseline, PsoRbo
from unittest import TestCase
import numpy as np
import random
import warnings


class TestPsoRbo(TestCase):

    """
    Check the methods of the PSO-RBO solver work as expected
    """

    def test_relvars_from_flows(
            self, general_config: str = 'G01', instance_name: str = 'Percentile25', num_dams: int = 2,
            num_scenarios: int = 200, epsilon: float = 1e-4
    ):

        """
        Check PSO-RBO's `relvars_from_flows` method actually produces equivalent relvars.
        :param general_config:
        :param instance_name:
        :param num_dams:
        :param num_scenarios:
        :return:
        """

        instance = Instance.from_name(instance_name, num_dams=num_dams)
        baseline = Baseline(solver='PSO-RBO', general_config=general_config)
        baseline.config.num_particles = num_scenarios
        if baseline.config.max_relvar < 1.:
            warnings.warn(
                f"When max_relvar = {baseline.config.max_relvar} < 1.0, the reconstructed flows "
                f"will not match the original flows. Test skipped."
            )
            return

        pso_rbo = baseline.get_solver(instance=instance, config=baseline.config)
        assert isinstance(pso_rbo, PsoRbo)  # Make PyCharm inspections happy

        river_basin = RiverBasin(
            instance=instance, mode="linear", num_scenarios=num_scenarios, do_history_updates=False
        )

        # Array of shape num_time_steps x num_dams x num_scenarios
        flows = np.array(
            [
                [
                    [
                        random.uniform(0, instance.get_max_flow_of_channel(dam_id))
                        for _ in range(num_scenarios)
                    ]
                    for dam_id in instance.get_ids_of_dams()
                ]
                for _ in range(instance.get_largest_impact_horizon())
            ]
        )

        # Run river basin with random flows
        river_basin.deep_update_flows(flows)
        clipped_flows1 = river_basin.all_past_clipped_flows

        # Run river basin with equivalent relvars
        relvars = pso_rbo.relvars_from_flows(clipped_flows1)
        river_basin.reset()
        river_basin.deep_update_relvars(relvars)
        clipped_flows2 = river_basin.all_past_clipped_flows

        msg = (
            "The flows corresponding to the equivalent relvars are actually different from the original flows."
            f"\nOriginal flows: {clipped_flows1}.\nEquivalent relvars: {relvars}"
            f"\nCorresponding flows: {clipped_flows2}"
        )
        self.assertTrue(
            (np.abs(clipped_flows2 - clipped_flows1) < epsilon).all(),
            msg=msg
        )
