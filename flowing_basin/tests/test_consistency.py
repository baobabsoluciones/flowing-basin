from flowing_basin.solvers import PSO, PSOConfiguration
from flowing_basin.core import Instance, Solution, Configuration
from flowing_basin.tools import RiverBasin
from unittest import TestCase


class TestConsistency(TestCase):

    test_instance_names = [f"Percentile{percentile:02}" for percentile in range(0, 110, 10)]
    configs = {
        "real": {
            "flow_smoothing": 2,
            "limit_zones_penalty": 50.0,
            "startups_penalty": 50.0,
            "volume_objectives": {},
            "volume_exceedance_bonus": 0.0,
            "volume_shortage_penalty": 0.0,
        },
        "simplified": {
            "flow_smoothing": 0,
            "limit_zones_penalty": 0.0,
            "startups_penalty": 0.0,
            "volume_objectives": {},
            "volume_exceedance_bonus": 0.0,
            "volume_shortage_penalty": 0.0,
        }
    }

    def execute_simulator(self, instance: Instance, sol: Solution, config: Configuration):

        try:
            mode = config.__getattribute__("mode")
        except AttributeError:
            mode = "linear"

        river_basin = RiverBasin(
            instance=instance,
            flow_smoothing=config.flow_smoothing,
            num_scenarios=1,
            mode=mode,
            paths_power_models=None,
            do_history_updates=True,
            update_to_decisions=False
        )

        pass

    def test_consistency_pso(self):

        for config_name, config_general in TestConsistency.configs.items():
            for use_relvar in [True, False]:
                pso_config = {
                    **config_general,
                    "mode": "linear",
                    "use_relvars": use_relvar,
                    "max_relvar": 0.5,
                    "bounds_handling": "periodic",
                    "topology": "star",
                    "cognitive_coefficient": 2.905405139888455,
                    "social_coefficient": 0.4232260541405988,
                    "inertia_weight": 0.4424113459034113,
                    "num_particles": 200,
                    "max_iterations": 1,
                }
                pso_config = PSOConfiguration(**pso_config)
                for instance_name in TestConsistency.test_instance_names:
                    instance = Instance.from_name(instance_name)
                    pso = PSO(instance=instance, config=pso_config)
                    pso.solve()
                    pass
