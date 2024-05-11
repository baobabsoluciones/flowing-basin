import os
from flowing_basin.solvers.rl import GeneralConfiguration
from flowing_basin.solvers import (
    Heuristic, HeuristicConfiguration, LPModel, LPConfiguration, PSO, PSOConfiguration,
    PsoRbo, PsoRboConfiguration
)


class Baseline:

    """
    Class to use solvers like MILP or PSO as RL baselines
    """

    solver_classes = {
        "Heuristic": (Heuristic, HeuristicConfiguration),
        "MILP": (LPModel, LPConfiguration),
        "PSO": (PSO, PSOConfiguration),
        "PSO-RBO": (PsoRbo, PsoRboConfiguration)
    }
    config_info = (os.path.join(os.path.dirname(__file__), "../rl_data/configs/general"), GeneralConfiguration)
    baselines_folder = os.path.join(os.path.dirname(__file__), "../rl_data/baselines")
    instance_names = [f"Percentile{percentile:02}" for percentile in range(0, 110, 10)]

    def __init__(self, general_config: str, solver: str):

        general_config = self.get_general_config(general_config)
        self.solver_class, self.config_class = Baseline.solver_classes[solver]
        # TODO: create a self.config attribute using self.config_class with general_config and other values

    @staticmethod
    def get_general_config(general_config: str) -> GeneralConfiguration:

        """
        Get the GeneralConfiguration object from the configuration string (e.g., "G0")
        """

        config_folder, config_class = Baseline.config_info
        config_path = os.path.join(config_folder, general_config + ".json")
        config = config_class.from_json(config_path)
        return config
