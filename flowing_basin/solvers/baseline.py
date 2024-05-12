import os
from cornflow_client.core.tools import load_json
from flowing_basin.core import Instance
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
    hyperparams_folder = os.path.join(os.path.dirname(__file__), "../hyperparams")
    config_info = (os.path.join(os.path.dirname(__file__), "../rl_data/configs/general"), GeneralConfiguration)
    baselines_folder = os.path.join(os.path.dirname(__file__), "../rl_data/baselines")
    baselines_filename = "instance{instance_name}_{solver}.json"
    instance_names = [f"Percentile{percentile:02}" for percentile in range(0, 110, 10)]

    def __init__(self, general_config: str, solver: str, verbose: int = 1):

        self.verbose = verbose
        self.solver = solver
        self.solver_class, config_class = Baseline.solver_classes[self.solver]

        solver_config_path = os.path.join(Baseline.hyperparams_folder, f"{self.solver.lower()}.json")
        solver_config_dict = load_json(solver_config_path)

        self.general_config = general_config
        general_config_dict = Baseline.get_general_config_dict(self.general_config)
        general_config_obj = GeneralConfiguration.from_dict(general_config_dict)
        self.num_dams = general_config_obj.num_dams

        Baseline.update_config(solver_config_dict, general_config_dict)
        self.config = config_class.from_dict(solver_config_dict)

    def solve(self):

        """
        Solve each instance and save it in the corresponding baselines folder
        """

        for instance_name in Baseline.instance_names:

            instance = Instance.from_name(instance_name, num_dams=self.num_dams)
            solver = self.solver_class(instance=instance, config=self.config)

            if self.verbose > 0:
                print(f"Using {self.solver} under {self.general_config} to solve instance {instance_name}...")
            solver.solve()

            sol_inconsistencies = solver.solution.check()
            if sol_inconsistencies:
                raise Exception(f"There are inconsistencies in the given solution: {sol_inconsistencies}")
            solver.solution.data["instance_name"] = instance_name

            sol_filename = Baseline.baselines_filename.format(instance_name=instance_name, solver=self.solver)
            sol_path = os.path.join(Baseline.baselines_folder, self.general_config, sol_filename)
            solver.solution.to_json(sol_path)
            if self.verbose > 0:
                print(
                    f"Saved solution of {self.solver} under {self.general_config} "
                    f"for instance {instance_name} in path {sol_path}"
                )

    @staticmethod
    def get_general_config_dict(general_config: str) -> dict:

        """
        Get the GeneralConfiguration dict from the configuration string (e.g., "G0")
        """

        config_folder, config_class = Baseline.config_info
        config_path = os.path.join(config_folder, general_config + ".json")
        config_dict = load_json(config_path)
        return config_dict

    @staticmethod
    def update_config(solver_config_dict: dict, general_config_dict: dict):
        """
        Sets the attributes of `solver_config_dict` with missing values to
        the values of these attributes in `general_config_dict`.
        Assumes the attributes with missing values in `solver_config_dict` are all present in `general_config_dict`.
        Only the attributes with missing values in `solver_config_dict` are changed.
        """
        for key, value in solver_config_dict.items():
            if value is None:
                solver_config_dict[key] = general_config_dict[key]
