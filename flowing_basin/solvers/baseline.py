import os
from copy import deepcopy
from datetime import datetime
from dataclasses import fields
from cornflow_client.core.tools import load_json
from flowing_basin.core import Instance, Solution, Configuration
from flowing_basin.solvers.rl import GeneralConfiguration
from flowing_basin.solvers import (
    Heuristic, HeuristicConfiguration, LPModel, LPConfiguration, PSO, PSOConfiguration,
    PsoRbo, PsoRboConfiguration
)
import optuna
from optuna.trial import Trial


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
    hyperparams_values_folder = os.path.join(os.path.dirname(__file__), "../hyperparams/values")
    hyperparams_bounds_folder = os.path.join(os.path.dirname(__file__), "../hyperparams/tuning_bounds")
    hyperparams_best_folder = os.path.join(os.path.dirname(__file__), "../hyperparams/tuning_best")
    config_info = (os.path.join(os.path.dirname(__file__), "../rl_data/configs/general"), GeneralConfiguration)
    baselines_folder = os.path.join(os.path.dirname(__file__), "../rl_data/baselines")
    baselines_filename = "instance{instance_name}_{solver}.json"
    instance_names_eval = [f"Percentile{percentile:02}" for percentile in range(0, 110, 10)]
    instance_names_tune = ["Percentile25", "Percentile75"]

    def __init__(
            self, general_config: str, solver: str, verbose: int = 1,
            max_time: float = None, tuned_hyperparams: bool = False
    ):

        self.general_config = general_config
        self.solver = solver
        self.verbose = verbose
        self.solver_class, config_class = Baseline.solver_classes[self.solver]

        general_config_dict = Baseline.get_general_config_dict(self.general_config)
        general_config_obj = GeneralConfiguration.from_dict(general_config_dict)
        self.num_dams = general_config_obj.num_dams

        if not tuned_hyperparams:
            solver_config_path = os.path.join(Baseline.hyperparams_values_folder, f"{self.solver.lower()}.json")
            solver_config_dict = load_json(solver_config_path)
            Baseline.copy_missing_values(solver_config_dict, general_config_dict)
            self.config = config_class.from_dict(solver_config_dict)

        else:
            path_config = os.path.join(Baseline.hyperparams_best_folder, self.general_config, f"{self.solver}.json")
            self.config = config_class.from_json(path_config)

        if any(field.name == "max_time" for field in fields(self.config)) and max_time is not None:
            self.config.max_time = max_time

    def log(self, msg: str, verbose: int = 0):
        """Print the given message."""
        if self.verbose > verbose:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            print(f"{current_time} [Solver {self.solver}] [Configuration {self.general_config}] {msg}")

    def solve_instance(self, instance: Instance, config: Configuration = None, sol_path: str = None) -> Solution:
        """
        Solve the given instance. Saves the solution in the given path.
        :return: The Solution of the given instance
        """

        if config is None:
            config = self.config
        solver = self.solver_class(instance=instance, config=config)
        instance_name = instance.get_instance_name()
        msg_header = f"[Instance {instance_name}]"

        self.log(f"{msg_header} Starting to calculate the solution...")
        solver.solve()

        sol_inconsistencies = solver.solution.check()
        if sol_inconsistencies:
            raise Exception(f"There are inconsistencies in the calculated solution: {sol_inconsistencies}")

        solver.solution.data["instance_name"] = instance_name
        self.log(
            f"{msg_header} Finished calculating solution with "
            f"objective function value {solver.solution.get_objective_function()}"
        )

        if sol_path is not None:
            solver.solution.to_json(sol_path)
            self.log(f"{msg_header} Saved solution in path {sol_path}")

        return solver.solution

    def solve(self, instance_names: list[str] = None):
        """
        Solve each instance and save it in the corresponding baselines folder.
        """

        if instance_names is None:
            instance_names = Baseline.instance_names_eval

        for instance_name in instance_names:
            instance = Instance.from_name(instance_name, num_dams=self.num_dams)
            sol_filename = Baseline.baselines_filename.format(instance_name=instance_name, solver=self.solver)
            sol_path = os.path.join(Baseline.baselines_folder, self.general_config, sol_filename)
            self.solve_instance(instance, sol_path=sol_path)

    def tune(self, num_trials: int, instance_names: list[str] = None, num_replications: int = 5):
        """
        Use Optuna to find tuned hyperparameters
        """

        if instance_names is None:
            instance_names = Baseline.instance_names_tune

        path_bounds = os.path.join(Baseline.hyperparams_bounds_folder, f"{self.solver}.json")
        hyperparams_bounds = load_json(path_bounds)

        def get_instance_objective(config: Configuration, instance_name: str) -> float:
            """Calculate the objective function value for the given instance and normalize it"""
            instance = Instance.from_name(instance_name, num_dams=self.num_dams)
            solution = self.solve_instance(instance, config=config)
            objective_val = solution.get_objective_function()
            avg_inflow = instance.calculate_total_avg_inflow()
            avg_price = instance.get_avg_price()
            norm_objective_val = objective_val / (avg_inflow * avg_price)
            self.log(
                f"[Instance {instance_name}] Normalized objective function value: "
                f"{objective_val} / ({avg_inflow} * {avg_price}) = {norm_objective_val}"
            )
            return norm_objective_val

        def get_replication_objectice(config: Configuration, replication_num: int) -> float:
            """Calculate the objective function value for a single replication"""
            replication_objective = 0.
            for instance_name in instance_names:
                replication_objective += get_instance_objective(config=config, instance_name=instance_name)
            replication_objective /= len(instance_names)
            self.log(
                f"[Replication {replication_num}] Average normalized objective function value: {replication_objective}"
            )
            return replication_objective

        def get_trial_objective(config: Configuration, trial_num: int) -> float:
            """Calculate the objective function value for a trial"""
            trial_objective = 0.
            for replication_num in range(num_replications):
                trial_objective += get_replication_objectice(config=config, replication_num=replication_num)
            trial_objective /= num_replications
            self.log(
                f"[Trial {trial_num}] Total average normalized objective function value: {trial_objective}"
            )
            return trial_objective

        def objective(trial: Trial):
            # Trial: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            config = deepcopy(self.config)
            for attribute, info in hyperparams_bounds.items():
                # The suggestion of "num_particles" will be done later
                if attribute == "num_particles":
                    continue
                suggest_method = dict(
                    float=trial.suggest_float, int=trial.suggest_int, categorical=trial.suggest_categorical
                )[info['type']]
                value = suggest_method(attribute, **info['values'])
                setattr(config, attribute, value)
            if isinstance(config, PSOConfiguration):
                # Due to performance issues, the max value of "num_particles" is lower with the "pyramid" topology
                values = hyperparams_bounds["num_particles"]['values']
                high = {2: 200, 6: 500}[self.num_dams] if config.topology == "pyramid" else values['high']
                config.num_particles = trial.suggest_int(
                    "num_particles", low=values['low'], high=high, step=values['step']
                )
                # Suggest a "topology_num_neighbors" and a "topology_minkowski_p_norm" whn the topology needs it
                if config.topology == "random" or config.topology == "ring":
                    neighbors_step = 5
                    config.topology_num_neighbors = trial.suggest_int(
                        "topology_num_neighbors",
                        low=neighbors_step, high=int((config.num_particles-2) / neighbors_step) * neighbors_step,
                        step=neighbors_step
                    )
                if config.topology == "ring":
                    config.topology_minkowski_p_norm = trial.suggest_categorical(
                        "topology_minkowski_p_norm", choices=[1, 2]
                    )
            config.__post_init__()
            trial_num = trial.number
            self.log(f"[Trial {trial_num}] Configuration: {config}")
            return get_trial_objective(config=config, trial_num=trial_num)

        # Study: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=num_trials)
        self.log(
            f"Finished hyperparameter tuning. "
            f"Best trial is Trial {study.best_trial.number} with vale {study.best_trial.value}"
        )

        best_config = deepcopy(self.config)
        self.copy_values(best_config, tuned_hyperparams=study.best_params)
        path_config = os.path.join(Baseline.hyperparams_best_folder, self.general_config, f"{self.solver}.json")
        best_config.to_json(path_config)
        self.log(f"Saved best configuration to {path_config}.")

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
    def copy_missing_values(solver_config_dict: dict, general_config_dict: dict):
        """
        Sets the attributes of `solver_config_dict` with missing values to
        the values of these attributes in `general_config_dict`.
        Assumes the attributes with missing values in `solver_config_dict` are all present in `general_config_dict`.
        Only the attributes with missing values in `solver_config_dict` are changed.
        """
        for key, value in solver_config_dict.items():
            if value is None:
                if key not in general_config_dict:
                    raise ValueError(
                        f"The missing attribute {key} is not among the general config attributes: {general_config_dict}"
                    )
                solver_config_dict[key] = general_config_dict[key]

    @staticmethod
    def copy_values(config: Configuration, tuned_hyperparams: dict):
        """
        Copies all values of `tuned_hyperparams` to `config`.
        Assumes all attributes in `tuned_hyperparams` are present in `config`.
        """
        config_attrs = {field.name for field in fields(config)}
        for attribute, value in tuned_hyperparams.items():
            if attribute not in config_attrs:
                raise ValueError(f"The attribute {attribute} is not among the config attributes: {config_attrs}")
            setattr(config, attribute, value)
        config.__post_init__()
