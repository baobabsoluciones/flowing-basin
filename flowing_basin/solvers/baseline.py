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
from flowing_basin.solvers.common import BASELINES_FOLDER, get_all_baselines, get_all_baselines_folder, barchart_instances
import optuna
from optuna.trial import Trial


class Baseline:

    """
    Class to use solvers like MILP or PSO (that may act as RL baselines)
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
    baselines_filename = "instance{instance_name}_{solver}.json"
    instance_names_eval = [f"Percentile{percentile:02}" for percentile in range(0, 110, 10)]
    instance_names_tune = ["Percentile25", "Percentile75"]

    def __init__(
            self, general_config: str, solver: str, verbose: int = 1,
            max_time: float = None, tuned_hyperparams: bool = False
    ):

        """
        Initialize a Baseline object.

        :param general_config: General configuration (e.g., "G1")
        :param solver: Solver that wants to be used (e.g., 'MILP')
        :param verbose: Level of detail of the information printed on screen (if 0, nothing gets printed) (default: 1)
        :param max_time: If given, override the configuration's max_time if there is one
        :tuned_hyperparams: Whether to tuse tuned hyperparameters or not (default: False)
        """

        self.general_config = general_config
        self.solver = solver
        self.verbose = verbose
        self.solver_class, config_class = Baseline.solver_classes[self.solver]

        general_config_dict = Baseline.get_general_config_dict(self.general_config)
        general_config_obj = GeneralConfiguration.from_dict(general_config_dict)
        self.num_dams = general_config_obj.num_dams

        if not tuned_hyperparams:
            # Use hyperparams from `flowing_basin/hyperparams/values` and save sol in `flowing_basin/rl_data/baselines`
            solver_config_path = os.path.join(Baseline.hyperparams_values_folder, f"{self.solver.lower()}.json")
            solver_config_dict = load_json(solver_config_path)
            Baseline.copy_missing_values(solver_config_dict, general_config_dict)
            self.config = config_class.from_dict(solver_config_dict)
            self.sol_folder_name = ""

        else:
            # Use hyperparams from `flowing_basin/hyperparams/tuning_best` and
            # save sol in `flowing_basin/rl_data/baselines/tuned`
            path_config = os.path.join(Baseline.hyperparams_best_folder, self.general_config, f"{self.solver}.json")
            self.config = config_class.from_json(path_config)
            self.sol_folder_name = "tuned"

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

    def solve(self, instance_names: list[str] = None, num_replications: int = None, filename_tail: str = ""):
        """
        Solve each instance and save it in the corresponding baselines folder.
        :param instance_names: Names of the instances to solve
        :param num_replications: Total number of replications (including previous executions of this program)
        :param filename_tail: Whether to add something at the end of each filename
        """

        def add_filename_tail(filename: str, tail: str):
            """Add the given `tail` to the end of the `filename`, without changing the file extension."""
            filename_parts = filename.split('.')
            new_filename = filename_parts[0] + f"_{tail}." + filename_parts[1]
            return new_filename

        def get_sol_path(inst_name: str, repl_num: int = None):
            """Get the path to the solution with the given replication number."""
            sol_filename = Baseline.baselines_filename.format(instance_name=inst_name, solver=self.solver)
            if repl_num is not None:
                sol_filename = add_filename_tail(sol_filename, f"replication{repl_num}")
            if filename_tail:
                sol_filename = add_filename_tail(sol_filename, filename_tail)
            return os.path.join(BASELINES_FOLDER, self.sol_folder_name, self.general_config, sol_filename)

        def find_first_repl_num(inst_name: str):
            """Find the first replication number such that the solution file does not exist."""
            repl_num = 0
            while os.path.exists(get_sol_path(inst_name=inst_name, repl_num=repl_num)):
                repl_num += 1
            return repl_num

        def solve_instance(inst_name: str, repl_num: int = None):
            """Solve the given instance with the given replication number."""
            instance = Instance.from_name(instance_name, num_dams=self.num_dams)
            sol_path = get_sol_path(inst_name=inst_name, repl_num=repl_num)
            self.log(f"[Replication {repl_num}] Solving {instance_name} in replication {repl_num}...")
            self.solve_instance(instance, sol_path=sol_path)
            self.log(f"[Replication {repl_num}] Finished solving {instance_name} in replication {repl_num}.")

        if instance_names is None:
            instance_names = Baseline.instance_names_eval

        for instance_name in instance_names:
            if num_replications is None:
                solve_instance(inst_name=instance_name)
            else:
                replication_num = find_first_repl_num(instance_name)
                final_replication_num = num_replications - 1
                if replication_num > final_replication_num:
                    self.log(
                        f"[Instance {instance_name}] The given final number of replications is {num_replications}, "
                        f"but the solver {self.solver} already has {replication_num} solutions for {instance_name}."
                    )
                while replication_num <= final_replication_num:
                    solve_instance(inst_name=instance_name, repl_num=replication_num)
                    replication_num += 1

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
            # TODO: in case the number of replications is high, I should trial.report() each replication objective
            #   to enable pruning and accelerate tuning: https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html
            self.log(
                f"[Trial {trial_num}] Total average normalized objective function value: {trial_objective}"
            )
            return trial_objective

        def handle_pso_config(trial: Trial, config: PSOConfiguration):
            """Set the missing required attributes for this specific case"""
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
                    low=neighbors_step, high=int((config.num_particles - 2) / neighbors_step) * neighbors_step,
                    step=neighbors_step
                )
            if config.topology == "ring":
                config.topology_minkowski_p_norm = trial.suggest_categorical(
                    "topology_minkowski_p_norm", choices=[1, 2]
                )

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
                handle_pso_config(trial=trial, config=config)
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


class Baselines:

    """
    Utility class to analyze multiple solvers at the same time
    """

    def __init__(self, general_config: str, solvers: list[str], include_folders: list[str] = None):

        """
        Initialize a Baselines object.

        :param general_config: General configuration (e.g., "G1")
        :param solvers: Solvers that want to be analyzed (e.g., ['MILP', 'PSO', ...])
        :param include_folders: Additional folders in which to look for solutions (e.g. ['old', 'tuned'])
        """

        self.general_config = general_config
        self.solvers = solvers

        # Get all solutions of the given solvers
        self.solutions = []
        for baseline in get_all_baselines(self.general_config):
            if baseline.get_solver() in self.solvers:
                self.solutions.append(baseline)

        # Add old solutions
        if include_folders is not None:
            for folder_name in include_folders:
                for baseline in get_all_baselines_folder(folder_name=folder_name, general_config=self.general_config):
                    solver = baseline.get_solver()
                    if solver in self.solvers:
                        solver += f' ({folder_name})'
                        baseline.set_solver(solver)
                        if solver not in self.solvers:
                            self.solvers.append(solver)
                        self.solutions.append(baseline)

    def barchart_instances(self):

        """
        Plot a barchart with the objective function value of each solver at every instance.
        @return:
        """

        # Get values: dict[solver, dict[instance, value]]
        values = {solver: dict() for solver in self.solvers}
        for solution in self.solutions:
            values[solution.get_solver()].update({solution.get_instance_name(): solution.get_objective_function()})

        barchart_instances(
            values=values, value_type="Income (€)", title=', '.join(self.solvers), general_config=self.general_config
        )
