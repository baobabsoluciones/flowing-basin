import os
from copy import deepcopy
from datetime import datetime
from dataclasses import fields
import numpy as np
from cornflow_client.core.tools import load_json
from flowing_basin.core import Instance, Solution, Configuration, Experiment
from flowing_basin.solvers.rl import GeneralConfiguration, ReinforcementLearning, RLRun
from flowing_basin.solvers import (
    Heuristic, HeuristicConfiguration, LPModel, LPConfiguration, PSO, PSOConfiguration,
    PsoRbo, PsoRboConfiguration
)
from flowing_basin.solvers.common import (
    BASELINES_FOLDER, get_all_baselines, get_all_baselines_folder, barchart_instances_ax, barchart_instances,
    confidence_interval, lighten_color, preprocess_values, extract_percentile
)
import optuna
from optuna.trial import Trial
from matplotlib import pyplot as plt


class Baseline:

    """
    Class to run solvers or agents (or tune solvers) with a RL general configuration
    and saving solutions in the RL baselines folder
    """

    # RBO is just the Heuristic with different hyperparameter values
    solver_classes = {
        "Heuristic": (Heuristic, HeuristicConfiguration),
        "RBO": (Heuristic, HeuristicConfiguration),
        "MILP": (LPModel, LPConfiguration),
        "PSO": (PSO, PSOConfiguration),
        "PSO-RBO": (PsoRbo, PsoRboConfiguration)
    }
    baseline_solvers = set(solver_classes.keys())

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

        self._rl = None
        self._solver_class = None
        self._is_solver = self.solver in Baseline.baseline_solvers
        if self._is_solver:
            self._is_named_policy = False
        else:
            if not self.solver.startswith("rl-"):
                raise ValueError(
                    f"You cannot use the Baseline class with {self.solver}. "
                    f"It must either be a valid solver ({', '.join(Baseline.baseline_solvers)}) or a RL agent (rl-...)."
                )
            self._is_named_policy = self.solver.split("rl-")[1] in RLRun.named_policies

        self.num_dams = None
        self.config = None
        self.sol_folder_name = None

        if self._is_solver:
            self._init_solver(max_time=max_time, tuned_hyperparams=tuned_hyperparams)
        else:
            self._init_agent()

    def _init_solver(self, max_time: float, tuned_hyperparams: bool):

        """Initialize a baseline solver (such as 'MILP')."""

        self._solver_class, config_class = Baseline.solver_classes[self.solver]

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

    def _init_agent(self):

        """Initialize a baseline agent (such as 'rl-greedy')."""

        if self._is_named_policy:
            self._rl = ReinforcementLearning(f"A1{self.general_config}O2R1T2", verbose=2)
        else:
            self._rl = ReinforcementLearning(self.solver, verbose=self.verbose)
        self.config = self._rl.config
        self.num_dams = self.config.num_dams
        self.sol_folder_name = ""

        if self._rl.config_names['G'] != self.general_config:
            raise ValueError(
                f"The agent's general config {self._rl.config_names['G']} does not match {self.general_config}"
            )

    def log(self, msg: str, verbose: int = 0):
        """Print the given message."""
        if self.verbose > verbose:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            print(f"{current_time} [Solver {self.solver}] [Configuration {self.general_config}] {msg}")

    def get_solver(self, instance: Instance, config: Configuration = None) -> Experiment:
        """
        Get the solver object for the given instance
        """
        if config is None:
            config = self.config
        if self._is_solver:
            solver = self._solver_class(instance=instance, config=config)
        else:
            raise ValueError(f"Cannot get solver of {self.solver}. Please use `solve_instance()` directly.")
        return solver

    def solve_instance(self, instance: Instance, config: Configuration = None, sol_path: str = None) -> Solution:
        """
        Solve the given instance. Saves the solution in the given path.
        :return: The Solution of the given instance
        """

        instance_name = instance.get_instance_name()
        msg_header = f"[Instance {instance_name}]"

        self.log(f"{msg_header} Starting to calculate the solution...")
        if self._is_solver:
            solver = self.get_solver(instance=instance, config=config)
            solver.solve(options=dict())
        else:
            if config is not None:
                raise ValueError(f"Cannot solve instance with {self.solver} using a different configuration: {config}.")
            if self._is_named_policy:
                solver = self._rl.run_named_policy(self.solver.split("rl-")[1], instance=instance)
                # run_named_policy() already executes the solve() method
            else:
                solver = self._rl.run_agent(instance=instance)
                # run_agent() already executes the solve() method

        sol_inconsistencies = solver.solution.check()
        if sol_inconsistencies:
            raise Exception(f"There are inconsistencies in the calculated solution: {sol_inconsistencies}")

        solver.solution.data["instance_name"] = instance_name
        solver.solution.data["solver"] = self.solver
        self.log(
            f"{msg_header} Finished calculating solution with "
            f"objective function value {solver.solution.get_objective_function()}"
        )

        if sol_path is not None:
            solver.solution.to_json(sol_path)
            self.log(f"{msg_header} Saved solution in path {sol_path}")

        return solver.solution

    def solve(
            self, instance_names: list[str] = None, num_replications: int = None, save_sol: bool = True
    ) -> list[Solution]:
        """
        Solve each instance and save it in the corresponding baselines folder.
        :param instance_names: Names of the instances to solve
        :param num_replications: Total number of replications (including previous executions of this program)
        :param save_sol: Whether to save the solutions or not
        """

        if instance_names is None:
            instance_names = Baseline.instance_names_eval

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
            return os.path.join(BASELINES_FOLDER, self.sol_folder_name, self.general_config, sol_filename)

        def find_first_repl_num(inst_name: str):
            """Find the first replication number such that the solution file does not exist."""
            repl_num = 0
            while os.path.exists(get_sol_path(inst_name=inst_name, repl_num=repl_num)):
                repl_num += 1
            return repl_num

        def solve_instance(inst_name: str, repl_num: int = None) -> Solution:
            """Solve the given instance with the given replication number."""
            instance = Instance.from_name(instance_name, num_dams=self.num_dams)
            sol_path = get_sol_path(inst_name=inst_name, repl_num=repl_num) if save_sol else None
            self.log(f"[Replication {repl_num}] Solving {instance_name} in replication {repl_num}...")
            sol = self.solve_instance(instance, sol_path=sol_path)
            self.log(f"[Replication {repl_num}] Finished solving {instance_name} in replication {repl_num}.")
            return sol

        solutions = []
        for instance_name in instance_names:
            if num_replications is None:
                solution = solve_instance(inst_name=instance_name)
                solutions.append(solution)
            else:
                replication_num = find_first_repl_num(instance_name)
                final_replication_num = num_replications - 1
                if replication_num > final_replication_num:
                    self.log(
                        f"[Instance {instance_name}] The given final number of replications is {num_replications}, "
                        f"but the solver {self.solver} already has {replication_num} solutions for {instance_name}."
                    )
                while replication_num <= final_replication_num:
                    solution = solve_instance(inst_name=instance_name, repl_num=replication_num)
                    solutions.append(solution)
                    replication_num += 1
        return solutions

    def tune(
            self, num_trials: int, instance_names: list[str] = None, num_replications: int = 5,
            objective_type: str = 'improvement_rl_greedy'
    ):
        """
        Use Optuna to find tuned hyperparameters
        """

        if not self._is_solver:
            raise ValueError(
                f"Cannot tune {self.solver} in Baseline class. To tune RL agents, please go to flowing_basin/rl_zoo."
            )

        valid_objective_types = {'normalized_income', 'improvement_rl_greedy'}
        error_msg = f"The objective type {objective_type} is not valid. Valid options are {valid_objective_types}."
        if objective_type not in valid_objective_types:
            raise ValueError(error_msg)

        if instance_names is None:
            instance_names = Baseline.instance_names_tune

        path_bounds = os.path.join(Baseline.hyperparams_bounds_folder, f"{self.solver}.json")
        hyperparams_bounds = load_json(path_bounds)

        def get_greedy_value(instance_name: str) -> float:
            """Get the objective function value of rl-greedy in the given instance."""
            rl = ReinforcementLearning(f"A1{self.general_config}O2R1T2", verbose=2)
            instance = Instance.from_name(instance_name, num_dams=rl.config.num_dams)
            sol = rl.run_named_policy(policy_name="greedy", instance=instance).solution
            inconsistencies = sol.check()
            if inconsistencies:
                raise ValueError("There are inconsistencies in the solution:", inconsistencies)
            return sol.get_objective_function()

        def get_instance_objective(config: Configuration, instance_name: str) -> float:
            """Calculate the objective function value for the given instance and adjust it"""
            instance = Instance.from_name(instance_name, num_dams=self.num_dams)
            solution = self.solve_instance(instance, config=config)
            objective_val = solution.get_objective_function()
            if objective_type == 'normalized_income':
                avg_inflow = instance.get_total_avg_inflow()
                avg_price = instance.get_avg_price()
                adjusted_objective_val = objective_val / (avg_inflow * avg_price)
                self.log(
                    f"[Instance {instance_name}] Normalized objective function value: "
                    f"{objective_val} / ({avg_inflow} * {avg_price}) = {adjusted_objective_val}"
                )
            elif objective_type == 'improvement_rl_greedy':
                greedy_value = greedy_values[instance_name]
                adjusted_objective_val = (objective_val - greedy_value) / greedy_value
                self.log(
                    f"[Instance {instance_name}] Adjusted objective function value: "
                    f"({objective_val} - {greedy_value}) / {greedy_value} = {adjusted_objective_val}"
                )
            else:
                raise ValueError(error_msg)
            return adjusted_objective_val

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

        def set_attribute(attribute: str, trial: Trial, config: Configuration):
            """Set the given attribute in the configuration"""
            suggest_method = dict(
                float=trial.suggest_float, int=trial.suggest_int, categorical=trial.suggest_categorical
            )[hyperparams_bounds[attribute]['type']]
            value = suggest_method(attribute, **hyperparams_bounds[attribute]['values'])
            setattr(config, attribute, value)

        def handle_pso_config(trial: Trial, config: PSOConfiguration):
            """Set the missing required attributes for the PSO"""
            # "use_relvars" can only be False when "max_relvar" is 1.
            if config.max_relvar < 1.:
                config.use_relvars = True
            else:
                set_attribute("use_relvars", trial=trial, config=config)
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

        def handle_heuristic_config(trial: Trial, config: HeuristicConfiguration):
            """Set the missing required attributes for the Heuristic"""
            # Set "random_biased_flows" and "random_biased_sorting" if both attributes are in `hyperparams_bounds` file
            # Otherwise, keep the values in `Baseline.hyperparams_values_folder`
            if "random_biased_flows" in hyperparams_bounds and "random_biased_sorting" in hyperparams_bounds:
                # Note both attributes cannot be False at the same time
                set_attribute("random_biased_flows", trial=trial, config=config)
                if config.random_biased_flows:
                    set_attribute("random_biased_sorting", trial=trial, config=config)
                else:
                    config.random_biased_sorting = True
            # Set additional required attributes
            if config.random_biased_flows:
                set_attribute("prob_below_half", trial=trial, config=config)
            if config.random_biased_sorting:
                set_attribute("common_ratio", trial=trial, config=config)

        def objective(trial: Trial):
            """Objective function to maximize in the tuning process"""
            # Trial: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
            config = deepcopy(self.config)
            attributes_to_handle_later = {
                "num_particles", "use_relvars", "random_biased_flows", "random_biased_sorting",
                "prob_below_half", "common_ratio"
            }
            for attribute in hyperparams_bounds:
                # The suggestion of these attributes will be done later
                if attribute not in attributes_to_handle_later:
                    set_attribute(attribute, trial=trial, config=config)
            if isinstance(config, PSOConfiguration):
                handle_pso_config(trial=trial, config=config)
            if isinstance(config, HeuristicConfiguration):
                handle_heuristic_config(trial=trial, config=config)
            config.__post_init__()
            trial_num = trial.number
            self.log(f"[Trial {trial_num}] Configuration: {config}")
            return get_trial_objective(config=config, trial_num=trial_num)

        def copy_implicit_values(config: Configuration):
            """Copy the values hard-coded in the functions above
            (for example, `random_biased_sorting` in `handle_heuristic_config`)"""
            if isinstance(config, PSOConfiguration):
                if config.max_relvar < 1.:
                    config.use_relvars = True
            if isinstance(config, HeuristicConfiguration):
                if "random_biased_flows" in hyperparams_bounds and "random_biased_sorting" in hyperparams_bounds:
                    if not config.random_biased_flows:
                        config.random_biased_sorting = True

        # Precomputed values
        greedy_values = None
        if objective_type == 'improvement_rl_greedy':
            greedy_values = {instance_name: get_greedy_value(instance_name) for instance_name in instance_names}

        # Create or load study
        # Study class: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study
        # Saving studies: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html
        study_name = f"tuning_{self.solver}_{self.general_config}"
        storage_name = "sqlite:///{}.db".format(study_name)  # This will create a file in whichever folder this is run
        study = optuna.create_study(
            direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=True
        )

        # Show the trials already done in the study
        num_trials_done = len(study.trials)
        self.log(f"Existing study has {num_trials_done} trials done.")
        if num_trials_done > 0:
            str_trials_done = '\n'.join(
                [f'Trial {trial.number} | {trial.value} | {trial.params}' for trial in study.trials]
            )
            self.log(str_trials_done)

        # Optimize study for the remaining number of trials
        self.log(f"Optimizing for the remaining {num_trials - num_trials_done} trials...")
        study.optimize(objective, n_trials=num_trials - num_trials_done)
        self.log(
            f"Finished hyperparameter tuning. "
            f"Best trial is Trial {study.best_trial.number} with vale {study.best_trial.value}"
        )

        best_config = deepcopy(self.config)
        self.copy_values(best_config, tuned_hyperparams=study.best_params)
        copy_implicit_values(best_config)

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

    # The hyperparameters that work best in each scenario, according to previous experimentation
    # TODO: get the 'tuned' hyperparams for GO1 and G21 and analyze if they perform better
    BEST_PARAMS = {
        'PSO': {'G0': '', 'G01': '', 'G1': 'tuned', 'G2': 'tuned', 'G21': '', 'G3': 'tuned'}
    }

    # Draw the colors using the 'paired' color map
    # Reference: https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
    paired_cmap = plt.get_cmap('Paired')
    blue, dark_blue = paired_cmap(0), paired_cmap(1)
    green, dark_green = paired_cmap(2), paired_cmap(3)
    red, dark_red = paired_cmap(4), paired_cmap(5)
    orange, dark_orange = paired_cmap(6), paired_cmap(7)
    purple, dark_purple = paired_cmap(8), paired_cmap(9)
    SOLVER_COLORS = {
        'MILP': dark_blue, 'PSO': green, 'PSO (tuned)': dark_green, 'PSO (best)': dark_green,
        'PSO-RBO': purple, 'PSO-RBO (tuned)': dark_purple, 'PSO-RBO (best)': dark_purple,
        'Heuristic': dark_red, 'rl-random': orange, 'rl-greedy': dark_orange
    }

    def __init__(
            self, general_config: str, solvers: list[str], solvers_best: list[str] = None,
            include_folders: list[str] = None, include_solutions: list[Solution] = None,
    ):

        """
        Initialize a Baselines object.

        :param general_config: General configuration (e.g., "G1")
        :param solvers: Solvers whose solutions will be read in the folders (e.g., ['MILP', 'PSO', ...])
        :param include_folders: Additional folders in which to look for solutions (e.g. ['old', 'tuned'])
        :param include_solutions: Additional solutions to include in the calculations (not present in the folders)
        :param solvers_best: Solvers with which to use the best hyperparameters
        (which are the default or tuned, depending on the general config)
        """

        if solvers_best is None:
            solvers_best = []
        if include_folders is None:
            include_folders = []
        if include_solutions is None:
            include_solutions = []

        if not set(solvers_best).issubset(set(solvers)):
            raise ValueError(
                f"The solvers specified as 'best', {solvers_best}, must be a subset of the solvers, {solvers}"
            )
        solvers_normal = [solver for solver in solvers if solver not in solvers_best]

        self.general_config = general_config
        self.solvers = solvers_normal

        self.solutions = []
        self._add_normal_solutions(solvers=solvers_normal)
        for solver in solvers_best:
            self._add_single_folder_solutions(
                Baselines.BEST_PARAMS[solver][self.general_config], solvers=solvers_best, solver_name='best'
            )
        self._add_folders_solutions(include_folders, solvers=solvers)
        self._add_extra_solutions(include_solutions)

        # Keep the original order of the parameter so, for example,
        # 'PSO (best)' in `self.solvers` is at the same place as 'PSO' in `solvers`
        self.solvers.sort(key=lambda solver: solvers.index(solver.split(' ')[0]))

    def _add_normal_solutions(self, solvers: list[str]):
        """Add solutions of the parent folder for the given solvers"""
        for baseline in get_all_baselines(self.general_config):
            if baseline.get_solver() in solvers:
                self.solutions.append(baseline)

    def _add_folders_solutions(self, folder_names: list[str], solvers: list[str]):
        """Add solutions of the given subfolders for the given solvers"""
        for folder_name in folder_names:
            self._add_single_folder_solutions(folder_name, solvers=solvers)

    def _add_single_folder_solutions(self, folder_name: str, solvers: list[str], solver_name: str = None):
        """Add solutions of the given subfolder for the given solvers"""
        if solver_name is None:
            solver_name = folder_name
        for baseline in get_all_baselines_folder(folder_name=folder_name, general_config=self.general_config):
            solver = baseline.get_solver()
            if solver in solvers:
                solver += f' ({solver_name})'
                baseline.set_solver(solver)
                if solver not in self.solvers:
                    self.solvers.append(solver)
                self.solutions.append(baseline)

    def _add_extra_solutions(self, sols: list[Solution]):
        """Add the given solutions"""
        for sol in sols:
            self.solutions.append(sol)
            solver = sol.get_solver()
            if solver not in self.solvers:
                self.solvers.append(solver)

    def get_solver_instance_history_values(
            self, num_timestamps: int = 300
    ) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
        """
        Get an array of historic objective function values for every solver and every instance

        :return: Tuple with:
            i) array of shape (num_timestamps,) with the timestamps;
            ii) dict[solver, dict[instance, array]], where each array is of shape (num_timestamps, num_replications)
        """

        max_timestamp = max(sol.get_last_time_stamp() for sol in self.solutions)
        common_timestamps = np.linspace(0., max_timestamp, num_timestamps)

        history_values = {solver: dict() for solver in self.solvers}
        for solution in self.solutions:
            solver = solution.get_solver()
            instance_name = solution.get_instance_name()
            interp_values = solution.get_history_objective_function_value(common_timestamps)
            if instance_name not in history_values[solver]:
                history_values[solver].update({instance_name: interp_values.reshape(-1, 1)})
            else:
                old_array = history_values[solver][instance_name]
                history_values[solver][instance_name] = np.insert(old_array, old_array.shape[1], interp_values, axis=1)

        return common_timestamps, history_values

    def get_solver_instance_final_values(self) -> dict[str, dict[str, list[float]]]:
        """
        Get the list of final objective function values (one per replication) for every solver and every instance.
        :return: dict[solver, dict[instance, values]]
        """
        final_values = {solver: dict() for solver in self.solvers}
        for solution in self.solutions:
            solver = solution.get_solver()
            instance_name = solution.get_instance_name()
            if instance_name not in final_values[solver]:
                final_values[solver].update({instance_name: [solution.get_objective_function()]})
            else:
                final_values[solver][instance_name].append(solution.get_objective_function())
        return final_values

    def get_solver_instance_final_timestamps(self) -> dict[str, dict[str, list[float]]]:
        """
        Get the list of final timestamps (one per replication) for every solver and every instance.
        :return: dict[solver, dict[instance, values]]
        """
        final_timestamps = {solver: dict() for solver in self.solvers}
        for solution in self.solutions:
            solver = solution.get_solver()
            instance_name = solution.get_instance_name()
            if instance_name not in final_timestamps[solver]:
                final_timestamps[solver].update({instance_name: [solution.get_last_time_stamp()]})
            else:
                final_timestamps[solver][instance_name].append(solution.get_last_time_stamp())
        return final_timestamps

    def get_solver_instance_smoothing_violations(self, in_percentage: bool = True) -> dict[str, dict[str, list[float]]]:
        """
        Get the list of the number of flow smoothing violations (one per replication)
        for every solver and every instance.

        :param in_percentage: Whether to give the violation count as a % of the total number of periods
        :return: dict[solver, dict[instance, num_violations]]
        """

        violation_values = {solver: dict() for solver in self.solvers}

        for solution in self.solutions:

            solver = solution.get_solver()
            instance_name = solution.get_instance_name()
            config = solution.get_configuration()

            general_config_dict = Baseline.get_general_config_dict(self.general_config)
            general_config_obj = GeneralConfiguration.from_dict(general_config_dict)
            num_dams = general_config_obj.num_dams
            instance = Instance.from_name(instance_name, num_dams=num_dams)
            initial_flows = {
                dam_id: instance.get_initial_lags_of_channel(dam_id)[0] for dam_id in instance.get_ids_of_dams()
            }
            max_flows = {
                dam_id: instance.get_max_flow_of_channel(dam_id) for dam_id in instance.get_ids_of_dams()
            }

            num_violations = solution.get_num_flow_smoothing_violations(
                flow_smoothing=config.flow_smoothing,
                initial_flows=initial_flows,
                max_flows=max_flows
            )
            if in_percentage:
                num_violations = num_violations / (len(solution.get_decisions_time_steps()) * solution.get_num_dams())
            if instance_name not in violation_values[solver]:
                violation_values[solver].update({instance_name: [num_violations]})
            else:
                violation_values[solver][instance_name].append(num_violations)

        return violation_values

    def plot_history_values_instances(self, filename: str = None):

        """
        Plot the historic objective function values of all solvers in each instance
        """

        timestamps, values = self.get_solver_instance_history_values()
        solvers, instances = preprocess_values(values)

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(solvers)]
        _, axes = plt.subplots(1, len(instances), figsize=(20, 4), sharey='all')
        axes[0].set_ylabel("Income (€)")

        for instance_name, ax in zip(instances, axes):
            for solver, color in zip(solvers, default_colors):
                values_solver = values[solver][instance_name]
                values_mean = np.mean(values_solver, axis=1)
                ax.plot(timestamps, values_mean, label=solver, color=color)
                if values_solver.shape[1] > 1:
                    lower, upper = confidence_interval(values_solver)
                    ax.fill_between(x=timestamps, y1=lower, y2=upper, color=lighten_color(color))
            ax.set_xlabel("Time (s)")
            ax.set_xticks([timestamps[0], timestamps[-1]])
            ax.set_title(instance_name)
            ax.legend()
        solvers_title = ', '.join(solvers)
        plt.suptitle(f"Evolution of {solvers_title} in {self.general_config}")
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def barchart_instances_ax(self, ax: plt.Axes):
        """
        Plot a barchart in the given Axes with the objective function value of each solver at every instance.
        :param ax: matplotlib.pyplot Axes object
        """
        values = self.get_solver_instance_final_values()
        barchart_instances_ax(
            ax, values=values, value_type="Income (€)", title=', '.join(self.solvers),
            general_config=self.general_config, solver_colors=Baselines.SOLVER_COLORS
        )

    def barchart_instances(self):
        """
        Plot a barchart with the objective function value of each solver at every instance.
        """
        values = self.get_solver_instance_final_values()
        barchart_instances(
            values=values, value_type="Income (€)", title=', '.join(self.solvers),
            general_config=self.general_config, solver_colors=Baselines.SOLVER_COLORS
        )

    def get_csv_milp_final_gaps(self) -> list[list[str | float]]:
        """
        Create a list of lists representing a CSV file
        with the final MILP gap in every instance.
        """

        # Get dict[instance, final_gap] sorted by percentile number
        final_gaps = dict()
        for solution in self.solutions:
            if solution.get_solver() == 'MILP':
                final_gaps[solution.get_instance_name()] = solution.get_final_gap_value()
        final_gaps = dict(sorted(final_gaps.items(), key=lambda x: extract_percentile(x[0])))  # noqa

        rows = []
        first_row = []
        for instance in final_gaps.keys():
            first_row.append(instance)
        first_row.append("Average")
        rows.append(first_row)

        second_row = []
        for instance in final_gaps.keys():
            second_row.append(f"{final_gaps[instance]:.2f}%")
        second_row.append(f"{np.mean(list(final_gaps.values())):.2f}%")
        rows.append(second_row)

        return rows

    def get_csv_instance_smoothing_violations(self, in_percentage: bool = True) -> list[list[str | float]]:
        """
        Create a list of lists representing a CSV file
        with the final objective function value of each solver in every instance.

        :param in_percentage: Whether to give the violation count as a % of the total number of periods
        """

        values = self.get_solver_instance_smoothing_violations(in_percentage)
        solvers, instances = preprocess_values(values)
        rows = []

        first_row = ["Solver"]
        for intance in instances:
            first_row.append(intance)
        first_row.append("Average")
        rows.append(first_row)

        for solver in solvers:
            solver_row = [solver]
            # Value for each instance
            for instance in instances:
                # Use the mean across all replications
                instance_mean = np.mean(values[solver][instance])
                solver_row.append(f"{instance_mean:.2%}" if in_percentage else f"{instance_mean:.2f}")
            # Mean across all instances
            solver_mean = np.mean(list(values[solver].values()))
            solver_row.append(f"{solver_mean:.2%}" if in_percentage else f"{solver_mean:.2f}")
            rows.append(solver_row)
        return rows

    def get_csv_instance_final_timestamps(self) -> list[list[str | float]]:
        """
        Create a list of lists representing a CSV file
        with the exectuion time of each solver in every instance.
        """

        values = self.get_solver_instance_final_timestamps()
        solvers, instances = preprocess_values(values)
        rows = []

        first_row = ["Solver"]
        for intance in instances:
            first_row.append(intance)
        first_row.append("Average")
        rows.append(first_row)

        for solver in solvers:
            solver_row = [solver]
            for instance in instances:
                # Use the mean across all replications
                instance_mean = np.mean(values[solver][instance])
                solver_row.append(f"{instance_mean:.0f}")
            # Mean across all instances
            solver_mean = np.mean(list(values[solver].values()))
            solver_row.append(f"{solver_mean:.0f}")
            rows.append(solver_row)
        return rows

    def get_csv_instance_final_values(self, reference: str = None) -> list[list[str | float]]:
        """
        Create a list of lists representing a CSV file
        with the final objective function value of each solver in every instance.

        :param reference: Solver to use as the reference
        """

        values = self.get_solver_instance_final_values()
        solvers, instances = preprocess_values(values)
        rows = []

        if reference is not None and reference not in solvers:
            raise ValueError(f"The reference must be one of the solvers, but {reference} is not in {solvers}")

        first_row = ["Solver"]
        for intance in instances:
            first_row.append(intance)
        first_row.append("Average")
        rows.append(first_row)

        for solver in solvers:

            solver_row = [solver]

            # Value for each instance
            for instance in instances:
                # Use the mean across all replications
                instance_mean = np.mean(values[solver][instance])
                if reference is None:
                    solver_row.append(f"{instance_mean:.0f}")
                else:
                    ref_value = np.mean(values[reference][instance])
                    if ref_value > 0:
                        improvement = (instance_mean - ref_value) / ref_value
                        solver_row.append(f"{'+'if improvement > 0 else ''}{improvement:.2%}")
                    else:
                        solver_row.append(f"+inf%")

            # Mean across all instances
            solver_mean = np.mean(list(values[solver].values()))
            if reference is None:
                solver_row.append(f"{solver_mean:.0f}")
            else:
                ref_mean = np.mean(list(values[reference].values()))
                if ref_mean > 0:
                    improvement = (solver_mean - ref_mean) / ref_mean
                    solver_row.append(f"{'+' if improvement > 0 else ''}{improvement:.2%}")
                else:
                    solver_row.append(f"+inf%")

            rows.append(solver_row)

        return rows
