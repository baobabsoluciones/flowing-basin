from flowing_basin.core import Instance, Solution, TrainingData
from flowing_basin.solvers.rl import (
    GeneralConfiguration, ObservationConfiguration, ActionConfiguration, RewardConfiguration, TrainingConfiguration,
    RLConfiguration, RLEnvironment, RLTrain, RLRun
)
from flowing_basin.solvers.rl.feature_extractors import Projector
from cornflow_client.core.tools import load_json
from stable_baselines3.common.env_checker import check_env
import numpy as np
import math
from matplotlib import pyplot as plt
import os
import re
import warnings
from time import perf_counter
from datetime import datetime


class ReinforcementLearning:

    # RL data
    configs_info = {
        "G": (os.path.join(os.path.dirname(__file__), "../../rl_data/configs/general"), GeneralConfiguration),
        "O": (os.path.join(os.path.dirname(__file__), "../../rl_data/configs/observation"), ObservationConfiguration),
        "A": (os.path.join(os.path.dirname(__file__), "../../rl_data/configs/action"), ActionConfiguration),
        "R": (os.path.join(os.path.dirname(__file__), "../../rl_data/configs/reward"), RewardConfiguration),
        "T": (os.path.join(os.path.dirname(__file__), "../../rl_data/configs/training"), TrainingConfiguration)
    }
    observation_records_folder = os.path.join(os.path.dirname(__file__), "../../rl_data/observation_records")

    # General data
    constants_path = os.path.join(os.path.dirname(__file__), "../../data/constants/constants_2dams.json")
    train_data_path = os.path.join(os.path.dirname(__file__), "../../data/history/historical_data_clean_train.pickle")
    test_data_path = os.path.join(os.path.dirname(__file__), "../../data/history/historical_data_clean_test.pickle")

    models_folder = os.path.join(os.path.dirname(__file__), "../../rl_data/models")
    baselines_folder = os.path.join(os.path.dirname(__file__), "../../rl_data/baselines")
    tensorboard_folder = os.path.join(os.path.dirname(__file__), "../../rl_data/tensorboard_logs")

    observation_records = ["record_raw_obs", "record_normalized_obs", "record_projected_obs"]  # As the attributes in RLEnvironment
    static_projectors = ["identity", "QuantilePseudoDiscretizer"]
    obs_basic_type_length = 2  # Length of the observation code that indicates the basic type (e.g., "O121" -> "O1")
    obs_type_length = 3  # Length of the observation code that indicates the type (e.g., "O121" -> "O12")
    obs_collection_pos = 2  # Position of the digit that MAY indicate a collection method (e.g., "O12" -> "2")
    obs_collection_codes = {'1', '2'}  # Digits that DO indicate a collection method (e.g., not "0" in "O10")

    def __init__(self, config_name: str, verbose: int = 1, save_obs: bool = True):

        self.verbose = verbose
        self.save_obs = save_obs
        config_name = config_name
        self.config_names = self.extract_substrings(config_name)
        self.config_full_name = ''.join(self.config_names.values())  # Identical to `config_name`, but in alphabetical order

        self.config = self.get_config(self.config_names)
        self.agent_name = f"rl-{self.config_full_name}"
        self.agent_path = os.path.join(ReinforcementLearning.models_folder, self.agent_name)
        self.constants = Instance.from_dict(load_json(ReinforcementLearning.constants_path))

        # The first two digits in the observation name (e.g., "O211" -> "O21")
        # indicate the type of observations that should be used for the projector
        self.obs_records_path = os.path.join(
            ReinforcementLearning.observation_records_folder, self.config_names["O"][0:3]
        )

    def train(self, save_agent: bool = True) -> RLTrain | None:

        """
        Train an agent with the given configuration.

        :param save_agent: Whether to save the agent or not
        (not saving the agent may be interesting for testing purposes).
        """

        if os.path.exists(self.agent_path) and save_agent:
            warnings.warn(f"Training aborted. Folder '{self.agent_path}' already exists.")
            return

        train = RLTrain(
            config=self.config,
            update_observation_record=self.save_obs,
            projector=self.create_projector(),
            path_constants=ReinforcementLearning.constants_path,
            path_train_data=ReinforcementLearning.train_data_path,
            path_test_data=ReinforcementLearning.test_data_path,
            path_folder=self.agent_path if save_agent else None,
            path_tensorboard=ReinforcementLearning.tensorboard_folder if save_agent else None,
            experiment_id=self.agent_name,
            verbose=self.verbose
        )
        if self.verbose >= 1:
            print(f"Training agent with {''.join(self.config_names.values())} for {self.config.num_timesteps} timesteps...")
        start = perf_counter()
        train.solve()
        if self.verbose >= 1:
            print(f"Trained for {self.config.num_timesteps} timesteps in {perf_counter() - start}s.")
        if self.save_obs and save_agent:
            for obs_record in ReinforcementLearning.observation_records:
                obs_record_path = os.path.join(self.agent_path, f'{obs_record}.npy')
                obs = np.array(getattr(train.train_env, obs_record))
                np.save(obs_record_path, obs)
                if self.verbose >= 1:
                    print(f"Saved {obs_record} with {obs.shape} observations in file '{obs_record_path}'.")

        return train

    def collect_obs(self) -> RLEnvironment | None:

        """
        Collect observations for the observation type (e.g., for "O211", the observation type is "O21")
        """

        if os.path.exists(self.obs_records_path):
            warnings.warn(f"Observation collection aborted. Folder '{self.obs_records_path}' already exists.")
            return

        if len(self.config_names["O"]) < 3:
            warnings.warn(
                f"Observation collection aborted. "
                f"The observation config name does not indicate the collection method (no second digit)."
            )
            return

        # Get the collection method, indicated by the second digit in the observation name (e.g., "O211" -> "1")
        collection_code = self.config_names["O"][ReinforcementLearning.obs_collection_pos]
        if collection_code not in ReinforcementLearning.obs_collection_codes:
            warnings.warn(
                f"Observation collection aborted. "
                f"The observation config name does not indicate a collection method (second digit is not '1' or '2')."
            )
            return
        collection_method = {
            "1": "training",
            "2": "random"
        }[collection_code]

        # The agent must be executed using no projector
        # This is done taking only the first 2 values of the observation config name (e.g., "O211" -> "O2")
        reduced_config_names = self.config_names.copy()
        reduced_config_names["O"] = reduced_config_names["O"][:ReinforcementLearning.obs_basic_type_length]
        reduced_config = self.get_config(reduced_config_names)
        reduced_agent_name = f"rl-{''.join(reduced_config_names.values())}"

        if collection_method == 'training':
            if self.verbose >= 1:
                print(f"Collecting observations for {reduced_config.num_timesteps} timesteps while training agent...")
            reduced_agent_path = os.path.join(ReinforcementLearning.models_folder, reduced_agent_name)
            train = RLTrain(
                config=reduced_config,
                projector=Projector.create_projector(reduced_config),
                path_constants=ReinforcementLearning.constants_path,
                path_train_data=ReinforcementLearning.train_data_path,
                path_test_data=ReinforcementLearning.test_data_path,
                path_folder=reduced_agent_path,
                path_tensorboard=ReinforcementLearning.tensorboard_folder,
                experiment_id=reduced_agent_name,
                update_observation_record=True,
                verbose=self.verbose
            )
            start = perf_counter()
            train.solve()
            if self.verbose >= 1:
                print(f"Collected for {reduced_config.num_timesteps} timesteps in {perf_counter() - start}s.")
            env = train.train_env

        elif collection_method == 'random':
            if self.verbose >= 1:
                print(f"Collecting observations for {reduced_config.num_timesteps} timesteps with random agent...")
            env = RLEnvironment(
                config=reduced_config,
                projector=Projector.create_projector(reduced_config),
                path_constants=ReinforcementLearning.constants_path,
                path_historical_data=ReinforcementLearning.train_data_path,
                update_observation_record=True
            )
            start = perf_counter()
            num_timesteps = 0
            while num_timesteps < reduced_config.num_timesteps:
                env.reset()
                done = False
                while not done:
                    action = env.action_space.sample()
                    _, _, done, _, _ = env.step(action)
                    num_timesteps += 1
                    if self.verbose >= 2 and num_timesteps % 495 == 0:
                        print(
                            f"Collected {num_timesteps} / {reduced_config.num_timesteps} "
                            f"({num_timesteps / reduced_config.num_timesteps * 100:.1f}%) observations "
                            f"in {perf_counter() - start}s so far"
                        )
            if self.verbose >= 1:
                print(f"Collected for {reduced_config.num_timesteps} timesteps in {perf_counter() - start}s.")

        else:
            raise ValueError(f"Invalid value for `collection_method`: {collection_method}.")

        normalized_obs = np.array(env.record_normalized_obs)
        if self.verbose >= 1:
            print("Observations collected, (num_observations, num_features):", normalized_obs.shape)
        os.makedirs(self.obs_records_path)
        np.save(os.path.join(self.obs_records_path, 'observations.npy'), normalized_obs)
        reduced_config.to_json(os.path.join(self.obs_records_path, 'config.json'))
        if self.verbose >= 1:
            print(f"Created folder '{self.obs_records_path}'.")

        return env

    def check_train_env(self, max_timestep: int = float('inf'), obs_types: list[str] = None, initial_date: str = None):

        """
        Check the training environment works as expected

        :param max_timestep: Number of timesteps for which to run the environment (default, until the episode finishes)
        :param obs_types: Observation types to show (default, all of them: "raw", "normalized", and "projected")
        :param initial_date: Initial date of the instance of the environment, in format '%Y-%m-%d %H:%M' (default, random)
        """

        if obs_types is None:
            obs_types = ["raw", "normalized", "projected"]

        env = self.create_train_env()

        # Check the environment does not have any errors according to StableBaselines3
        check_env(env)

        if initial_date is not None:
            initial_date = datetime.strptime(initial_date, '%Y-%m-%d %H:%M')

        env.reset(initial_row=initial_date)
        print("\nINSTANCE:")
        print(env.instance.data)
        instance_checks = env.instance.check()
        if instance_checks:
            raise RuntimeError(f"There are problems with the created instance: {instance_checks}.")

        done = False
        timestep = 0
        while not done and timestep < max_timestep:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            print(f"\n---- Timestep {timestep} ----")
            print("\nACTION:", action)
            print("FLOW:", info['flow'])
            print("\nREWARD:", reward)
            print("REWARD DETAILS:", env.get_reward_details())
            for obs_type in obs_types:
                print(f"\n{obs_type} OBSERVATION:".upper())
                obs_to_print = info[f'{obs_type}_obs'] if obs_type != 'projected' else obs
                env.print_obs(obs_to_print)

            timestep += 1

    def create_train_env(self) -> RLEnvironment:

        """
        Create the training environment for the agent
        """

        env = RLEnvironment(
            config=self.config,
            projector=self.create_projector(),
            path_constants=ReinforcementLearning.constants_path,
            path_historical_data=ReinforcementLearning.train_data_path,
            update_observation_record=self.save_obs
        )
        return env

    def create_projector(self) -> Projector:

        """
        Get the projector corresponding to the given configuration,
        assuming the required observations folder exists.
        """

        if os.path.exists(self.obs_records_path):
            observations = np.load(os.path.join(self.obs_records_path, 'observations.npy'))
            obs_config = RLConfiguration.from_json(os.path.join(self.obs_records_path, 'config.json'))
            if self.verbose >= 1:
                print(f"Using observations from '{self.obs_records_path}' for projector.")
        else:
            if self.config.projector_type != "identity":
                raise FileNotFoundError(
                    f"Cannot build projector because the projector type is not 'identity' "
                    f"and the observations folder '{self.obs_records_path}' does not exist."
                )
            observations = None
            obs_config = None
        projector = Projector.create_projector(self.config, observations, obs_config)
        return projector

    def plot_histogram(self, obs: np.ndarray, projected: bool, title: str):

        """

        :param obs: Array of shape num_observations x num_features with the flattened observations
        :param projected: Indicates if the observations are projected observations or raw/normalized observations
        :param title: Title of the histogram
        """

        # Method 'auto' raises an error for giving almost 0 width bins
        # when input array is small and the array has extreme outliers.
        # (This was observed with the feature past_clipped when having 1001 observations.)
        # This is avoided by forcing the usage of the Sturges method for bin width estimation,
        # which is better for small datasets.
        bins_method = 'sturges' if obs.shape[0] < 50_000 else 'auto'

        # Raw or normalized flattened observation
        if not projected:
            indices = self.config.get_obs_indices(flattened=True)
            for dam_id in self.constants.get_ids_of_dams():
                max_sight = max(self.config.num_steps_sight[feature, dam_id] for feature in self.config.features)
                num_features = len(self.config.features)
                fig, axs = plt.subplots(max_sight, num_features)
                fig.suptitle(f"Histograms of {title} for {dam_id}")
                for feature_index, feature in enumerate(self.config.features):
                    for lookback in range(self.config.num_steps_sight[feature, dam_id]):
                        ax = axs[lookback, feature_index]
                        if self.constants.get_order_of_dam(dam_id) == 1 or feature not in self.config.unique_features:
                            index = indices[dam_id, feature, lookback]
                            ax.hist(obs[:, index], bins=bins_method)
                        ax.set_yticklabels([])  # Hide y-axis tick labels
                        ax.yaxis.set_ticks_position('none')  # Hide y-axis tick marks
                        if lookback == 0:
                            ax.set_title(feature)
                plt.tight_layout()
                plt.show()

        # Projected observation
        else:
            n_components = obs.shape[1]
            num_cols = math.ceil(math.sqrt(n_components))
            # We want to guarantee that
            # num_rows * num_cols > n_components ==> num_rows = math.ceil(n_components / num_cols)
            num_rows = math.ceil(n_components / num_cols)
            fig, axs = plt.subplots(num_rows, num_cols)
            fig.suptitle(f"Histograms of {title}")
            component = 0
            for row in range(num_rows):
                for col in range(num_cols):
                    if component < n_components:
                        ax = axs[row, col]
                        ax.hist(obs[:, component], bins='auto')
                        ax.set_yticklabels([])  # Hide y-axis tick labels
                        ax.yaxis.set_ticks_position('none')  # Hide y-axis tick marks
                        ax.set_title(f"Component {component}")
                    component += 1
            plt.tight_layout()
            plt.show()

    def plot_histograms_projector_obs(self):

        """
        Plot the histograms of the observations used to train the projector,
        as well as these observations after being transformed by this projector.
        """

        projector = self.create_projector()
        obs_type = self.config_names['O'][:ReinforcementLearning.obs_type_length]

        self.plot_histogram(projector.observations, projected=False, title=f"Original observations {obs_type}")
        self.plot_histograms_projected(projector.observations, projector, title=str(obs_type))

    def plot_histograms_agent_obs(self, apply_projections: bool = True):

        """
        Plot the histogram of the record of raw, normalized and projected observations experienced by the agent

        :param apply_projections: If False, get the projected observations from "record_projected_obs" in the folder;
        if True, manually apply the projector to get the projected observations
        """

        obs_normalized = None
        for obs_record in ReinforcementLearning.observation_records:
            obs_folder = os.path.join(self.agent_path, f'{obs_record}.npy')
            try:
                obs = np.load(obs_folder)
            except FileNotFoundError:
                warnings.warn(f"Histograms of {obs_record} not plotted. File '{obs_folder}' does not exist.")
                continue

            if obs_record != "record_projected_obs":
                self.plot_histogram(obs, projected=False, title=obs_record)

            # Get the projected observations from the folder
            if obs_record == "record_projected_obs" and not apply_projections:
                proj_type = self.config.projector_type
                proj_type = proj_type if not isinstance(proj_type, list) else ', '.join(proj_type)
                projected = proj_type not in ReinforcementLearning.static_projectors
                self.plot_histogram(obs, projected=projected, title=f"{obs_record} ({proj_type})")

            if obs_record == "record_normalized_obs":
                obs_normalized = obs

        # Manually apply the projections
        if apply_projections:
            if obs_normalized is None:
                warnings.warn("Cannot apply projections because the normalized observations were not found.")
                return
            projector = self.create_projector()
            self.plot_histograms_projected(obs_normalized, projector)

    def plot_histograms_projected(self, obs: np.ndarray, projector: Projector, title: str = None):

        """
        Plot the histograms of the projections of the given observations
        """

        if title is None:
            title = self.agent_name

        def indicate_variance(proj_type: str):
            if proj_type not in ReinforcementLearning.static_projectors:
                return f"({self.config.projector_explained_variance * 100:.0f}%)"
            else:
                return ""

        if isinstance(self.config.projector_type, list):
            proj_types = []
            for proj, proj_type in zip(projector.projectors, self.config.projector_type):  # noqa
                proj_types.append(proj_type)
                is_projected = proj_type not in ReinforcementLearning.static_projectors
                obs = proj.transform(obs)
                self.plot_histogram(
                    obs,
                    projected=is_projected,
                    title=f"Observations from {title} after applying "
                          f"{', '.join(proj_types)} {indicate_variance(proj_type)}"
                )
        else:
            is_projected = self.config.projector_type not in ReinforcementLearning.static_projectors
            obs = projector.project(obs)
            self.plot_histogram(
                obs,
                projected=is_projected,
                title=f"Observations from {title} after applying "
                      f"{self.config.projector_type} {indicate_variance(self.config.projector_type)}"
            )

    def run_agent(self, instance: Instance | str, model: str = "best_model") -> Solution:

        """
        Solve the given instance with the current agent

        :param instance: Instance to solve, or its name
        :param model: Either "model" (the model in the last timestep of training)
            or "best_model" (the model with the highest evaluation from StableBaselines3 EcalCallback)
        """

        if isinstance(instance, str):
            instance = Instance.from_name(instance)

        model_path = os.path.join(self.agent_path, f"{model}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"There is no trained model {self.agent_name}. File {model_path} doesn't exist."
            )

        run = RLRun(
            config=self.config,
            instance=instance,
            projector=self.create_projector(),
            solver_name=self.agent_name
        )
        run.solve(model_path)

        return run.solution

    def run_named_policy(self, policy_name: str, instance: Instance) -> Solution:

        """
        Solve the given instance with a special policy ("random" or "greedy")
        """

        if policy_name not in RLRun.named_policies:
            raise ValueError(
                f"Invalid value for `policy_name`: {policy_name}. Allowed values are {RLRun.named_policies}."
            )

        run = RLRun(
            config=self.config,
            instance=instance,
            projector=self.create_projector(),
            solver_name=f"rl-{policy_name}"
        )
        run.solve(policy_name)
        return run.solution

    def plot_training_curve_agent(
            self, agents: list[str] = None, baselines: list[str] = None,
            values: list[str] = None, instances: str | list[str] = 'fixed'
    ):

        """
        Plot the training curve of the agent
        compared with the training curve of the given agents
        and the values of the given baseline solvers
        """

        if agents is None:
            agents = []

        ReinforcementLearning.plot_training_curves(
            agents=[self.agent_name, *agents], baselines=baselines, values=values, instances=instances
        )

    @staticmethod
    def plot_all_training_curves(
            agents_regex_filter: str = '.*', permutation: str = 'AGORT', baselines: list[str] = None,
            values: list[str] = None, instances: str | list[str] = 'fixed'
    ):

        """
        Plot the training curve of all agents matching the given regex pattern
        """

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        ReinforcementLearning.plot_training_curves(
            agents=agents, baselines=baselines, values=values, instances=instances
        )

    @staticmethod
    def plot_training_curves(
            agents: list[str], baselines: list[str] = None,
            values: list[str] = None, instances: str | list[str] = 'fixed'
    ):

        """
        Plot the training curves of the given agents
        and the values of the given baseline solvers.

        :param agents: List of agent IDs in rl_data/models (e.g., ["rl-A1G0O22R1T0", "rl-A1G0O221R1T0"]).
        :param baselines: List of solvers in the baselines folder (e.g., ["MILP", "rl-random", "rl-greedy"]).
        :param values: Can be "income" or "acc_reward".
        :param instances: Can be "fixed", "random", or a list of specific fixed instances.
        """

        if baselines is None:
            baselines = ['MILP', 'rl-random', 'rl-greedy']
        if values is None:
            values = ['income']

        # Ensure all agents have the same General configuration
        general_config = ReinforcementLearning.common_general_config(
            agents, warning_msg="The plotted graphs will not be comparable and baselines will not be shown."
        )

        # For each agent, create a TrainingData object from "training_data.json"
        # and then add "evaluations.npz" from the EvalCallback to the TrainingData object
        training_objects = []
        for agent in agents:
            training_object = ReinforcementLearning.get_training_data(agent)
            training_object.add_random_instances(agent, ReinforcementLearning.get_path_evaluations(agent))
            training_objects.append(training_object)
        training = sum(training_objects)

        # Add baselines
        if general_config is not None:
            for baseline in ReinforcementLearning.get_all_baselines(general_config):
                if baseline.get_solver() in baselines:
                    training += baseline

        _, ax = plt.subplots()
        training.plot_training_curves(ax, values=values, instances=instances)  # noqa
        plt.show()

    @staticmethod
    def barchart_training_times(agents_regex_filter: str = '.*', permutation: str = 'AGORT'):

        """
        Show the training time of all agents matching the given regex in a barchart
        """

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        training_times = ReinforcementLearning.get_training_times(agents)

        plt.bar(agents, training_times)
        plt.xticks(rotation='vertical')  # Put the agent IDs vertically
        plt.xlabel('Agents')
        plt.ylabel('Training time (min)')
        plt.title('Training time of agents')
        plt.tight_layout()  # Avoid the agent IDs being cut down at the bottom of the figure
        plt.show()

    @staticmethod
    def print_training_times(agents_regex_filter: str = '.*', permutation: str = 'AGORT'):

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        training_times = ReinforcementLearning.get_training_times(agents)

        print("agent; training time (min)")
        for agent, training_time in zip(agents, training_times):
            print(f"{agent}; {training_time:.2f}")

    @staticmethod
    def print_max_avg_incomes(agents_regex_filter: str = '.*', permutation: str = 'AGORT') -> list[list[str]] | None:

        """
        Print a CSV table with the maximum average income of each agent
        and the corresponding average income of the baselines
        """

        results = [
            ["method", "max_avg_income"]
        ]

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        general_config = ReinforcementLearning.common_general_config(
            agents, warning_msg="Cannot print CSV table (the results are not comparable and no baseline can be found)."
        )
        if general_config is None:
            return

        # Add rows with max average income of agents
        training_data_agents = []
        for agent in agents:
            training_data_agent = ReinforcementLearning.get_training_data(agent)
            training_data_agents.append(training_data_agent)
            results.append([agent, training_data_agent.get_max_avg_value()])
        training_data = sum(training_data_agents)

        baselines = ReinforcementLearning.get_all_baselines(general_config)
        for baseline in baselines:
            training_data += baseline

        # Add rows with baseline average incomes
        for baseline_solver, baseline_avg_income in training_data.get_baseline_values().items():  # noqa
            results.append([baseline_solver, baseline_avg_income])

        # Expand each row with the performance w.r.t. the baselines
        for baseline_solver, baseline_avg_income in training_data.get_baseline_values().items():  # noqa
            for result in results[1:]:
                result_avg_income = result[1]
                perf_over_baseline = (result_avg_income - baseline_avg_income) / baseline_avg_income  # noqa
                result += [f"{perf_over_baseline:+.2%} {baseline_solver}"]

        # Turn results to string and print them
        for line in results:
            line = [f'{int(round(el)):,}' if isinstance(el, float) else el for el in line]
            print(';'.join(line))

        return results

    @staticmethod
    def print_spaces(agents_regex_filter: str = '.*', permutation: str = 'AGORT'):

        """
        Print the shapes of the action and observation spaces of each agent
        :param agents_regex_filter:
        :param permutation:
        :return:
        """

        print("agent;action_space;obs_space")
        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        for agent in agents:
            config = ReinforcementLearning.get_config(
                ReinforcementLearning.extract_substrings(agent)
            )
            env = RLEnvironment(
                config=config,
                projector=Projector.create_projector(config),
                path_constants=ReinforcementLearning.constants_path,
                path_historical_data=ReinforcementLearning.train_data_path,
                update_observation_record=False
            )
            print(f"{agent};{env.action_space.shape};{env.observation_space.shape}")

    @staticmethod
    def multi_level_sorting_key(agent: str, permutation: str) -> tuple:

        """
        Breaks down string "rl-A1G0O22R1T1" into the numbers (A) "1", (G) "0", (O) "22", (R) "1", (T) "1",
        and then sorts these numbers by the given permutation of the letters AGORT.

        :param agent: Input agent ID string
        :param permutation: Permutation of the config letters, for example 'AGORT' or 'GTOAR'
        """

        substrings = ReinforcementLearning.extract_substrings(agent)
        sorted_values = [substrings[letter] for letter in permutation]
        return tuple(sorted_values)

    @staticmethod
    def get_avg_training_time(agents_regex_filter: str = '.*', permutation: str = 'AGORT') -> float:

        """
        Get the average training time of all agents matching the given regex
        """

        agents = ReinforcementLearning.get_all_agents(agents_regex_filter, permutation)
        training_times = ReinforcementLearning.get_training_times(agents)

        return sum(training_times) / len(training_times)

    @staticmethod
    def get_training_times(agents: list[str]) -> list[float]:

        """
        Get the training times of the given agents in minutes
        """

        training_times = []
        for agent in agents:
            training_data = ReinforcementLearning.get_training_data(agent)
            training_times.append(training_data.get_training_time())
        return training_times

    @staticmethod
    def get_training_data(agent: str) -> TrainingData:

        """
        Get the training data of the given agent.
        """

        agent_folder = os.path.join(ReinforcementLearning.models_folder, agent)
        training_data_path = os.path.join(agent_folder, "training_data.json")
        training_data = TrainingData.from_json(training_data_path)
        return training_data

    @staticmethod
    def get_path_evaluations(agent: str) -> str:

        agent_folder = os.path.join(ReinforcementLearning.models_folder, agent)
        evaluation_data_path = os.path.join(agent_folder, "evaluations.npz")
        return evaluation_data_path

    @staticmethod
    def common_general_config(agents: list[str], warning_msg: str = '') -> str | None:

        """
        Get the general configuration shared by all given agents
        and give a warning if this is not the case.
        """

        agents_substrings = [ReinforcementLearning.extract_substrings(agent) for agent in agents]
        agents_general = {substrings['G'] for substrings in agents_substrings}
        if len(agents_general) == 1:
            general_config = agents_general.pop()
        else:
            warnings.warn(f"Agents have different general configurations. {warning_msg}")
            general_config = None
        return general_config

    @staticmethod
    def get_all_agents(regex_filter: str = '.*', permutation: str = 'AGORT') -> list[str]:

        """
        Get all agent IDs in alphabetical order, filtering by the given regex pattern
        """

        parent_directory = ReinforcementLearning.models_folder
        all_items = os.listdir(parent_directory)

        # Filter to take only the directories and those matching the regex pattern
        regex = re.compile(regex_filter)
        all_models = [
            item for item in all_items if os.path.isdir(os.path.join(parent_directory, item)) and regex.match(item)
        ]
        all_models.sort(
            key=lambda model: ReinforcementLearning.multi_level_sorting_key(model, permutation=permutation)
        )

        return all_models

    @staticmethod
    def get_all_baselines(general_config: str) -> list[Solution]:

        """
        Scan the folder with solutions that act as baselines for the RL agents.

        :param general_config: General configuration ("G0" or "G1")
        """

        sols = []
        parent_dir = os.path.join(ReinforcementLearning.baselines_folder, general_config)
        for file in os.listdir(parent_dir):
            if file.endswith('.json'):
                full_path = os.path.join(parent_dir, file)
                sol = Solution.from_json(full_path)
                sols.append(sol)
        return sols

    @staticmethod
    def extract_substrings(input_string: str) -> dict[str, str]:

        """
        Break down a string of alphanumeric characters
        into substrings of numbers lead by different alphabetic characters.
        Example: "G0O21A1R1T0" is broken down into {"A": "A1", "G": "G0", "O": "O21", "R": "R1", "T": "T0"}.
        """

        pattern = re.compile(r'([A-Za-z]+)(\d+)')
        matches = pattern.findall(input_string)
        result = ["".join(match) for match in matches]
        result = sorted(result)
        result = {substring[0]: substring for substring in result}
        return result

    @staticmethod
    def get_config(config_names: dict[str, str]) -> RLConfiguration:

        """
        Get the RLConfiguration object from the configuration substrings,
        for example {"A": "A1", "G": "G0", "O": "O21", "R": "R1", "T": "T0"}.
        """

        configs = []
        for config_letter, config_name in config_names.items():
            config_folder, config_class = ReinforcementLearning.configs_info[config_letter]
            if config_letter == "O" and len(config_name) > 2:
                collection_pos = ReinforcementLearning.obs_collection_pos
                maybe_collection_code = config_name[collection_pos]
                if maybe_collection_code in ReinforcementLearning.obs_collection_codes:
                    # Replace the observation collection method for projector by a generic `X` (e.g., "O211" -> "O2X1")
                    config_name = config_name[:collection_pos] + "X" + config_name[collection_pos + 1:]
            config_path = os.path.join(config_folder, config_name + ".json")
            configs.append(config_class.from_json(config_path))
        config = RLConfiguration.from_components(configs)
        return config




