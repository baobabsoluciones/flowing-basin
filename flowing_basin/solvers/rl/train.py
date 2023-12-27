from flowing_basin.core import Instance, Solution, Experiment
from flowing_basin.solvers.rl import RLEnvironment, RLConfiguration
from flowing_basin.solvers.rl.feature_extractors import Projector, VanillaCNN
from flowing_basin.solvers.rl.callbacks import SaveOnBestTrainingRewardCallback, TrainingDataCallback
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os


class RLTrain(Experiment):

    def __init__(
            self,
            config: RLConfiguration,
            path_constants: str,
            path_train_data: str,
            path_test_data: str,
            path_observations_folder: str,
            path_folder: str = '.',
            paths_power_models: dict[str, str] = None,
            instance: Instance = None,
            solution: Solution = None,
            verbose: int = 1,
    ):

        super().__init__(instance=instance, solution=solution)
        if solution is None:
            self.solution = None

        self.verbose = verbose

        # Configuration, environment and model (RL agent)
        self.config = config
        observations = np.load(os.path.join(path_observations_folder, 'observations.npy'))
        obs_config = RLConfiguration.from_json(os.path.join(path_observations_folder, 'config.json'))
        self.projector = Projector.create_projector(self.config, observations, obs_config)
        self.path_folder = path_folder
        self.train_env = RLEnvironment(
            config=self.config,
            projector=self.projector,
            path_constants=path_constants,
            path_historical_data=path_train_data,
            paths_power_models=paths_power_models,
            instance=instance,
        )

        os.makedirs(self.path_folder, exist_ok=True)
        filepath_log = os.path.join(self.path_folder, ".")
        self.train_env = Monitor(self.train_env, filename=filepath_log)

        self.model = None
        self.initialize_agent()

        # Variables for periodic evaluation of agent during training
        self.eval_env = RLEnvironment(
            config=self.config,
            projector=self.projector,
            path_constants=path_constants,
            path_historical_data=path_test_data,
            paths_power_models=paths_power_models,
            instance=instance,
        )
        self.eval_env = Monitor(self.eval_env)

        self.training_data = None

    def initialize_agent(self):

        if self.config.feature_extractor == 'MLP':

            policy_type = "MlpPolicy"
            policy_kwargs = None

        elif self.config.feature_extractor == 'CNN':

            policy_type = "CnnPolicy"
            fe_variant = self.config.get("feature_extractor_variant", "vanilla")

            extractor_class = None
            if fe_variant == "vanilla":
                extractor_class = VanillaCNN

            policy_kwargs = dict(
                features_extractor_class=extractor_class,
                features_extractor_kwargs=dict(features_dim=128),
            )

        else:

            raise NotImplementedError(
                f"Feature extractor of type {self.config.feature_extractor} is not supported. Either MLP or CNN"
            )

        self.model = SAC(
            policy_type, self.train_env,
            verbose=1, tensorboard_log=self.config.get("tensorboard_log"), policy_kwargs=policy_kwargs
        )
        
    def solve(self, options: dict = None) -> dict:  # noqa

        """
        Train the model and save it in the given path.
        :param options: Unused argument
        """

        # Set callbacks
        callbacks = []
        if self.config.training_data_callback:
            training_data_callback = TrainingDataCallback(
                eval_freq=self.config.training_data_timesteps_freq,
                instances=self.config.training_data_instances,
                policy_id=self.experiment_id,
                config=self.config,
                projector=self.projector,
                verbose=self.verbose
            )
            callbacks.append(training_data_callback)
        if self.config.evaluation_callback:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=None,
                log_path=self.path_folder,
                eval_freq=self.config.evaluation_timesteps_freq,
                n_eval_episodes=self.config.evaluation_num_episodes,
                deterministic=True,
                render=False,
                verbose=self.verbose
            )
            callbacks.append(eval_callback)
        if self.config.checkpoint_callback:
            checkpoint_callback = SaveOnBestTrainingRewardCallback(
                check_freq=self.config.checkpoint_timesteps_freq,
                log_dir=self.path_folder,
                verbose=self.verbose
            )
            callbacks.append(checkpoint_callback)

        # Train model
        self.model.learn(
            total_timesteps=self.config.num_timesteps,
            log_interval=self.config.log_episode_freq,
            callback=CallbackList(callbacks),
            tb_log_name=self.config.get("tensorboard_log", "SAC")
        )

        # Save model
        filepath_agent = os.path.join(self.path_folder, "model.zip")
        self.model.save(filepath_agent)
        if self.verbose >= 1:
            print(f"Created ZIP file '{filepath_agent}'.")

        # Save training data
        if self.config.training_data_callback:
            self.training_data = training_data_callback.training_data  # noqa
            filepath_training = os.path.join(self.path_folder, "training_data.json")
            self.training_data.to_json(filepath_training)

        return dict()

