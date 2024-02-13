from flowing_basin.core import Instance, Solution, Experiment, TrainingData
from flowing_basin.solvers.rl import RLEnvironment, RLConfiguration
from flowing_basin.solvers.rl.feature_extractors import Projector, VanillaCNN
from flowing_basin.solvers.rl.callbacks import SaveOnBestTrainingRewardCallback, TrainingDataCallback
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
import warnings
import re


class RLTrain(Experiment):

    """
    Class to train the RL agent

    :param config:
    :param projector:
    :param path_constants:
    :param path_train_data:
    :param path_test_data:
    :param path_folder: Folder in which to save the agent and its information. If None, the agent will not be saved.
    :param path_tensorboard: Folder with the tensorboard logs in which to add the agent log info. If None, no logging.
    :param update_observation_record:
    :param save_replay_buffer:
    :param instance:
    :param solution:
    :param experiment_id:
    :param verbose:
    :param num_timesteps: If given, override the number of timesteps in the configuration (only for testing purposes)
    """

    def __init__(
            self,
            config: RLConfiguration,
            projector: Projector,
            path_constants: str,
            path_train_data: str,
            path_test_data: str,
            path_folder: str = None,
            path_tensorboard: str = None,
            paths_power_models: dict[str, str] = None,
            update_observation_record: bool = False,
            save_replay_buffer: bool = True,
            instance: Instance = None,
            solution: Solution = None,
            experiment_id: str = None,
            verbose: int = 1,
            num_timesteps: int = None
    ):

        super().__init__(instance=instance, solution=solution, experiment_id=experiment_id)
        if solution is None:
            self.solution = None

        self.verbose = verbose
        self.config = config
        self.num_timesteps = num_timesteps if num_timesteps is not None else self.config.num_timesteps
        self.projector = projector
        self.save_replay_buffer = save_replay_buffer

        # Path to the folder with all the information about the agent, including the model itself
        self.path_old_folder = None
        self.path_old_model = None
        self.path_old_replay_buffer = None
        self.path_old_training_data = None
        self.path_new_folder = None
        self.path_new_model = None
        self.path_new_replay_buffer = None
        self.path_new_training_data = None
        self._set_agent_files(path_folder)

        # Path to tensorboard logging directory
        self.path_tensorboard = path_tensorboard

        # Training environment
        self.train_env = RLEnvironment(
            config=self.config,
            projector=self.projector,
            update_observation_record=update_observation_record,
            path_constants=path_constants,
            path_historical_data=path_train_data,
            paths_power_models=paths_power_models,
            instance=instance,
        )
        if self.path_new_folder is not None:
            self.train_env = Monitor(self.train_env, filename=os.path.join(self.path_new_folder, "."))
        else:
            self.train_env = Monitor(self.train_env)

        # Model (RL agent)
        self.model = None
        self._initialize_agent()

        # Variables for periodic evaluation of agent during training
        self.eval_env = RLEnvironment(
            config=self.config,
            projector=self.projector,
            update_observation_record=update_observation_record,
            path_constants=path_constants,
            path_historical_data=path_test_data,
            paths_power_models=paths_power_models,
            instance=instance,
        )
        self.eval_env = Monitor(self.eval_env)

    def _set_agent_files(self, path_folder: str = None):

        """
        Create the folder with agent information and set the paths to its files
        :param path_folder:
        :return:
        """

        # Get the old (previous training iteration) and new (current training iteration) agent folders
        self.path_old_folder = None
        self.path_new_folder = path_folder
        if path_folder is not None:
            # Get the training iteration of the given folder (e.g., for "rl-A113G0O2R1T3_1", it is 1)
            match = re.search(r"_\d+", path_folder)
            if match is None:
                # Either the previous iteration was 0, or there was no previous iteration
                if os.path.exists(path_folder):
                    # The previous training iteration is 0 and the new one is 1
                    self.path_old_folder = path_folder
                    self.path_new_folder = path_folder + "_1"
            else:
                # The previous training iteration was not 0, and we need to replace the number with the new one
                training_iter = int(match.group()[1:])
                self.path_old_folder = path_folder
                self.path_new_folder = re.sub(r"_\d+", f"_{training_iter + 1}", path_folder)

        # Set paths of files inside old folder
        if self.path_old_folder is not None:
            self.path_old_model = os.path.join(self.path_old_folder, "model.zip")
            self.path_old_replay_buffer = os.path.join(self.path_old_folder, 'replay_buffer.pickle')
            self.path_old_training_data = os.path.join(self.path_old_folder, "training_data.json")

        # Set paths of files inside new folder
        if self.path_new_folder is not None:
            self.path_new_model = os.path.join(self.path_new_folder, "model.zip")
            self.path_new_replay_buffer = os.path.join(self.path_new_folder, 'replay_buffer.pickle')
            self.path_new_training_data = os.path.join(self.path_new_folder, "training_data.json")

    def _initialize_agent(self):

        """
        Define the model
        """

        if self.config.feature_extractor == 'MLP':

            policy_type = "MlpPolicy"
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], qf=[256, 256])
            )

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

        # Make default value explicit to avoid error when loading
        policy_kwargs.update(
            dict(use_sde=False)
        )

        # Check if training should continue from pre-trained model
        self.from_pretrained = False
        if self.path_old_model is not None and os.path.exists(self.path_old_model):
            self.from_pretrained = True
            if self.verbose >= 1:
                print(f"The training will continue from a pre-trained model saved in '{self.path_old_model}'.")

        # Define the model
        model_kwargs = dict(
            learning_rate=self.config.learning_rate, buffer_size=self.config.replay_buffer_size,
            verbose=self.verbose, tensorboard_log=self.path_tensorboard, policy_kwargs=policy_kwargs
        )
        if self.from_pretrained:
            self.model = SAC.load(self.path_old_model, env=self.train_env, **model_kwargs)
        else:
            self.model = SAC(policy_type, self.train_env, **model_kwargs)

        # Load replay buffer of pre-trained model
        if self.path_old_replay_buffer is not None and os.path.exists(self.path_old_replay_buffer):
            self.model.load_replay_buffer(self.path_old_replay_buffer)
            if self.verbose >= 1:
                print(f"Loaded replay buffer of pre-trained model with {self.model.replay_buffer.size()} transitions.")
        elif self.from_pretrained:
            warnings.warn(
                f"Pre-trained model does not have a replay buffer in '{self.path_old_replay_buffer}'. "
                f"Training may not be smooth."
            )

        if self.verbose >= 2:
            print("Model architecture to train:")
            print(self.model.policy)
        
    def solve(self, options: dict = None) -> dict:  # noqa

        """
        Train the model and save it in the given path.
        :param options: Unused argument
        """

        # Set callbacks
        callbacks = []
        if self.config.training_data_callback:
            if self.path_old_training_data is not None and os.path.exists(self.path_old_training_data):
                training_data = TrainingData.from_json(self.path_old_training_data)
                if self.verbose >= 1:
                    print(f"Loaded training data of pre-trained model from '{self.path_old_training_data}'.")
            else:
                training_data = None
            training_data_callback = TrainingDataCallback(
                eval_freq=self.config.training_data_timesteps_freq,
                instances=self.config.training_data_instances,
                policy_id=self.experiment_id,
                config=self.config,
                projector=self.projector,
                training_data=training_data,
                verbose=self.verbose
            )
            callbacks.append(training_data_callback)
        if self.config.evaluation_callback:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=self.path_new_folder if self.config.evaluation_save_best else None,
                log_path=self.path_new_folder,
                eval_freq=self.config.evaluation_timesteps_freq,
                n_eval_episodes=self.config.evaluation_num_episodes,
                deterministic=True,
                render=False,
                verbose=self.verbose
            )
            callbacks.append(eval_callback)
        if self.config.checkpoint_callback and self.path_new_folder is not None:
            checkpoint_callback = SaveOnBestTrainingRewardCallback(
                check_freq=self.config.checkpoint_timesteps_freq,
                log_dir=self.path_new_folder,
                verbose=self.verbose
            )
            callbacks.append(checkpoint_callback)

        # Train model
        learn_kwargs = dict(
            total_timesteps=self.num_timesteps,
            log_interval=self.config.log_episode_freq,
            callback=CallbackList(callbacks),
            tb_log_name=self.experiment_id
        )
        if self.from_pretrained:
            learn_kwargs.update(
                dict(reset_num_timesteps=False)
            )
        self.model.learn(**learn_kwargs)

        # Saving the replay buffer allows smoother training later
        if self.save_replay_buffer and self.path_new_replay_buffer is not None:
            self.model.save_replay_buffer(self.path_new_replay_buffer)
            if self.verbose >= 1:
                print(
                    f"Saved replay buffer with {self.model.replay_buffer.size()} transitions "
                    f"in file '{self.path_new_replay_buffer}'."
                )

        # Save model
        if self.path_new_folder is not None:
            filepath_agent = self.path_new_model
            self.model.save(filepath_agent)
            if self.verbose >= 1:
                print(f"Created ZIP file '{filepath_agent}'.")

        # Save training data
        if self.config.training_data_callback and self.path_new_training_data is not None:
            training_data_callback.training_data.to_json(self.path_new_training_data)  # noqa

        return dict()

