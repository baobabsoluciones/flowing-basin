from flowing_basin.core import Instance
from flowing_basin.solvers.rl import Environment
from cornflow_client.core.tools import load_json
import pandas as pd
from datetime import datetime
import pickle
from random import randint


class Training:
    def __init__(
        self,
        length_episodes: int,
        path_constants: str,
        path_training_data: str,
        paths_power_models: dict[str, str],
        num_prices: int,
        num_incoming_flows: int,
        num_unreg_flows: int,
        initial_row: int | datetime = None,
    ):

        self.length_episodes = length_episodes

        self.constants = load_json(path_constants)
        self.training_data = pd.read_pickle(path_training_data)

        self.env = Environment(
            instance=self.create_instance(
                length_episodes=self.length_episodes,
                constants=self.constants,
                training_data=self.training_data,
                initial_row=initial_row,
            ),
            paths_power_models=paths_power_models,
            num_prices=num_prices,
            num_incoming_flows=num_incoming_flows,
            num_unreg_flows=num_unreg_flows,
        )

    def reset(self, instance: Instance):

        self.env.reset(instance)

    @staticmethod
    def create_instance(
        length_episodes: int,
        constants: dict,
        training_data: pd.DataFrame,
        initial_row: int | datetime = None,
    ) -> Instance:

        # Incomplete instance (we create a deepcopy of constants to avoid modifying it)
        data = pickle.loads(pickle.dumps(constants, -1))
        instance_constants = Instance.from_dict(data)

        # Get necessary constants
        dam_ids = instance_constants.get_ids_of_dams()
        channel_last_lags = {
            dam_id: instance_constants.get_relevant_lags_of_dam(dam_id)[-1]
            for dam_id in dam_ids
        }

        # Required rows from data frame
        total_rows = len(training_data.index)
        min_row = max(channel_last_lags.values())
        max_row = total_rows - length_episodes
        if isinstance(initial_row, datetime):
            initial_row = training_data.index[
                training_data["datetime"] == initial_row
            ].tolist()[0]
        if initial_row is None:
            initial_row = randint(min_row, max_row)
        assert initial_row in range(
            min_row, max_row + 1
        ), f"{initial_row=} should be between {min_row=} and {max_row=}"
        last_row = initial_row + length_episodes - 1
        last_row_decisions = last_row - max(
            [
                instance_constants.get_relevant_lags_of_dam(dam_id)[0]
                for dam_id in instance_constants.get_ids_of_dams()
            ]
        )

        # Add time-dependent values to the data

        data["datetime"]["start"] = training_data.loc[
            initial_row, "datetime"
        ].strftime("%Y-%m-%d %H:%M")
        data["datetime"]["end_decisions"] = training_data.loc[
            last_row_decisions, "datetime"
        ].strftime("%Y-%m-%d %H:%M")

        data["incoming_flows"] = training_data.loc[
            initial_row:last_row, "incoming_flow"
        ].values.tolist()
        data["energy_prices"] = training_data.loc[
            initial_row:last_row, "price"
        ].values.tolist()

        for order, dam_id in enumerate(dam_ids):

            # If dam is not dam1 or dam2,
            # it will be e.g. dam3_dam2copy (a copy of dam2) or dam4_dam1copy (a copy of dam1)
            if dam_id not in ["dam1", "dam2"]:
                dam_id = dam_id[dam_id.rfind("_") + 1: dam_id.rfind("copy")]

            # Initial volume
            # Not to be confused with the volume at the end of the first time step
            data["dams"][order]["initial_vol"] = training_data.loc[
                initial_row, dam_id + "_vol"
            ]

            initial_lags = training_data.loc[
                initial_row - channel_last_lags[dam_id] : initial_row - 1,
                dam_id + "_flow",
            ].values.tolist()
            initial_lags.reverse()
            data["dams"][order]["initial_lags"] = initial_lags

            data["dams"][order]["unregulated_flows"] = training_data.loc[
                initial_row:last_row, dam_id + "_unreg_flow"
            ].values.tolist()

        # Complete instance
        instance = Instance.from_dict(data)

        return instance
