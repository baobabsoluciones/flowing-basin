from flowing_basin.core import Instance
from flowing_basin.solvers.rl import Environment
from cornflow_client.core.tools import load_json
import pandas as pd
from datetime import datetime
import pickle
import os
from random import randint


class Training:
    def __init__(
        self,
        length_episodes: int,
        paths_power_models: dict[str, str],
        num_prices: int,
        num_incoming_flows: int,
        num_unreg_flows: int,
        initial_row: int | datetime = None
    ):

        self.length_episodes = length_episodes

        self.constants = load_json(
            os.path.join(
                os.path.dirname(__file__), "../../data/rl_training_data/constants.json"
            )
        )
        self.training_data = pd.read_pickle(
            os.path.join(
                os.path.dirname(__file__),
                "../../data/rl_training_data/training_data.pickle",
            )
        )

        self.env = Environment(
            instance=self.create_instance(initial_row=initial_row),
            paths_power_models=paths_power_models,
            num_prices=num_prices,
            num_incoming_flows=num_incoming_flows,
            num_unreg_flows=num_unreg_flows,
        )

    def reset(self, instance: Instance):

        self.env.reset(instance)

    def create_instance(self, initial_row: int | datetime = None) -> Instance:

        # Incomplete instance (we create a deepcopy of constants to avoid modifying it)
        data = pickle.loads(pickle.dumps(self.constants, -1))
        instance_constants = Instance.from_dict(data)

        # Get necessary constants
        dam_ids = instance_constants.get_ids_of_dams()
        channel_last_lags = {
            dam_id: instance_constants.get_relevant_lags_of_dam(dam_id)[-1]
            for dam_id in dam_ids
        }

        # Required rows from data frame
        if isinstance(initial_row, datetime):
            initial_row = self.training_data.index[
                self.training_data["datetime"] == initial_row
            ].tolist()[0]
        if initial_row is None:
            total_rows = len(self.training_data.index)
            initial_row = randint(0, total_rows - self.length_episodes)
        last_row = initial_row + self.length_episodes - 1

        # Add time-dependent values to the data

        data["datetime"]["start"] = self.training_data.loc[
            initial_row, "datetime"
        ].strftime("%Y-%m-%d %H:%M")
        data["datetime"]["end"] = self.training_data.loc[last_row, "datetime"].strftime(
            "%Y-%m-%d %H:%M"
        )

        data["incoming_flows"] = self.training_data.loc[
            initial_row:last_row, "incoming_flow"
        ].values.tolist()
        data["energy_prices"] = self.training_data.loc[
            initial_row:last_row, "price"
        ].values.tolist()

        for order, dam_id in enumerate(dam_ids):

            data["dams"][order]["initial_vol"] = self.training_data.loc[
                initial_row, dam_id + "_vol"
            ]

            initial_lags = self.training_data.loc[
                initial_row - channel_last_lags[dam_id] : initial_row - 1,
                dam_id + "_flow",
            ].values.tolist()
            initial_lags.reverse()
            data["dams"][order]["initial_lags"] = initial_lags

            data["dams"][order]["unregulated_flows"] = self.training_data.loc[
                initial_row:last_row, dam_id + "_unreg_flow"
            ].values.tolist()

        # Complete instance
        instance = Instance.from_dict(data)

        return instance


