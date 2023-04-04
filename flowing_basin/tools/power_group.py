from flowing_basin.core import Instance
import warnings
import pickle
import lightgbm
import numpy as np


class PowerGroup:

    def __init__(
        self,
        idx: str,
        past_flows: np.ndarray,
        instance: Instance,
        paths_power_models: dict[str, str],
        num_scenarios: int,
    ):

        self.num_scenarios = num_scenarios

        self.idx = idx
        self.power_model = self.get_power_model(paths_power_models[self.idx])
        self.relevant_lags = instance.get_relevant_lags_of_dam(self.idx)
        self.turbined_flow_points = instance.get_turbined_flow_obs_for_power_group(self.idx)
        self.startup_flows = instance.get_startup_flows_of_power_group(self.idx)
        self.shutdown_flows = instance.get_shutdown_flows_of_power_group(self.idx)

        # Power generated (MW) and turbined flow (m3/s)
        self.power = self.get_power(past_flows)
        self.turbined_flow = self.get_turbined_flow(self.power)

    def reset(self, past_flows: np.ndarray, num_scenarios: int):

        """
        Reset power and turbined flow
        Power models and turbined flow observations are not reset as they are constant
        """

        self.num_scenarios = num_scenarios

        self.power = self.get_power(past_flows)
        self.turbined_flow = self.get_turbined_flow(self.power)

    @staticmethod
    def get_power_model(path_power_model: str) -> lightgbm.LGBMClassifier:

        """
        Light Gradient Boosting model that predicts the power generated by the power group
        given the flows that had previously passed though the channel (lags)

        The model can be fed the past flows or lags in multiple ways.
        If the relevant lags are 2, 3, 4, 5, with flow values 5.4, 5.6, 3.4, 2.1, then:
        OpA. list:              model_load.predict([[5.4, 5.6, 3.4, 2.1]])
        OpB. NumPy array:       model_load.predict(np.array([5.4, 5.6, 3.4, 2.1]).reshape(1,-1))
        OpC. Pandas dataframe:  model_load.predict(df[["EX_Q_K_lag2", "EX_Q_K_lag3", "EX_Q_K_lag4", "EX_Q_K_lag5"]])
        """

        # Load model under a warnings catch,
        # to avoid seeing the warnings for using a more recent version of scikit-learn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_load = pickle.load(open(path_power_model, "rb"))

        return model_load

    def get_power(self, past_flows: np.ndarray) -> np.ndarray:

        """
        Get the current power generated by the power group.

        :param past_flows:
            Array of shape num_scenarios x num_lags with
            all past flows of the channel (m3/s)
        :return:
            Array of shape num_scenarios with
            the power generated in this time step in every scenario (MW)
        """

        # Take only the relevant columns of the array
        first_lag = self.relevant_lags[0]
        last_lag = self.relevant_lags[-1]
        power = self.power_model.predict(past_flows[:, first_lag - 1: last_lag])

        return power

    def get_turbined_flow(self, power: np.ndarray) -> np.ndarray:

        """
        Get the current turbined flow by interpolating the power-flow curve with the current generated power.

        :param power:
            Array of shape num_scenarios with
            the current power generated by the power group in every scenario (MW)
        :return:
            Array of shape num_scenarios with
            the corresponding turbined flows to the given powers (m3/s)
        """

        # Interpolate power to get flow
        turbined_flow = np.interp(
            power,
            self.turbined_flow_points["observed_powers"],
            self.turbined_flow_points["observed_flows"],
        )

        return turbined_flow

    def get_num_active_power_groups(self, turbined_flow: np.ndarray) -> np.ndarray:

        """
        Get the number of active power groups.
        0 = no power group active
        1 = one power group active (flow above one start-up flow)
        1.5 = uncertain (flow between start-up and shutdown flows)
        2 = two power groups active (flow above two start-up flows)
        ...
        """

        # Obtain the number of exceeded startup flows for every scenario
        # - Turn the FLOW 1D array into a 2D array, repeating the 1D array in as many ROWS as there are STARTUP FLOWS
        # - Turn the STARTUP FLOWS 1D array into a 2D array, repeating the 1D array in as many COLS as there are FLOWS
        # - Comparing both 2D arrays, determine, for each flow, the startup flows are exceeded, and sum them
        flow_broadcast_to_startup_flows = np.tile(turbined_flow, (len(self.startup_flows), 1))
        startup_flows_broadcast_to_flows = np.transpose(np.tile(self.startup_flows, (len(turbined_flow), 1)))
        exceeded_startup_flows = np.sum(flow_broadcast_to_startup_flows > startup_flows_broadcast_to_flows, axis=0)

        # Obtain the number of exceeded shutdown flows for every scenario
        # This will only be different for flows in between a startup and shutdown flow
        flow_broadcast_to_shutdown_flows = np.tile(turbined_flow, (len(self.shutdown_flows), 1))
        shutdown_flows_broadcast_to_flows = np.transpose(np.tile(self.shutdown_flows, (len(turbined_flow), 1)))
        exceeded_shutdown_flows = np.sum(flow_broadcast_to_shutdown_flows > shutdown_flows_broadcast_to_flows, axis=0)

        # Get the average of the number of exceeded startup flows and the number of exceeded shutdown flows
        num_active_power_groups = (exceeded_startup_flows + exceeded_shutdown_flows) / 2

        return num_active_power_groups

    def update(self, past_flows: np.ndarray) -> np.ndarray:

        """
        Update the current power generated by the power group, as well as its turbined flow.

        :param past_flows:
            Array of shape num_scenarios x num_lags with
            all past flows of the channel (m3/s)
        :return:
            Array of shape num_scenarios with
            the turbined flow of every scenario (m3/s)
        """

        self.power = self.get_power(past_flows)
        self.turbined_flow = self.get_turbined_flow(self.power)

        # Bring turbined flow upstream, since it is used to update the volume of the next dam
        return self.turbined_flow
