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
        mode: str,
    ):

        valid_modes = {"linear", "nonlinear"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid value for 'mode': {mode}. Allowed values are {valid_modes}")

        if mode == "nonlinear" and paths_power_models is None:
            raise TypeError(
                "Parameter 'paths_power_models' is required when 'mode' is 'nonlinear', but it was not given."
            )

        self.num_scenarios = num_scenarios
        self.mode = mode

        self.idx = idx
        if paths_power_models is not None:
            self.power_model = self.get_nonlinear_power_model(paths_power_models[self.idx])
        self.relevant_lags = instance.get_relevant_lags_of_dam(self.idx)
        self.verification_lags = instance.get_verification_lags_of_dam(self.idx)
        self.turbined_flow_points = instance.get_turbined_flow_obs_for_power_group(
            self.idx
        )
        self.startup_flows = instance.get_startup_flows_of_power_group(self.idx)
        self.shutdown_flows = instance.get_shutdown_flows_of_power_group(self.idx)
        self.time_step_hours = instance.get_time_step_seconds() / 3600

        # Save the decision horizon and this power group's impact horizon
        self.decision_horizon = instance.get_decision_horizon()
        self.impact_horizon = self.decision_horizon + instance.get_relevant_lags_of_dam(self.idx)[0]
        # TODO: again, shouldn't it be get_relevant_lags_of_dam(self.idx)[-1]?

        # Time-dependent attributes
        self.time = None
        self.power = None
        self.turbined_flow = None
        self.previous_num_active_groups = None
        self.num_active_groups = None
        self.income = None
        self.num_startups = None
        self.num_times_in_limit = None
        self.acc_income = None
        self.acc_num_startups = None
        self.acc_num_times_in_limit = None

        # Initialize the time-dependent attributes (variables)
        self._reset_variables(past_flows)

    def _reset_variables(self, past_flows: np.ndarray):

        """
        Reset all time-varying attributes of the power group:
        power, turbined flow, number of active groups, total number of startups, and number of times in limit zones.
        Power models, turbined flow observations, and startup flows are not reset as they are constant.
        """

        # Identifier of the time step
        # It should be equal to the RiverBasin's time identifier at all times
        self.time = -1

        # Power generated (MW), turbined flow (m3/s) and number of active groups
        if self.mode == "nonlinear":
            self.update_power_turbined_flow_nonlinear(past_flows)
        if self.mode == "linear":
            self.update_power_turbined_flow_linear(past_flows)
        self.previous_num_active_groups = None
        self.num_active_groups = self.get_num_active_power_groups(self.turbined_flow)

        # Total (accumulated) number of power group startups, and of times in limit zones
        self.income = None
        self.num_startups = None
        self.num_times_in_limit = None
        self.acc_income = np.zeros(self.num_scenarios)
        self.acc_num_startups = np.zeros(self.num_scenarios)
        self.acc_num_times_in_limit = np.zeros(self.num_scenarios)

        return

    def reset(self, past_flows: np.ndarray, num_scenarios: int):

        """
        Reset the power group.
        """

        self.num_scenarios = num_scenarios

        self._reset_variables(past_flows)

        return

    @staticmethod
    def get_nonlinear_power_model(path_power_model: str) -> lightgbm.LGBMClassifier:

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

    def update_power_turbined_flow_nonlinear(self, past_flows: np.ndarray):

        """
        Set the current power by using the ML power models (nonlinear method),
        and set the current turbined_flow by interpolating the power.

        power - array of shape num_scenarios with the power generated in this time step in every scenario (MW)

        turbined_flow - array of shape num_scenarios with the turbined flow in every scenario (m3/s)

        :param past_flows:
            Array of shape num_scenarios x num_lags with
            the relevant past flows of the channel (m3/s)
        """

        # Get power from past flows
        # Take only the relevant columns of the array
        first_lag = self.relevant_lags[0]
        last_lag = self.relevant_lags[-1]
        self.power = self.power_model.predict(past_flows[:, first_lag - 1: last_lag])

        # Get turbined flow from power
        self.turbined_flow = self.get_turbined_flow_from_power(self.power)

    def update_power_turbined_flow_linear(self, past_flows: np.ndarray):

        """
        Set the current turbined flow by averaging the verification lags (linear method),
        and set the current power by interpolating the turbined flow.

        turbined_flow - array of shape num_scenarios with the turbined flow in every scenario (m3/s)

        power - array of shape num_scenarios with the power generated in this time step in every scenario (MW)

        :param past_flows:
            Array of shape num_scenarios x num_lags with
            the relevant past flows of the channel (m3/s)
        """

        # Get turbined flow from past flows
        first_lag = self.verification_lags[0]
        last_lag = self.verification_lags[-1]
        self.turbined_flow = np.mean(past_flows[:, first_lag - 1: last_lag], axis=1)

        # Get power from turbined flow
        self.power = self.get_power_from_turbined_flow(self.turbined_flow)

    def get_turbined_flow_from_power(self, power: np.ndarray) -> np.ndarray:

        """
        Get the turbined flow by interpolating the power-flow curve with the given power.

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

    def get_power_from_turbined_flow(self, turbined_flow: np.ndarray) -> np.ndarray:

        """
        Get the power by interpolating the power-flow curve with the given turbined flow.

        :param turbined_flow:
            Array of shape num_scenarios with
            the current turbined flow in every scenario (m3/s)
        :return:
            Array of shape num_scenarios with
            the corresponding powers to the given turbined flows (MW)
        """

        power = np.interp(
            turbined_flow,
            self.turbined_flow_points["observed_flows"],
            self.turbined_flow_points["observed_powers"],
        )

        return power

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
        flow_broadcast_to_startup_flows = np.tile(
            turbined_flow, (len(self.startup_flows), 1)
        )
        startup_flows_broadcast_to_flows = np.transpose(
            np.tile(self.startup_flows, (len(turbined_flow), 1))
        )
        exceeded_startup_flows = np.sum(
            flow_broadcast_to_startup_flows > startup_flows_broadcast_to_flows, axis=0
        )

        # Obtain the number of exceeded shutdown flows for every scenario
        # This will only be different for flows in between a startup and shutdown flow
        flow_broadcast_to_shutdown_flows = np.tile(
            turbined_flow, (len(self.shutdown_flows), 1)
        )
        shutdown_flows_broadcast_to_flows = np.transpose(
            np.tile(self.shutdown_flows, (len(turbined_flow), 1))
        )
        exceeded_shutdown_flows = np.sum(
            flow_broadcast_to_shutdown_flows > shutdown_flows_broadcast_to_flows, axis=0
        )

        # Get the average of the number of exceeded startup flows and the number of exceeded shutdown flows
        num_active_power_groups = (exceeded_startup_flows + exceeded_shutdown_flows) / 2

        return num_active_power_groups

    def update(self, price: float, past_flows: np.ndarray) -> np.ndarray:

        """
        Update the current power generated by the power group, as well as its turbined flow.

        :param price: Price of energy in current time step (EUR/MWh)
        :param past_flows:
            Array of shape num_scenarios x num_lags with
            all past flows of the channel (m3/s)
        :return:
            Array of shape num_scenarios with
            the turbined flow of every scenario (m3/s)
        """

        self.time += 1
        self.previous_num_active_groups = self.num_active_groups.copy()

        if self.mode == "nonlinear":
            self.update_power_turbined_flow_nonlinear(past_flows)
        if self.mode == "linear":
            self.update_power_turbined_flow_linear(past_flows)
        self.num_active_groups = self.get_num_active_power_groups(self.turbined_flow)

        self.income = self.power * self.time_step_hours * price  # MW * h * EUR/MWh
        self.num_startups = np.maximum(
            0,
            np.floor(self.num_active_groups)
            - np.floor(self.previous_num_active_groups),
        )
        self.num_times_in_limit = np.invert(
            np.equal(self.num_active_groups, np.round(self.num_active_groups))
        )

        if self.time < self.impact_horizon:
            self.acc_income += self.income
        if self.time < self.decision_horizon:
            self.acc_num_startups += self.num_startups
            self.acc_num_times_in_limit += self.num_times_in_limit

        # Bring turbined flow upstream, since it is used to update the volume of the next dam
        return self.turbined_flow
