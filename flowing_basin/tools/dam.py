from flowing_basin.core import Instance
from .channel import Channel
import numpy as np


class Dam:
    def __init__(
        self,
        idx: str,
        instance: Instance,
        paths_power_models: dict[str, str],
        flow_smoothing: int,
        num_scenarios: int,
        mode: str,
    ):

        self.num_scenarios = num_scenarios
        self.flow_smoothing = flow_smoothing

        self.idx = idx
        self.order = instance.get_order_of_dam(self.idx)

        self.decision_horizon = instance.get_decision_horizon()

        # Constant values for the whole period - time step (seconds, s), min/max volumes (m3)
        self.time_step = instance.get_time_step_seconds()
        self.min_volume = instance.get_min_vol_of_dam(self.idx)
        self.max_volume = instance.get_max_vol_of_dam(self.idx)

        # Time-dependent attributes
        self.volume = None
        self.final_volume = None
        self.previous_flow_out = None
        self.all_previous_variations = None
        self.flow_contribution = None
        self.unregulated_flow = None
        self.flow_out_assigned = None
        self.flow_out_smoothed = None
        self.flow_out_clipped1 = None
        self.flow_out_clipped2 = None

        # Initialize the time-dependent attributes (variables)
        self._reset_variables(instance)

        self.channel = Channel(
            idx=self.idx,
            dam_vol=self.volume,  # noqa
            instance=instance,
            paths_power_models=paths_power_models,
            num_scenarios=self.num_scenarios,
            mode=mode,
        )

    def _reset_variables(self, instance: Instance):

        """
        Reset all time-varying attributes of the dam: volume and previous flows.
        Min and max volumes are not reset as they are constant.
        """

        # Initial volume of dam (m3) - the STARTING volume in this time step, or the FINAL volume of the previous one
        self.volume = np.repeat(
            instance.get_initial_vol_of_dam(self.idx), self.num_scenarios
        )

        # Clip initial volume of dam
        self.volume = np.clip(self.volume, self.min_volume, self.max_volume)

        # Volume at the end of the decision horizon
        self.final_volume = None

        # Number of periods in which we force to keep the direction (flow increasing OR decreasing) in the dam
        # Note that this rule may not apply if flows must be clipped
        self.previous_flow_out = np.repeat(
            instance.get_initial_lags_of_channel(self.idx)[0], self.num_scenarios
        )
        self.all_previous_variations = np.zeros((1, self.num_scenarios))

        # Other relevant information for when decisions are made
        self.flow_contribution = None
        self.unregulated_flow = None
        self.flow_out_assigned = None
        self.flow_out_smoothed = None
        self.flow_out_clipped1 = None
        self.flow_out_clipped2 = None

        return

    def reset(self, instance: Instance, flow_smoothing: int, num_scenarios: int):

        """
        Reset dam and the channel within.
        """

        self.flow_smoothing = flow_smoothing
        self.num_scenarios = num_scenarios

        self._reset_variables(instance)

        self.channel.reset(
            dam_vol=self.volume, instance=instance, num_scenarios=self.num_scenarios
        )

    def update(
        self,
        flow_out: np.ndarray,
        price: float,
        unregulated_flow: float,
        time: int,
        flow_contribution: np.ndarray,
    ) -> np.ndarray:

        """
        Update the volume of the dam, and the state of its connected channel.

        :param flow_out:
            Array of shape num_scenarios with
            the flow we want to have exiting the dam in every scenario (m3/s)
        :param price: Price of energy in current time step (EUR/MWh)
        :param unregulated_flow: Unregulated flow entering the dam in the current time step (m3/s)
        :param flow_contribution:
            Array of shape num_scenarios with
            the flow entering this dam (from the river or the previous dam) in every scenario (m3/s)
        :param time:
            Identifier of the time step, used to save the volume at the decision horizon
            and to accumulate the income of only the decision time steps
        :return:
            Array of shape num_scenarios with
            the turbined flow in the power group in every scenario (m3/s)
        """

        # Flow IN ---- #

        # Obtain flow coming into the dam from the river or the previous dam
        self.flow_contribution = flow_contribution

        # Obtain unregulated flow coming into the dam
        self.unregulated_flow = unregulated_flow

        # Flow OUT ---- #

        # Flow coming out of the dam
        self.flow_out_assigned = flow_out

        # Flow smoothed
        # Prevent change of direction among the last N = flow_smoothing periods:
        # Check all elements of the current variation have the same sign as the last N variations
        # Otherwise, set these flows to the previous flows
        current_assigned_variation = self.flow_out_assigned - self.previous_flow_out
        previous_variations = (
            self.all_previous_variations[-self.flow_smoothing :]
            if self.flow_smoothing > 0
            else np.zeros(self.num_scenarios).reshape((1, -1))
        )
        sign_changes_each_period = (
            previous_variations * current_assigned_variation < -1e-6  # -1e-6 instead of 0. <-- tolerance for rounding errors
        )  # Broadcasting
        sign_changes_any_period = np.sum(sign_changes_each_period, axis=0) > 0
        self.flow_out_smoothed = np.where(
            sign_changes_any_period, self.previous_flow_out, self.flow_out_assigned
        )

        # print(time, self.flow_out_assigned, self.flow_out_smoothed, current_assigned_variation, previous_variations, sign_changes_each_period)

        # Flow clipped according to the flow limit of the channel
        self.flow_out_clipped1 = np.clip(
            self.flow_out_smoothed, 0, self.channel.flow_limit
        )

        # Volume ---- #

        # Volume at the END of this time step
        old_volume = self.volume
        volume_increase = (
            self.unregulated_flow + self.flow_contribution
        ) * self.time_step
        volume_decrease = self.flow_out_clipped1 * self.time_step
        self.volume = old_volume + volume_increase - volume_decrease

        # Volume clipped to min value
        self.volume = np.clip(self.volume, self.min_volume, None)

        # Flow clipped if volume was below minimum
        # This only changes the value of the flow if the volume was increased to the minimum value
        self.flow_out_clipped2 = (
            old_volume + volume_increase - self.volume
        ) / self.time_step

        # Volume clipped to max value
        self.volume = np.clip(self.volume, None, self.max_volume)

        # print("dams", time, volume_increase.item(), volume_decrease.item(), self.flow_out_clipped2.item(), self.volume.item())

        # Volume at the end of the decision horizon ---- #

        if time == self.decision_horizon - 1:
            self.final_volume = self.volume

        # Values to smooth flow in next time step ---- #

        current_actual_variation = self.flow_out_clipped2 - self.previous_flow_out
        self.all_previous_variations = np.concatenate(
            [self.all_previous_variations, current_actual_variation.reshape(1, -1)],
            axis=0,
        )
        self.previous_flow_out = self.flow_out_clipped2.copy()

        # Channel ---- #

        # We update the channel with the new volume (the FINAL volume in this time step),
        # because the channel stores the FINAL maximum flow, which is calculated with this volume
        return self.channel.update(price=price, flow=self.flow_out_clipped2, dam_vol=self.volume, time=time)
