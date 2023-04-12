import torch


class Agent:

    """
    Trivial agent (generates random actions), used for testing
    """

    def __init__(
        self, action_lower_limits: torch.Tensor, action_upper_limits: torch.Tensor
    ):

        assert (
            action_lower_limits.size() == action_upper_limits.size()
        ), "Lower and upper limits for action must be the same size"

        self.action_size = action_lower_limits.size()
        self.action_lower_limits = action_lower_limits
        self.action_upper_limits = action_upper_limits

    def policy(self) -> torch.Tensor:

        return (
            torch.rand(self.action_size)
            * (self.action_upper_limits - self.action_lower_limits)
            - self.action_lower_limits
        )
