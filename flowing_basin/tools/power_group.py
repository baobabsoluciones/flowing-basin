class PowerGroup:

    def __init__(self, flows_over_time: list,
                 path_power_model: str):

        self.path_power_model = path_power_model

        self.power = self.get_power(flows_over_time)

    def get_power(self, flows_over_time: list) -> float:

        """
        Using the parameters of the model saved in self.path_power_model,
        returns the power generated in this time step
        :param flows_over_time: Flows assigned in the previous time steps to the preceding channel
        :return:
        """

        pass

    def update(self, flows_over_time: list) -> None:

        """
        Update the current power generated by the power group
        :param flows_over_time:
        :return:
        """

        self.power = self.get_power(flows_over_time)