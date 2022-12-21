class PowerGroup:

    def __init__(self, flows_over_time: list, path_power_model: str):

        self.flows_over_time = flows_over_time
        self.path_power_model = path_power_model

        self.power = self.get_power()

        pass

    def get_power(self) -> float:

        """
        Using the parameters of the model saved in self.path_power_model,
        and the flows assigned in the previous time steps to the preceding channel,
        saved in self.flows_over_time,
        returns the power generated in this time step
        :return:
        """

        pass

    def update(self, flows_over_time: list):

        self.flows_over_time = flows_over_time
        self.power = self.get_power()
