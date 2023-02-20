from cornflow_client.core.tools import load_json
import os


class Training:

    def __init__(self):

        self.constants = load_json(
            os.path.join(os.path.dirname(__file__), "../../data/constants.json")
        )

        # TODO: finish this
