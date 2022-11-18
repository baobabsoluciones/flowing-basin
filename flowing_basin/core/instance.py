import pickle

from cornflow_client import InstanceCore, get_empty_schema


class Instance(InstanceCore):
    schema = get_empty_schema()
    schema_checks = get_empty_schema()

    @classmethod
    def from_dict(cls, data):
        return cls(data)

    def to_dict(self):
        return pickle.loads(pickle.dumps(self.data, -1))
