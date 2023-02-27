from flowing_basin.core import Instance
from flowing_basin.tools import RiverBasin
import numpy as np
import pyswarms as ps


class PSO:
    def __init__(self):

        # Se debe crear la instancia (objeto de Instance; se pueden tomar los datos "input_example1.json")
        # y guardar las rutas a los modelos de ML (los archivos "model_E1.sav" y "model_E2.sav"),
        # y con ello crear el entorno (objeto de RiverBasin):
        self.instance = Instance.from_json("../data/input_example1.json")
        paths_power_models = {
            "dam1": "../ml_models/model_E1.sav",
            "dam2": "../ml_models/model_E2.sav",
        }
        self.river_basin = RiverBasin(
            instance=self.instance, paths_power_models=paths_power_models
        )
        self.dams = list(self.instance.data["dams"].keys())

        self.no_periods = 5  # 7 * 24 * 4

        self.swarm_size = 10
        self.dim = len(self.instance.data["dams"]) * self.no_periods
        self.iters = 1000
        # epsilon = 1.0

        self.options = {"c1": 1, "c2": 0.5, "w": 0.9}

        self.constraints = ([0 for _ in range(self.dim)], [1 for _ in range(self.dim)])

        self.flows = dict()

    def particle_to_flows(self, particle):
        flows = {
            (p, d): particle[p * len(self.dams) + self.dams.index(d)]
            for p in range(self.no_periods)
            for d in self.dams
        }
        return flows

    def get_income(self, particle):
        income = 0
        flows = self.particle_to_flows(particle)
        for p in range(self.no_periods):
            f = [flows[p, d] for d in self.dams]
            self.river_basin.update(f)
            state = self.river_basin.get_state()
            print("period {}".format(p))
            income += (
                state["price"]
                * (state["dam1"]["power"] + state["dam2"]["power"])
                * self.instance.get_time_step()
                / 3600
            )
        return income

    def opt_func(self, X):
        n_particles = X.shape[0]  # number of particles
        total_income = [self.get_income(X[i]) for i in range(n_particles)]
        return np.array(total_income)

    def run_pso(self):

        # Call an instance of PSO
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.swarm_size,
            dimensions=self.dim,
            options=self.options,
            bounds=self.constraints,
        )

        # Perform optimization
        cost, flows = optimizer.optimize(self.opt_func, iters=self.iters)
        print("cost: {}".format(cost))
        print(flows)
        print("end")


if __name__ == "__main__":

    pso = PSO()
    pso.run_pso()
    print("end")
