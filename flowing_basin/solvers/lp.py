from flowing_basin.core import Instance, Solution, Experiment
import pulp as lp
from dataclasses import dataclass


@dataclass
class LPConfiguration:

    # Objective final volumes
    volume_objectives: dict[str, float]

    # Number of periods during which the flow through the channel may not undergo more than one variation
    step_min: int


class LPModel(Experiment):
    def __init__(
        self,
        instance: Instance,
        config: LPConfiguration,
        solution: Solution = None,
    ):

        super().__init__(instance=instance, solution=solution)
        self.config = config

    # Método de prueba que posteriormente se eliminará
    def LPModel_print(self):

        I = self.instance.get_ids_of_dams()
        T = list(range(self.instance.get_largest_impact_horizon()))
        L = {
            dam_id: self.instance.get_relevant_lags_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        Tramos = {
            dam_id: list(
                range(
                    len(
                        self.instance.get_turbined_flow_obs_for_power_group(dam_id)[
                            "observed_flows"
                        ]
                    )
                    - 1
                )
            )
            for dam_id in self.instance.get_ids_of_dams()
        }
        BreakPoints = {
            dam_id: list(
                range(
                    len(
                        self.instance.get_turbined_flow_obs_for_power_group(dam_id)[
                            "observed_flows"
                        ]
                    )
                )
            )
            for dam_id in self.instance.get_ids_of_dams()
        }

        D = self.instance.get_time_step_seconds()
        Qnr = {
            dam_id: self.instance.get_all_unregulated_flows_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        QMax = {
            dam_id: self.instance.get_max_flow_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        BP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(dam_id)[
                "observed_flows"
            ]
            for dam_id in self.instance.get_ids_of_dams()
        }

        PotBP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(dam_id)[
                "observed_powers"
            ]
            for dam_id in self.instance.get_ids_of_dams()
        }

        V0 = {
            dam_id: self.instance.get_initial_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        VMax = {
            dam_id: self.instance.get_max_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        VMin = {
            dam_id: self.instance.get_min_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        TMin = self.config.step_min

        A = {
            dam_id: self.instance.get_flow_limit_coef_a_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        B = {
            dam_id: self.instance.get_flow_limit_coef_b_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        Price = self.instance.get_all_prices()

        IniLags = {
            dam_id: self.instance.get_initial_lags_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        VolFinal = {
            dam_id: self.config.volume_objectives[dam_id]
            for dam_id in self.instance.get_ids_of_dams()
        }

        Q0 = self.instance.get_all_incoming_flows()

        print(f"{I=}")
        print(f"{T=}")
        print(f"{L=}")
        print(f"{Tramos=}")
        print(f"{BreakPoints=}")
        print(f"{D=}")
        print(f"{Qnr=}")
        print(f"{QMax=}")
        print(f"{BP=}")
        print(f"{PotBP=}")
        print(f"{V0=}")
        print(f"{VMax=}")
        print(f"{VMin=}")
        print(f"{A=}")
        print(f"{B=}")
        print(f"{Price=}")
        print(f"{IniLags=}")
        print(f"{TMin=}")
        print(f"{VolFinal=}")
        print(f"{Q0=}")
        
        print(len(Price), len(T), len(Q0))
        
    def solve(self, options: dict) -> dict:

        # LP Problem
        lpproblem = lp.LpProblem("Problema_General_24h_PL", lp.LpMaximize)

        # Sets
        I = self.instance.get_ids_of_dams()
        T = list(range(self.instance.get_largest_impact_horizon()))
        L = {
            dam_id: self.instance.get_relevant_lags_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        Tramos = {
            dam_id: list(
                range(
                    len(
                        self.instance.get_turbined_flow_obs_for_power_group(dam_id)[
                            "observed_flows"
                        ]
                    )
                    - 1
                )
            )
            for dam_id in self.instance.get_ids_of_dams()
        }
        BreakPoints = {
            dam_id: list(
                range(
                    len(
                        self.instance.get_turbined_flow_obs_for_power_group(dam_id)[
                            "observed_flows"
                        ]
                    )
                )
            )
            for dam_id in self.instance.get_ids_of_dams()
        }

        # Parameters
        D = self.instance.get_time_step_seconds()
        Qnr = {
            dam_id: self.instance.get_all_unregulated_flows_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        QMax = {
            dam_id: self.instance.get_max_flow_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        BP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(dam_id)[
                "observed_flows"
            ]
            for dam_id in self.instance.get_ids_of_dams()
        }

        PotBP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(dam_id)[
                "observed_powers"
            ]
            for dam_id in self.instance.get_ids_of_dams()
        }

        V0 = {
            dam_id: self.instance.get_initial_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        VMax = {
            dam_id: self.instance.get_max_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        VMin = {
            dam_id: self.instance.get_min_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        TMin = self.config.step_min

        A = {
            dam_id: self.instance.get_flow_limit_coef_a_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        B = {
            dam_id: self.instance.get_flow_limit_coef_b_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        Price = self.instance.get_all_prices()

        IniLags = {
            dam_id: self.instance.get_initial_lags_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        VolFinal = {
            dam_id: self.config.volume_objectives[dam_id]
            for dam_id in self.instance.get_ids_of_dams()
        }

        Q0 = self.instance.get_all_incoming_flows()

        # Variables
        vol = lp.LpVariable.dicts(
            "Volumen ", [(i, t) for i in I for t in T], lowBound=0, cat=lp.LpContinuous
        )
        qe = lp.LpVariable.dicts(
            "Caudal entrada ",
            [(i, t) for i in I for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        qs = lp.LpVariable.dicts(
            "Caudal salida ",
            [(i, t) for i in I for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        pot = lp.LpVariable.dicts(
            "Potencia ",
            [(i, t) for i in I for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        qtb = lp.LpVariable.dicts(
            "Caudal turbinado ",
            [(i, t) for i in I for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        qch = lp.LpVariable.dicts(
            "Cambio caudal ", [(i, t) for i in I for t in T], cat=lp.LpContinuous
        )
        y = lp.LpVariable.dicts(
            "01Variacion ", [(i, t) for i in I for t in T], cat=lp.LpBinary
        )
        w = lp.LpVariable.dicts(
            "01Franja ",
            [
                (i, t, bp)
                for i in I
                for t in T
                for bp in range(0, BreakPoints[i][-1] + 1)
            ],
            cat=lp.LpBinary,
        )
        z = lp.LpVariable.dicts(
            "PropFranj ",
            [(i, t, bp) for i in I for t in T for bp in BreakPoints[i]],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        tpot = lp.LpVariable.dicts(
            "Potencia total ", [t for t in T], lowBound=0, cat=lp.LpContinuous
        )

        # Constraints
        for i in I:
            for t in T:
                if t == T[0]:
                    lpproblem += vol[(i, t)] == V0[t] + D * (qe[(i, t)] - qs[(i, t)])
                else:
                    lpproblem += vol[(i, t)] == vol[(i, t - 1)] + D * (
                        qe[(i, t)] - qs[(i, t)]
                    )

        for i in I:
            for t in T:
                if i == I[0]:
                    lpproblem += qe[(i, t)] == Q0[t] + Qnr[i][t]
                else:
                    lpproblem += qe[(i, t)] == qtb[(I[I.index(i)-1], t)] + Qnr[i][t]

        # TODO: improve this constraint
        for i in I:
            for t in T:
                if i == I[0]:
                    if t == T[0]:
                        lpproblem += qtb[(i, t)] == (IniLags[i][0] + IniLags[i][1]) / len(
                            L[i]
                        )
                    if t == 1:
                        lpproblem += qtb[(i, t)] == (IniLags[i][0] + qs[(i, 0)]) / len(L[i])
                    if t >= 2:
                        lpproblem += qtb[(i, t)] == lp.lpSum(
                            (qs[(i, t - l)]) * (1 / len(L[i])) for l in L[i]
                        )
                if i == "dam2":
                    if t == T[0]:
                        lpproblem += qtb[(i, t)] == (
                            IniLags[i][2]
                            + IniLags[i][3]
                            + IniLags[i][4]
                            + IniLags[i][5]
                        ) / len(L[i])
                    if t == 1:
                        lpproblem += qtb[(i, t)] == (
                            IniLags[i][1]
                            + IniLags[i][2]
                            + IniLags[i][3]
                            + IniLags[i][4]
                        ) / len(L[i])
                    if t == 2:
                        lpproblem += qtb[(i, t)] == (
                            IniLags[i][0]
                            + IniLags[i][1]
                            + IniLags[i][2]
                            + IniLags[i][3]
                        ) / len(L[i])
                    if t == 3:
                        lpproblem += qtb[(i, t)] == (
                            qs[(i, 0)] + IniLags[i][0] + IniLags[i][1] + IniLags[i][2]
                        ) / len(L[i])
                    if t == 4:
                        lpproblem += qtb[(i, t)] == (
                            qs[(i, 1)] + qs[(i, 0)] + IniLags[i][0] + IniLags[i][1]
                        ) / len(L[i])
                    if t == 5:
                        lpproblem += qtb[(i, t)] == (
                            qs[(i, 2)] + qs[(i, 1)] + qs[(i, 0)] + IniLags[i][0]
                        ) / len(L[i])
                    if t >= 6:
                        lpproblem += qtb[(i, t)] == lp.lpSum(
                            (qs[(i, t - l)]) * (1 / len(L[i])) for l in L[i]
                        )

        for i in I:
            for t in T:
                lpproblem += pot[(i, t)] == lp.lpSum(
                    z[(i, t, bp)] * PotBP[i][bp - 1] for bp in BreakPoints[i]
                )

        for i in I:
            for t in T:
                lpproblem += qtb[(i, t)] == lp.lpSum(
                    z[(i, t, bp)] * BP[i][bp - 1] for bp in BreakPoints[i]
                )

        for i in I:
            for t in T:
                lpproblem += w[(i, t, 0)] == 0
                lpproblem += w[(i, t, BreakPoints[i][-1])] == 0

        for i in I:
            for t in T:
                for bp in BreakPoints[i]:
                    lpproblem += z[(i, t, bp)] <= lp.lpSum(
                        w[(i, t, bp)] for bp in range(bp - 1, bp + 1)
                    )

        for i in I:
            for t in T:
                lpproblem += lp.lpSum(z[(i, t, bp)] for bp in BreakPoints[i]) == 1

        for i in I:
            for t in T:
                lpproblem += lp.lpSum(w[(i, t, bp)] for bp in BreakPoints[i]) == 1

        for i in I:
            for t in T:
                if t == T[0]:
                    lpproblem += qch[(i, t)] == qs[(i, t)] - IniLags[i][0]
                else:
                    lpproblem += qch[(i, t)] == qs[(i, t)] - qs[(i, t - 1)]

        for i in I:
            for t in T:
                lpproblem += qch[(i, t)] <= y[(i, t)] * QMax[i]

        for i in I:
            for t in T:
                lpproblem += -qch[(i, t)] <= y[(i, t)] * QMax[i]

        for i in I:
            for t in T:
                lpproblem += (
                    lp.lpSum(y[(i, t + t1)] for t1 in range(0, TMin) if t + t1 <= len(T)-1)
                    <= 1
                )

        for i in I:
            for t in T:
                lpproblem += vol[(i, t)] <= VMax[i]

        for i in I:
            for t in T:
                lpproblem += vol[(i, t)] >= VMin[i]

        for i in I:
            for t in T:
                lpproblem += qs[(i, t)] <= QMax[i]

        for i in I:
            for t in T:
                lpproblem += qs[(i, t)] <= A[i] * vol[(i, t)] + B[i]

        for t in T:
            lpproblem += tpot[t] == lp.lpSum(pot[(i, t)] for i in I)

        for i in I:
            for t in T:
                if t == T[-1]:
                    lpproblem += vol[(i, t)] >= VolFinal[i]

        # Objective Function
        lpproblem += lp.lpSum(tpot[t] * Price[t] for t in T)

        # Solve
        solver = lp.GUROBI(path=None, keepFiles=0, MIPGap=0.087)
        lpproblem.solve(solver)

        # Flows
        # TODO develope
        qsalida1 = [0, 0]
        qsalida2 = [0, 0]
        for var in qs.values():
            if "dam1" in var.name:
                qsalida1.append(var.value())
                print(f"{var.name}: {var.value()}")
            if "dam2" in var.name:
                qsalida2.append(var.value())
                print(f"{var.name}: {var.value()}")
        sol_dict = {
            "dams": [
                {"flows": qsalida1, "id": "dam1"},
                {"flows": qsalida2, "id": "dam2"},
            ]
        }
        self.solution = Solution.from_dict(sol_dict)

        return dict()
