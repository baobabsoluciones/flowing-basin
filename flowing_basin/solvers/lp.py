from flowing_basin.core import Instance, Solution, Experiment
import pulp as lp
from dataclasses import dataclass


@dataclass
class LPConfiguration:
    # Objective final volumes
    volume_objectives: dict[str, float]

    # Penalty for unfulfilling the objective volumes, and the bonus for exceeding them (in €/m3)
    volume_shortage_penalty: float
    volume_exceedance_bonus: float

    # Penalty for each time step with the turbined flow in a limit zone (in €/occurrence)
    limit_zones_penalty: float

    # Number of periods during which the flow through the channel may not undergo more than one variation
    step_min: int

    # Gap for the solution
    MIPGap: float


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

        BreakPointsPQ = {
            dam_id: list(
                range(
                    len(
                        self.instance.get_turbined_flow_obs_for_power_group(
                            dam_id
                        )["observed_flows"]
                    )
                )
            )
            for dam_id in self.instance.get_ids_of_dams()
        }

        for key in BreakPointsPQ:
            for i in range(len(BreakPointsPQ[key])):
                BreakPointsPQ[key][i] += 1
        BreakPointsVQ = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                BreakPointsVQ[dam_id] = list(
                    range(
                        len(
                            self.instance.get_flow_limit_obs_for_channel(
                                dam_id
                            )["observed_vols"]
                        )
                    )
                )
            else:
                BreakPointsVQ[dam_id] = None

        for key in BreakPointsVQ:
            if BreakPointsVQ[key] != None:
                for i in range(len(BreakPointsVQ[key])):
                    BreakPointsVQ[key][i] += 1

        D = self.instance.get_time_step_seconds()
        Qnr = {
            dam_id: self.instance.get_all_unregulated_flows_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        QMax = {
            dam_id: self.instance.get_max_flow_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        QtBP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(
                dam_id
            )["observed_flows"]
            for dam_id in self.instance.get_ids_of_dams()
        }

        PotBP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(
                dam_id
            )["observed_powers"]
            for dam_id in self.instance.get_ids_of_dams()
        }

        ZonaLimitePQ = {}
        for dam_id in self.instance.get_ids_of_dams():
            ZonaLimitePQ[dam_id] = []
        for i in I:
            for bp in BreakPointsPQ[i]:
                if bp != BreakPointsPQ[i][-1]:
                    if PotBP[i][bp - 1] == PotBP[i][bp]:
                        ZonaLimitePQ[i].append(bp)

        QMaxBP = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                QMaxBP[dam_id] = self.instance.get_flow_limit_obs_for_channel(
                    dam_id
                )["observed_flows"]
            else:
                QMaxBP[dam_id] = None

        VolBP = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                VolBP[dam_id] = self.instance.get_flow_limit_obs_for_channel(
                    dam_id
                )["observed_vols"]
            else:
                VolBP[dam_id] = None

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
        # TODO delete A and B in schema

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

        BonusVol = self.config.volume_exceedance_bonus

        PenVol = self.config.volume_shortage_penalty

        PenZL = self.config.limit_zones_penalty

        print(f"{I=}")
        print(f"{T=}")
        print(f"{L=}")
        print(f"{BreakPointsPQ=}")
        print(f"{BreakPointsVQ=}")
        print(f"{D=}")
        print(f"{Qnr=}")
        print(f"{QMax=}")
        print(f"{QtBP=}")
        print(f"{PotBP=}")
        print(f"{ZonaLimitePQ=}")
        print(f"{QMaxBP=}")
        print(f"{VolBP=}")
        print(f"{V0=}")
        print(f"{VMax=}")
        print(f"{VMin=}")
        print(f"{Price=}")
        print(f"{IniLags=}")
        print(f"{TMin=}")
        print(f"{VolFinal=}")
        print(f"{Q0=}")
        print(f"{BonusVol=}")
        print(f"{PenVol=}")
        print(f"{PenZL=}")

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

        BreakPointsPQ = {
            dam_id: list(
                range(
                    len(
                        self.instance.get_turbined_flow_obs_for_power_group(
                            dam_id
                        )["observed_flows"]
                    )
                )
            )
            for dam_id in self.instance.get_ids_of_dams()
        }

        for key in BreakPointsPQ:
            for i in range(len(BreakPointsPQ[key])):
                BreakPointsPQ[key][i] += 1
        BreakPointsVQ = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                BreakPointsVQ[dam_id] = list(
                    range(
                        len(
                            self.instance.get_flow_limit_obs_for_channel(
                                dam_id
                            )["observed_vols"]
                        )
                    )
                )
            else:
                BreakPointsVQ[dam_id] = None

        for key in BreakPointsVQ:
            if BreakPointsVQ[key] != None:
                for i in range(len(BreakPointsVQ[key])):
                    BreakPointsVQ[key][i] += 1

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
        QtBP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(
                dam_id
            )["observed_flows"]
            for dam_id in self.instance.get_ids_of_dams()
        }

        PotBP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(
                dam_id
            )["observed_powers"]
            for dam_id in self.instance.get_ids_of_dams()
        }

        ZonaLimitePQ = {}
        for dam_id in self.instance.get_ids_of_dams():
            ZonaLimitePQ[dam_id] = []
        for i in I:
            for bp in BreakPointsPQ[i]:
                if bp != BreakPointsPQ[i][-1]:
                    if PotBP[i][bp - 1] == PotBP[i][bp]:
                        ZonaLimitePQ[i].append(bp)

        QMaxBP = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                QMaxBP[dam_id] = self.instance.get_flow_limit_obs_for_channel(
                    dam_id
                )["observed_flows"]
            else:
                QMaxBP[dam_id] = None

        VolBP = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                VolBP[dam_id] = self.instance.get_flow_limit_obs_for_channel(
                    dam_id
                )["observed_vols"]
            else:
                VolBP[dam_id] = None

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
        # TODO delete A and B in schema

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

        BonusVol = self.config.volume_exceedance_bonus

        PenVol = self.config.volume_shortage_penalty

        PenZL = self.config.limit_zones_penalty

        # Variables
        vol = lp.LpVariable.dicts(
            "Volumen ",
            [(i, t) for i in I for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
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
            "Cambio caudal ",
            [(i, t) for i in I for t in T],
            cat=lp.LpContinuous,
        )
        y = lp.LpVariable.dicts(
            "01Variacion ", [(i, t) for i in I for t in T], cat=lp.LpBinary
        )
        w_pq = lp.LpVariable.dicts(
            "01Franja_PQ ",
            [
                (i, t, bp)
                for i in I
                for t in T
                for bp in range(0, BreakPointsPQ[i][-1] + 1)
            ],
            cat=lp.LpBinary,
        )
        z_pq = lp.LpVariable.dicts(
            "PropFranj_PQ ",
            [(i, t, bp) for i in I for t in T for bp in BreakPointsPQ[i]],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        w_vq = lp.LpVariable.dicts(
            "01Franja_VQ ",
            [
                (i, t, bp)
                for i in I
                if QMaxBP[i] != None
                for t in T
                for bp in range(0, BreakPointsVQ[i][-1] + 1)
            ],
            cat=lp.LpBinary,
        )
        z_vq = lp.LpVariable.dicts(
            "PropFranj_VQ ",
            [
                (i, t, bp)
                for i in I
                if QMaxBP[i] != None
                for t in T
                for bp in BreakPointsVQ[i]
            ],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        q_max_vol = lp.LpVariable.dicts(
            "Caudal máximo volumen ",
            [(i, t) for i in I if QMaxBP[i] != None for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        tpot = lp.LpVariable.dicts(
            "Potencia total ", [t for t in T], lowBound=0, cat=lp.LpContinuous
        )
        pos_desv = lp.LpVariable.dicts(
            "Desviación positiva ",
            [i for i in I],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        neg_desv = lp.LpVariable.dicts(
            "Desviación negativa ",
            [i for i in I],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        cost_desv = lp.LpVariable.dicts(
            "Coste desviación ",
            [i for i in I],
            cat=lp.LpContinuous,
        )

        zl_tot = lp.LpVariable.dicts(
            "Zonas límites totales ",
            [i for i in I],
            cat=lp.LpInteger,
        )

        # Constraints
        for i in I:
            for t in T:
                if t == T[0]:
                    lpproblem += vol[(i, t)] == V0[i] + D * (
                        qe[(i, t)] - qs[(i, t)]
                    )
                else:
                    lpproblem += vol[(i, t)] == vol[(i, t - 1)] + D * (
                        qe[(i, t)] - qs[(i, t)]
                    )

        # TODO: edit latex
        for i in I:
            for t in T:
                if i == I[0]:
                    lpproblem += qe[(i, t)] == Q0[t] + Qnr[i][t]
                else:
                    lpproblem += (
                        qe[(i, t)] == qtb[(I[I.index(i) - 1], t)] + Qnr[i][t]
                    )

        # TODO: improve this constraint
        for i in I:
            for t in T:
                if i == I[0]:
                    if t == T[0]:
                        lpproblem += qtb[(i, t)] == (
                            IniLags[i][0] + IniLags[i][1]
                        ) / len(L[i])
                    if t == 1:
                        lpproblem += qtb[(i, t)] == (
                            IniLags[i][0] + qs[(i, 0)]
                        ) / len(L[i])
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
                            qs[(i, 0)]
                            + IniLags[i][0]
                            + IniLags[i][1]
                            + IniLags[i][2]
                        ) / len(L[i])
                    if t == 4:
                        lpproblem += qtb[(i, t)] == (
                            qs[(i, 1)]
                            + qs[(i, 0)]
                            + IniLags[i][0]
                            + IniLags[i][1]
                        ) / len(L[i])
                    if t == 5:
                        lpproblem += qtb[(i, t)] == (
                            qs[(i, 2)]
                            + qs[(i, 1)]
                            + qs[(i, 0)]
                            + IniLags[i][0]
                        ) / len(L[i])
                    if t >= 6:
                        lpproblem += qtb[(i, t)] == lp.lpSum(
                            (qs[(i, t - l)]) * (1 / len(L[i])) for l in L[i]
                        )

        for i in I:
            for t in T:
                lpproblem += pot[(i, t)] == lp.lpSum(
                    z_pq[(i, t, bp)] * PotBP[i][bp - 1]
                    for bp in BreakPointsPQ[i]
                )

        for i in I:
            for t in T:
                lpproblem += qtb[(i, t)] == lp.lpSum(
                    z_pq[(i, t, bp)] * QtBP[i][bp - 1]
                    for bp in BreakPointsPQ[i]
                )

        for i in I:
            for t in T:
                lpproblem += w_pq[(i, t, 0)] == 0
                lpproblem += w_pq[(i, t, BreakPointsPQ[i][-1])] == 0

        for i in I:
            for t in T:
                for bp in BreakPointsPQ[i]:
                    lpproblem += (
                        z_pq[(i, t, bp)]
                        <= w_pq[(i, t, bp - 1)] + w_pq[(i, t, bp)]
                    )

        for i in I:
            for t in T:
                lpproblem += (
                    lp.lpSum(z_pq[(i, t, bp)] for bp in BreakPointsPQ[i]) == 1
                )

        for i in I:
            for t in T:
                lpproblem += (
                    lp.lpSum(w_pq[(i, t, bp)] for bp in BreakPointsPQ[i]) == 1
                )

        for i in I:
            for t in T:
                if QMaxBP[i] != None:
                    lpproblem += q_max_vol[(i, t)] == lp.lpSum(
                        z_vq[(i, t, bp)] * QMaxBP[i][bp - 1]
                        for bp in BreakPointsVQ[i]
                    )

        for i in I:
            for t in T:
                if QMaxBP[i] != None:
                    lpproblem += vol[(i, t)] == lp.lpSum(
                        z_vq[(i, t, bp)] * VolBP[i][bp - 1]
                        for bp in BreakPointsVQ[i]
                    )

        for i in I:
            for t in T:
                if QMaxBP[i] != None:
                    lpproblem += w_vq[(i, t, 0)] == 0
                    lpproblem += w_vq[(i, t, BreakPointsVQ[i][-1])] == 0

        for i in I:
            for t in T:
                if QMaxBP[i] != None:
                    for bp in BreakPointsVQ[i]:
                        lpproblem += z_vq[(i, t, bp)] <= lp.lpSum(
                            w_vq[(i, t, tr)] for tr in range(bp - 1, bp + 1)
                        )

        for i in I:
            for t in T:
                if QMaxBP[i] != None:
                    lpproblem += (
                        lp.lpSum(z_vq[(i, t, bp)] for bp in BreakPointsVQ[i])
                        == 1
                    )

        for i in I:
            for t in T:
                if QMaxBP[i] != None:
                    lpproblem += (
                        lp.lpSum(w_vq[(i, t, bp)] for bp in BreakPointsVQ[i])
                        == 1
                    )

        for i in I:
            for t in T:
                if t == T[0]:
                    lpproblem += qch[(i, t)] == 0  # qs[(i, t)] - IniLags[i][0]
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
                    lp.lpSum(
                        y[(i, t + t1)]
                        for t1 in range(0, TMin)
                        if t + t1 <= len(T) - 1
                    )
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
                if QMaxBP[i] != None:
                    lpproblem += qs[(i, t)] <= q_max_vol[(i, t)]

        for t in T:
            lpproblem += tpot[t] == lp.lpSum(pot[(i, t)] for i in I)

        for i in I:
            for t in T:
                if t == T[-1]:
                    lpproblem += (
                        vol[i, t] == VolFinal[i] + pos_desv[i] - neg_desv[i]
                    )

        for i in I:
            lpproblem += (
                cost_desv[i] == pos_desv[i] * BonusVol - neg_desv[i] * PenVol
            )
        for i in I:
            lpproblem += zl_tot[i] == lp.lpSum(
                w_pq[(i, t, bp)] for t in T for bp in ZonaLimitePQ[i]
            )

        # Objective Function
        lpproblem += lp.lpSum(
            tpot[t] * Price[t] + cost_desv[i] - zl_tot[i] * PenZL
            for t in T
            for i in I
        )

        # Solve
        solver = lp.GUROBI(path=None, keepFiles=0, MIPGap=self.config.MIPGap)
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
