from flowing_basin.core import Instance, Solution, Experiment
import pulp as lp
from dataclasses import dataclass, asdict
import tempfile
import os


@dataclass
class LPConfiguration:
    # Objective final volumes
    volume_objectives: dict[str, float]

    # Penalty for unfulfilling the objective volumes, and the bonus for exceeding them (in €/m3)
    volume_shortage_penalty: float
    volume_exceedance_bonus: float

    # Penalty for each power group startup and for each time step with the turbined 
    #flow in a limit zone (in €/occurrence)
    startups_penalty: float
    limit_zones_penalty: float

    # Gap for the solution
    MIPGap: float

    # Solver timeout
    max_time: float

    # Number of periods during which the flow through the channel may not vary
    # in order to change the sense of the flow's change
    flow_smoothing: int

    # Number of periods during which the flow through the channel may not undergo more than one variation
    # step_min: int = None


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
        # Sets
        """
        Conjunto embalses: I
        """
        I = self.instance.get_ids_of_dams()
        """
        Conjunto franjas de tiempo: T
        """
        T = list(range(self.instance.get_largest_impact_horizon()))
        """
        Conjunto lags relevantes: L
        """
        L = {
            dam_id: self.instance.get_verification_lags_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Conjunto breakpoints en curva Potencia - Caudal turbinado: BreakPointsPQ
        """
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
        # Ajustamos este conjunto para corregir indexación
        for key in BreakPointsPQ:
            for i in range(len(BreakPointsPQ[key])):
                BreakPointsPQ[key][i] += 1
        """
        Conjunto breakpoints en curva Volumen - Caudal máximo: BreakPointsVQ
        """
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
        # Ajustamos este conjunto para corregir indexación
        for key in BreakPointsVQ:
            if BreakPointsVQ[key] != None:
                for i in range(len(BreakPointsVQ[key])):
                    BreakPointsVQ[key][i] += 1

        # Parameters
        """
        Parámetro duración de cada franja (s): D
        """
        D = self.instance.get_time_step_seconds()
        """
        Parámetro franja donde se compara con volumen objetivo: D_1
        """
        D_1 = self.instance.get_decision_horizon()
        """
        Parámetro caudal no regulado (m3/s): Qnr
        """
        Qnr = {
            dam_id: self.instance.get_all_unregulated_flows_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro caudal máximo permitido por canal (m3/s): QMax
        """
        QMax = {
            dam_id: self.instance.get_max_flow_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro caudal turbinado en los breakpoints
        de la curva Potencia - Caudal turbinado (m3/s): QtBP
        """
        QtBP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(
                dam_id
            )["observed_flows"]
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro potencia en los breakpoints
        de la curva Potencia - Caudal turbinado (MWh): PotBP
        """
        PotBP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(
                dam_id
            )["observed_powers"]
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro caudal máximo en los breakpoints
        de la curva Volumen - Caudal máximo (m3/s): QmaxBP
        """
        QmaxBP = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                QmaxBP[dam_id] = self.instance.get_flow_limit_obs_for_channel(
                    dam_id
                )["observed_flows"]
            else:
                QmaxBP[dam_id] = None
        """
        Parámetro volumen en los breakpoints
        de la curva Volumen - Caudal máximo (m3): VolBP
        """
        VolBP = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                VolBP[dam_id] = self.instance.get_flow_limit_obs_for_channel(
                    dam_id
                )["observed_vols"]
            else:
                VolBP[dam_id] = None
        """
        Parámetro volumen inicial (m3): V0
        """
        V0 = {
            dam_id: self.instance.get_initial_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro volumen máximo (m3): VMax
        """
        VMax = {
            dam_id: self.instance.get_max_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro volumen mínimo (m3): VMin
        """
        VMin = {
            dam_id: self.instance.get_min_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro franjas sin cambio en el caudal de salida: TMin
        """
        # TMin = self.config.step_min
        """
        Parámetro franjas sin cambio en el caudal de salida: K
        """
        K = self.config.flow_smoothing
        """
        Parámetro precio en cada franja (€/MWh): Price
        """
        Price = self.instance.get_all_prices()
        """
        Parámetro lags iniciales en el caudal de entrada
        a cada embalse (m3/s): IniLags
        """
        IniLags = {
            dam_id: self.instance.get_initial_lags_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro volumen final objetivo (m3): VolFinal
        """
        VolFinal = {
            dam_id: self.config.volume_objectives[dam_id]
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro caudal entrante al primer embalse (m3/s): Q0
        """
        Q0 = self.instance.get_all_incoming_flows()
        """
        Parámetro bonus por exceder volumen objetivo (€/m3): BonusVol
        """
        BonusVol = self.config.volume_exceedance_bonus
        """
        Parámetro penalización por no llegar a volumen objetivo (€/m3): PenVol
        """
        PenVol = self.config.volume_shortage_penalty
        """
        Parámetro penalización por franja en zona límite
        (€/nºfranjas): PenZL
        """
        PenZL = self.config.limit_zones_penalty
        """
        Parámetro penalización por arranque de grupo de potencia
        (€/nºarranques): PenSU
        """
        PenSU = self.config.startups_penalty
        """
        Parámetro caudal turbinado de apagado de turbina (m3/s): shutdown_flows
        """
        shutdown_flows = {
            dam_id: self.instance.get_shutdown_flows_of_power_group(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro caudal turbinado de arranque de turbina (m3/s): startup_flows
        """
        startup_flows = {
            dam_id: self.instance.get_startup_flows_of_power_group(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        # Proceso para eliminar la desviación de decimales de startup_flows y shutdown_flows
        for i in I:
            for y in range(len(startup_flows[i])):
                for w in range(len(QtBP[i])):
                    if (startup_flows[i][y] - QtBP[i][w]) <= 0.1 and (startup_flows[i][y] - QtBP[i][w]) >= -0.1:
                        QtBP[i][w] = startup_flows[i][y]

        for i in I:
            for y in range(len(shutdown_flows[i])):
                for w in range(len(QtBP[i])):
                    if (shutdown_flows[i][y] - QtBP[i][w]) <= 0.1 and (shutdown_flows[i][y] - QtBP[i][w]) >= -0.1:
                        QtBP[i][w] = shutdown_flows[i][y]
        """
        Parámetro zonas límite de cada embalse
        en la curva Potencia - Caudal turbinado: ZonaLimitePQ
        """
        ZonaLimitePQ = {}
        for dam_id in self.instance.get_ids_of_dams():
            ZonaLimitePQ[dam_id] = []
        for i in I:
            for bp in QtBP[i]:
                if bp in shutdown_flows[i]:
                    ZonaLimitePQ[i].append(QtBP[i].index(bp)+1)
        # Proceso para eliminar la primera franja
        # (antes del primer arranque) de este parámetro
        for i in I:
            ZonaLimitePQ[i].pop(0)
        """
        Parámetro conjunto de franjas para cada powergroup
        de cada embalse: FranjasGrupos
        """
        FranjasGrupos1 = {}
        FranjasGrupos = {}
        for i in I:
            FranjasGrupos1[i] = {}
            for gp in range(len(startup_flows[i])):
                FranjasGrupos1[i]["Grupo_potencia" + str(gp + 1)] = []
                for bp in QtBP[i]:
                    if gp == (len(startup_flows[i]) - 1):
                        if bp >= startup_flows[i][gp]:
                            FranjasGrupos1[i]["Grupo_potencia" + str(gp + 1)].append(QtBP[i].index(bp) + 1)
                            if bp == QtBP[i][-1]:
                                FranjasGrupos1[i]["Grupo_potencia" + str(gp + 1)].pop(-1)
                    else:
                        if bp >= startup_flows[i][gp] and bp < startup_flows[i][gp + 1]:
                            FranjasGrupos1[i]["Grupo_potencia" + str(gp + 1)].append(QtBP[i].index(bp) + 1)
            FranjasGrupos[i] = {"Grupo_potencia0": [1]}
            FranjasGrupos[i].update(FranjasGrupos1[i])

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
        print(f"{QmaxBP=}")
        print(f"{VolBP=}")
        print(f"{V0=}")
        print(f"{VMax=}")
        print(f"{VMin=}")
        # print(f"{TMin=}")
        print(f"{K=}")
        print(f"{Price=}")
        print(f"{IniLags=}")
        print(f"{VolFinal=}")
        print(f"{Q0=}")
        print(f"{BonusVol=}")
        print(f"{PenVol=}")
        print(f"{PenZL=}")
        print(f"{PenSU=}")
        print(f"{shutdown_flows=}")
        print(f"{startup_flows=}")
        print(f"{ZonaLimitePQ=}")
        print(f"{FranjasGrupos=}")
        print(f"{D_1=}")

        print(len(Price), len(T), len(Q0), len(L["dam1"]), len(L["dam2"]))

    @staticmethod
    def parse_gurobi_output(output: str) -> tuple[list[float], list[float], list[float]]:

        """
        Parse the output of the Gurobi solver
        to get the objective function values, gap values, and time stamps
        during the execution.
        """

        # Position of relevant info in table (from the right)
        LOWER_BOUND_POS = 5
        GAP_POS = 3
        TIME_POS = 1

        max_pos = max(LOWER_BOUND_POS, GAP_POS, TIME_POS)

        incumbent_values = []
        gap_values = []
        time_values = []

        for line in output.split('\n'):

            values = line.split()

            # Continue to next iteration if line does not have the desired info (GAP% and TIMEs)
            if len(values) < max_pos:
                continue
            if values[-GAP_POS].find("%") == -1 or values[-TIME_POS].find("s") == -1:
                continue

            incumbent_values.append(float(values[-LOWER_BOUND_POS]))
            gap_values.append(float(values[-GAP_POS].replace("%", "")))
            time_values.append(float(values[-TIME_POS].replace("s", "")))

        return incumbent_values, gap_values, time_values

    def solve(self, options: dict = None) -> dict:

        # LP Problem
        lpproblem = lp.LpProblem("Problema_General_24h_PL", lp.LpMaximize)

        # Sets
        """
        Conjunto embalses: I
        """
        I = self.instance.get_ids_of_dams()
        """
        Conjunto franjas de tiempo: T
        """
        T = list(range(self.instance.get_largest_impact_horizon()))
        """
        Conjunto lags relevantes: L
        """
        L = {
            dam_id: self.instance.get_verification_lags_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Conjunto breakpoints en curva Potencia - Caudal turbinado: BreakPointsPQ
        """
        BreakPointsPQ = {
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
        # Ajustamos este conjunto para corregir indexación
        for key in BreakPointsPQ:
            for i in range(len(BreakPointsPQ[key])):
                BreakPointsPQ[key][i] += 1
        """
        Conjunto breakpoints en curva Volumen - Caudal máximo: BreakPointsVQ
        """
        BreakPointsVQ = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                BreakPointsVQ[dam_id] = list(
                    range(
                        len(
                            self.instance.get_flow_limit_obs_for_channel(dam_id)[
                                "observed_vols"
                            ]
                        )
                    )
                )
            else:
                BreakPointsVQ[dam_id] = None
        # Ajustamos este conjunto para corregir indexación
        for key in BreakPointsVQ:
            if BreakPointsVQ[key] != None:
                for i in range(len(BreakPointsVQ[key])):
                    BreakPointsVQ[key][i] += 1
        # Parameters
        """
        Parámetro duración de cada franja (s): D
        """
        D = self.instance.get_time_step_seconds()
        """
        Parámetro franja donde se compara con volumen objetivo: D_1
        """
        D_1 = self.instance.get_decision_horizon()
        """
        Parámetro caudal no regulado (m3/s): Qnr
        """
        Qnr = {
            dam_id: self.instance.get_all_unregulated_flows_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro caudal máximo permitido por canal (m3/s): QMax
        """
        QMax = {
            dam_id: self.instance.get_max_flow_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro caudal turbinado en los breakpoints
        de la curva Potencia - Caudal turbinado (m3/s): QtBP
        """
        QtBP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(dam_id)[
                "observed_flows"
            ]
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro potencia en los breakpoints
        de la curva Potencia - Caudal turbinado (MWh): PotBP
        """
        PotBP = {
            dam_id: self.instance.get_turbined_flow_obs_for_power_group(dam_id)[
                "observed_powers"
            ]
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro caudal máximo en los breakpoints
        de la curva Volumen - Caudal máximo (m3/s): QmaxBP
        """
        QmaxBP = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                QmaxBP[dam_id] = self.instance.get_flow_limit_obs_for_channel(dam_id)[
                    "observed_flows"
                ]
            else:
                QmaxBP[dam_id] = None
        """
        Parámetro volumen en los breakpoints
        de la curva Volumen - Caudal máximo (m3): VolBP
        """
        VolBP = {}
        for dam_id in self.instance.get_ids_of_dams():
            if self.instance.get_flow_limit_obs_for_channel(dam_id) != None:
                VolBP[dam_id] = self.instance.get_flow_limit_obs_for_channel(dam_id)[
                    "observed_vols"
                ]
            else:
                VolBP[dam_id] = None
        """
        Parámetro volumen inicial (m3): V0
        """
        V0 = {
            dam_id: self.instance.get_initial_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro volumen máximo (m3): VMax
        """
        VMax = {
            dam_id: self.instance.get_max_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro volumen mínimo (m3): VMin
        """
        VMin = {
            dam_id: self.instance.get_min_vol_of_dam(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        #CON TMIN
        """
        Parámetro franjas sin cambio en el caudal de salida: TMin
        """
        #TMin = self.config.step_min

        #CON K FORMA CARLOS Y ÁLVARO
        """
        Parámetro franjas sin cambio en el caudal de salida: K
        """
        K = self.config.flow_smoothing


        """
        Parámetro precio en cada franja (€/MWh): Price
        """
        Price = self.instance.get_all_prices()
        """
        Parámetro lags iniciales en el caudal de entrada
        a cada embalse (m3/s): IniLags
        """
        IniLags = {
            dam_id: self.instance.get_initial_lags_of_channel(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro volumen final objetivo (m3): VolFinal
        """
        VolFinal = {
            dam_id: self.config.volume_objectives[dam_id]
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro caudal entrante al primer embalse (m3/s): Q0
        """
        Q0 = self.instance.get_all_incoming_flows()
        """
        Parámetro bonus por exceder volumen objetivo (€/m3): BonusVol
        """
        BonusVol = self.config.volume_exceedance_bonus
        """
        Parámetro penalización por no llegar a volumen objetivo (€/m3): PenVol
        """
        PenVol = self.config.volume_shortage_penalty
        """
        Parámetro penalización por franja en zona límite
        (€/nºfranjas): PenZL
        """
        PenZL = self.config.limit_zones_penalty
        """
        Parámetro penalización por arranque de grupo de potencia
        (€/nºarranques): PenSU
        """
        PenSU = self.config.startups_penalty
        """
        Parámetro caudal turbinado de apagado de turbina (m3/s): shutdown_flows
        """
        shutdown_flows = {
            dam_id: self.instance.get_shutdown_flows_of_power_group(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }
        """
        Parámetro caudal turbinado de arranque de turbina (m3/s): startup_flows
        """
        startup_flows = {
            dam_id: self.instance.get_startup_flows_of_power_group(dam_id)
            for dam_id in self.instance.get_ids_of_dams()
        }

        # Proceso para eliminar la desviación de decimales de startup_flows y shutdown_flows
        for i in I:
            for y in range(len(startup_flows[i])):
                for w in range(len(QtBP[i])):
                    if (startup_flows[i][y] - QtBP[i][w]) <= 0.1 and (
                            startup_flows[i][y] - QtBP[i][w]
                    ) >= -0.1:
                        startup_flows[i][y] = QtBP[i][w]
        for i in I:
            for y in range(len(shutdown_flows[i])):
                for w in range(len(QtBP[i])):
                    if (shutdown_flows[i][y] - QtBP[i][w]) <= 0.1 and (
                            shutdown_flows[i][y] - QtBP[i][w]
                    ) >= -0.1:
                        shutdown_flows[i][y] = QtBP[i][w]
        """
        Parámetro zonas límite de cada embalse
        en la curva Potencia - Caudal turbinado: ZonaLimitePQ
        """
        ZonaLimitePQ = {}
        for dam_id in self.instance.get_ids_of_dams():
            ZonaLimitePQ[dam_id] = []
        for i in I:
            for bp in QtBP[i]:
                if bp in shutdown_flows[i]:
                    ZonaLimitePQ[i].append(QtBP[i].index(bp) + 1)
        # Proceso para eliminar la primera franja
        # (antes del primer arranque) de este parámetro
        for i in I:
            ZonaLimitePQ[i].pop(0)
        """
        Parámetro conjunto de franjas para cada powergroup
        de cada embalse: FranjasGrupos
        """
        FranjasGrupos1 = {}
        FranjasGrupos = {}
        for i in I:
            FranjasGrupos1[i] = {}
            for gp in range(len(startup_flows[i])):
                FranjasGrupos1[i]["Grupo_potencia" + str(gp + 1)] = []
                for bp in QtBP[i]:
                    if gp == (len(startup_flows[i]) - 1):
                        if bp >= startup_flows[i][gp]:
                            FranjasGrupos1[i]["Grupo_potencia" + str(gp + 1)].append(
                                QtBP[i].index(bp) + 1
                            )
                            if bp == QtBP[i][-1]:
                                FranjasGrupos1[i]["Grupo_potencia" + str(gp + 1)].pop(-1)
                    else:
                        if bp >= startup_flows[i][gp] and bp < startup_flows[i][gp + 1]:
                            FranjasGrupos1[i]["Grupo_potencia" + str(gp + 1)].append(
                                QtBP[i].index(bp) + 1
                            )
            FranjasGrupos[i] = {"Grupo_potencia0": [1]}
            FranjasGrupos[i].update(FranjasGrupos1[i])
        # Variables
        """
        Variable volumen en cada embalse en cada franja de tiempo
        (m3): vol
        """
        vol = lp.LpVariable.dicts(
            "Volumen ", [(i, t) for i in I for t in T], lowBound=0, cat=lp.LpContinuous
        )
        """
        Variable caudal entrada en cada embalse
        en cada franja de tiempo (m3/s): qe
        """
        qe = lp.LpVariable.dicts(
            "Caudal entrada ",
            [(i, t) for i in I for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        """
        Variable caudal salida en cada embalse
        en cada franja de tiempo (m3/s): qe
        """
        qs = lp.LpVariable.dicts(
            "Caudal salida ",
            [(i, t) for i in I for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        """
        Variable potencia generada en cada embalse
        en cada franja de tiempo (MWh): pot
        """
        pot = lp.LpVariable.dicts(
            "Potencia ",
            [(i, t) for i in I for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        """
        Variable caudal turbinado en cada embalse
        en cada franja de tiempo (m3/s): qtb
        """
        qtb = lp.LpVariable.dicts(
            "Caudal turbinado ",
            [(i, t) for i in I for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        """
        Variable variación de caudal en cada embalse
        en cada franja de tiempo (m3/s): qch
        """
        qch = lp.LpVariable.dicts(
            "Cambio caudal ", [(i, t) for i in I for t in T], cat=lp.LpContinuous
        )

        #CON TMIN
        """
        Variable binaria: y
        1 si hay variación de caudal en la franja
        0 si no hay variación de caudal en la franja
        """
        #y = lp.LpVariable.dicts("01Variacion ", [(i, t) for i in I for t in T], cat=lp.LpBinary)

        #CON K FORMA CARLOS
        """
        Variable binaria: x+
        1 si hay variación positiva de caudal en la franja
        0 si no hay variación positiva de caudal en la franja
        """
        x_pos = lp.LpVariable.dicts("01VariacionPos ", [(i, t) for i in I for t in T], cat=lp.LpBinary)
        """
        Variable binaria: x-
        1 si hay variación negativa de caudal en la franja
        0 si no hay variación negativa de caudal en la franja
        """
        x_neg = lp.LpVariable.dicts("01VariacionNeg ", [(i, t) for i in I for t in T], cat=lp.LpBinary)

        #CON K FORMA ÁLVARO
        """
        Variable binaria: x+
        1 si hay variación nula de caudal en la franja
        0 si no hay variación nula de caudal en la franja
        """
        #x_0 = lp.LpVariable.dicts("01VariacionNula ", [(i, t) for i in I for t in T], cat=lp.LpBinary)


        """
        Variable asociada al IP with Piecewise Linear Functions
        de Winston en relación a la curva Potencia - Caudal turbinado
        """
        w_pq = lp.LpVariable.dicts(
            "01Franja_PQ ",
            [(i, t, bp) for i in I for t in T for bp in range(0, BreakPointsPQ[i][-1] + 1)],
            cat=lp.LpBinary,
        )
        """
        Variable asociada al IP with Piecewise Linear Functions
        de Winston en relación a la curva Potencia - Caudal turbinado
        """
        z_pq = lp.LpVariable.dicts(
            "PropFranj_PQ ",
            [(i, t, bp) for i in I for t in T for bp in BreakPointsPQ[i]],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        """
        Variable asociada al IP with Piecewise Linear Functions
        de Winston en relación a la curva Volumen - Caudal máximo
        """
        w_vq = lp.LpVariable.dicts(
            "01Franja_VQ ",
            [
                (i, t, bp)
                for i in I
                if QmaxBP[i] != None
                for t in T
                for bp in range(0, BreakPointsVQ[i][-1] + 1)
            ],
            cat=lp.LpBinary,
        )
        """
        Variable asociada al IP with Piecewise Linear Functions
        de Winston en relación a la curva Volumen - Caudal máximo
        """
        z_vq = lp.LpVariable.dicts(
            "PropFranj_VQ ",
            [(i, t, bp) for i in I if QmaxBP[i] != None for t in T for bp in BreakPointsVQ[i]],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        """
        Variable caudal máximo permitido por canal
        en función del volumen del embalse en cada franja
        (m3/s): q_max_vol
        """
        q_max_vol = lp.LpVariable.dicts(
            "Caudal máximo volumen ",
            [(i, t) for i in I if QmaxBP[i] != None for t in T],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        """
        Variable desviación positiva respecto a volumen objetivo
        (m3): pos_desv
        """
        pos_desv = lp.LpVariable.dicts(
            "Desviación positiva del ", [i for i in I], lowBound=0, cat=lp.LpContinuous
        )
        """
        Variable desviación negativa respecto a volumen objetivo
        (m3): pos_desv
        """
        neg_desv = lp.LpVariable.dicts(
            "Desviación negativa del ",
            [i for i in I],
            lowBound=0,
            cat=lp.LpContinuous,
        )
        """
        Variable beneficio por falta o exceso respecto
        al volumen objetivo (€): ben_desv
        """
        ben_desv = lp.LpVariable.dicts(
            "Beneficio por desviación volumen del ",
            [i for i in I],
            cat=lp.LpContinuous,
        )
        """
        Variable número total de franjas en zonas límite: zl_tot
        """
        zl_tot = lp.LpVariable.dicts(
            "Zonas límites totales del ",
            [i for i in I],
            cat=lp.LpInteger,
        )
        """
        Variable binaria: pwch
        1 si se ha arrancado un powergroup en la franja
        0 si no se ha arrancado un powergroup en la franja
        """
        pwch = lp.LpVariable.dicts(
            "01Arranque PG ",
            [(i, t, pg) for i in I for t in T for pg in FranjasGrupos[i]],
            cat=lp.LpBinary,
        )
        """
        Variable número total de arranques por embalse
        """
        pwch_tot = lp.LpVariable.dicts(
            "Arranque totales del ",
            [i for i in I],
            cat=lp.LpInteger,
        )
        """
        Variable potencia total generada en cada embalse
        (MWh): pot_embalse
        """
        pot_embalse = lp.LpVariable.dicts(
            "Potencia total del ",
            [i for i in I],
            cat=lp.LpContinuous,
        )

        # Constraints
        """
        Restricción balance de volumen
        """
        for i in I:
            for t in T:
                if t == T[0]:
                    lpproblem += vol[(i, t)] <= V0[i] + D * (qe[(i, t)] - qs[(i, t)])
                else:
                    lpproblem += vol[(i, t)] <= vol[(i, t - 1)] + D * (qe[(i, t)] - qs[(i, t)])
        """
        Restricción caudal de entrada
        """
        for i in I:
            for t in T:
                if i == I[0]:
                    lpproblem += qe[(i, t)] == Q0[t] + Qnr[i][t]
                else:
                    lpproblem += qe[(i, t)] == qtb[(I[I.index(i) - 1], t)] + Qnr[i][t]
        """
        Restricción caudal turbinado en base a lags relevantes
        """
        for i in I:
            for t in T:
                lpproblem += qtb[(i, t)] == lp.lpSum(
                    IniLags[i][l - 1 - t] for l in L[i] if l - 1 - t >= 0
                ) * (1 / len(L[i])) + lp.lpSum(qs[(i, t - l)] for l in L[i] if t - l >= 0) * (
                                     1 / len(L[i])
                             )
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Potencia - Caudal turbinado
        """
        for i in I:
            for t in T:
                lpproblem += pot[(i, t)] == lp.lpSum(
                    z_pq[(i, t, bp)] * PotBP[i][bp - 1] for bp in BreakPointsPQ[i]
                )
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Potencia - Caudal turbinado
        """
        for i in I:
            for t in T:
                lpproblem += qtb[(i, t)] == lp.lpSum(
                    z_pq[(i, t, bp)] * QtBP[i][bp - 1] for bp in BreakPointsPQ[i]
                )
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Potencia - Caudal turbinado
        """
        for i in I:
            for t in T:
                lpproblem += w_pq[(i, t, 0)] == 0
                lpproblem += w_pq[(i, t, BreakPointsPQ[i][-1])] == 0
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Potencia - Caudal turbinado
        """
        for i in I:
            for t in T:
                for bp in BreakPointsPQ[i]:
                    lpproblem += z_pq[(i, t, bp)] <= w_pq[(i, t, bp - 1)] + w_pq[(i, t, bp)]
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Potencia - Caudal turbinado
        """
        for i in I:
            for t in T:
                lpproblem += lp.lpSum(z_pq[(i, t, bp)] for bp in BreakPointsPQ[i]) == 1
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Potencia - Caudal turbinado
        """
        for i in I:
            for t in T:
                lpproblem += lp.lpSum(w_pq[(i, t, bp)] for bp in BreakPointsPQ[i]) == 1
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Volumen - Caudal máximo
        """
        for i in I:
            for t in T:
                if QmaxBP[i] != None:
                    lpproblem += q_max_vol[(i, t)] == lp.lpSum(
                        z_vq[(i, t, bp)] * QmaxBP[i][bp - 1] for bp in BreakPointsVQ[i]
                    )
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Volumen - Caudal máximo
        """
        for i in I:
            for t in T:
                if QmaxBP[i] != None:
                    if t == T[0]:
                        lpproblem += V0[i] == lp.lpSum(
                            z_vq[(i, t, bp)] * VolBP[i][bp - 1] for bp in BreakPointsVQ[i]
                        )
                    else:
                        lpproblem += vol[(i, t - 1)] == lp.lpSum(
                            z_vq[(i, t, bp)] * VolBP[i][bp - 1] for bp in BreakPointsVQ[i]
                        )
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Volumen - Caudal máximo
        """
        for i in I:
            for t in T:
                if QmaxBP[i] != None:
                    lpproblem += w_vq[(i, t, 0)] == 0
                    lpproblem += w_vq[(i, t, BreakPointsVQ[i][-1])] == 0
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Volumen - Caudal máximo
        """
        for i in I:
            for t in T:
                if QmaxBP[i] != None:
                    for bp in BreakPointsVQ[i]:
                        lpproblem += z_vq[(i, t, bp)] <= lp.lpSum(
                            w_vq[(i, t, tr)] for tr in range(bp - 1, bp + 1)
                        )
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Volumen - Caudal máximo
        """
        for i in I:
            for t in T:
                if QmaxBP[i] != None:
                    lpproblem += lp.lpSum(z_vq[(i, t, bp)] for bp in BreakPointsVQ[i]) == 1
        """
        Restricción asociada al IP with Piecewise Linear Functions de Winston
        en referencia a la curva Volumen - Caudal máximo
        """
        for i in I:
            for t in T:
                if QmaxBP[i] != None:
                    lpproblem += lp.lpSum(w_vq[(i, t, bp)] for bp in BreakPointsVQ[i]) == 1
        """
        Restricción cálculo variación caudal en cada franja
        """
        for i in I:
            for t in T:
                if t == T[0]:
                    lpproblem += qch[(i, t)] == qs[(i, t)] - IniLags[i][0]
                else:
                    lpproblem += qch[(i, t)] == qs[(i, t)] - qs[(i, t - 1)]

        #CON TMIN
        """
        Restricción para contabilizar variación caudal en cada franja
        """
        #for i in I:
        #    for t in T:
        #        lpproblem += qch[(i, t)] <= y[(i, t)] * QMax[i]
        """
        Restricción para contabilizar variación caudal en cada franja
        """
        #for i in I:
        #    for t in T:
        #        lpproblem += -qch[(i, t)] <= y[(i, t)] * QMax[i]
        """
        Restricción acotar cambios de caudal en tmin franjas
        """
        #for i in I:
        #    for t in T:
        #        lpproblem += (
        #                lp.lpSum(y[(i, t + t1)] for t1 in range(0, TMin) if t + t1 <= len(T) - 1)
        #                <= 1
        #        )

        #CON K FORMA CARLOS
        """
        Restricción para contabilizar variación positiva de caudal en cada franja
        """
        for i in I:
            for t in T:
                lpproblem += qch[(i, t)] <= x_pos[(i, t)] * QMax[i]
        """
        Restricción para contabilizar variación negativa de caudal en cada franja
        """
        for i in I:
            for t in T:
                lpproblem += -qch[(i, t)] <= x_neg[(i, t)] * QMax[i]
        """
        Restricción para que solo se cuenta la variación correcta
        """
        for i in I:
            for t in T:
                lpproblem += x_pos[(i, t)] + x_neg[(i, t)] <= 1

        #CON K FORMA ÁLVARO
        """
        Resticción para contabilizar variaciones nulas de caudal en cada franja
        """
        #for i in I:
        #    for t in T:
        #        if t == T[0]:
        #            lpproblem += x_0[(i, t)] >= (qs[(i, t)] - IniLags[i][0]) * (1 / QMax[i]) + 1 - 2 * x_pos[
        #                (i, t)] - 2 * x_neg[(i, t)]
        #            lpproblem += x_0[(i, t)] <= (IniLags[i][0] - qs[(i, t)]) * (1 / QMax[i]) + 1 + 2 * x_pos[
        #                (i, t)] + 2 * x_neg[(i, t)]
        #        else:
        #            lpproblem += x_0[(i, t)] >= (qs[(i, t)] - qs[(i, t-1)]) * (1 / QMax[i]) + 1 - 2 * x_pos[
        #                (i, t)] - 2 * x_neg[(i, t)]
        #            lpproblem += x_0[(i, t)] <= (qs[(i, t-1)] - qs[(i, t)]) * (1 / QMax[i]) + 1 + 2 * x_pos[
        #                (i, t)] + 2 * x_neg[(i, t)]
        """
        Restricción para que solo se cuenta la variación correcta
        """
        #for i in I:
        #    for t in T:
        #        lpproblem += x_pos[(i, t)] + x_neg[(i, t)] + x_0[(i, t)] == 1

        """
        Restricción golpe de ariete
        """
        for i in I:
            for t in T:
                for k in range(1, K+1):
                    if t - k >= 0:
                        lpproblem += x_pos[(i, t)] + x_neg[(i, t - k)] <= 1
                        lpproblem += x_neg[(i, t)] + x_pos[(i, t - k)] <= 1

        """
        Restricción volumen máximo
        """
        for i in I:
            for t in T:
                lpproblem += vol[(i, t)] <= VMax[i]
        """
        Restricción volumen mínimo
        """
        for i in I:
            for t in T:
                lpproblem += vol[(i, t)] >= VMin[i]
        """
        Restricción caudal máximo por canal
        (restricción por sección)
        """
        for i in I:
            for t in T:
                lpproblem += qs[(i, t)] <= QMax[i]
        """
        Restricción caudal máximo por canal
        (restricción por volumen)
        """
        for i in I:
            for t in T:
                if QmaxBP[i] != None:
                    lpproblem += qs[(i, t)] <= q_max_vol[(i, t)]
        """
        Restricción ganancia (€) total de cada embalse
        """
        for i in I:
            lpproblem += pot_embalse[i] == lp.lpSum(
                pot[(i, t)] * Price[t] * (D / 3600) for t in T
            )
        """
        Restricción cómputo desviación respecto a volumen objetivo
        """
        for i in I:
            for t in T:
                if t == T[D_1 - 1]:
                    lpproblem += vol[(i, t)] == VolFinal[i] + pos_desv[i] - neg_desv[i]
        """
        Restricción cálculo beneficio por exceso/falta de volumen
        respecto al objetivo
        """
        for i in I:
            lpproblem += ben_desv[i] == pos_desv[i] * BonusVol - neg_desv[i] * PenVol
        """
        Restricción número total de franjas en zona límite para cada embalse
        """
        for i in I:
            lpproblem += zl_tot[i] == lp.lpSum(
                w_pq[(i, t, bp)] for t in T for bp in ZonaLimitePQ[i]
            )

        # Función que, dado el conjunto FranjasGrupos y un powergroup específico, te devuelve
        # las franjas de la curva Potencia - Caudal turbinado que están en powergroups
        # superiores al dado.
        def obtener_franjas_pw_mayores(diccionario, clave):
            pw_posteriores = list(diccionario.keys())[
                             list(diccionario.keys()).index(clave) + 1:
                             ]

            franjas_posteriores = []
            for pw_posterior in pw_posteriores:
                franjas_posteriores += diccionario[pw_posterior]
            return franjas_posteriores

        """
        Restricción cómputo arranques de powergroups
        """
        for i in I:
            for t in T:
                if t != T[0]:
                    for pg in FranjasGrupos[i]:
                        if pg != list(FranjasGrupos[i].keys())[-1]:
                            lista_keys = list(FranjasGrupos[i].keys())
                            franjassuperiores = obtener_franjas_pw_mayores(FranjasGrupos[i], pg)
                            lpproblem += (
                                    lp.lpSum(
                                        w_pq[(i, t - 1, franja)] for franja in FranjasGrupos[i][pg]
                                    )
                                    + lp.lpSum(
                                        w_pq[(i, t, franja_sup)] for franja_sup in franjassuperiores
                                    )
                                    - 1
                                    <= pwch[(i, t, lista_keys[lista_keys.index(pg) + 1])]
                            )
                            lpproblem += (
                                    lp.lpSum(
                                        w_pq[(i, t - 1, franja)] for franja in FranjasGrupos[i][pg]
                                    )
                                    + lp.lpSum(
                                        w_pq[(i, t, franja_sup)] for franja_sup in franjassuperiores
                                    )
                                    >= 2 * pwch[(i, t, lista_keys[lista_keys.index(pg) + 1])]
                            )
                        if t == 0:
                            lpproblem += pwch[(i, t, pg)] == 0
        """
        Restricción cálculo número de arranques totales en cada embalse
        """
        for i in I:
            lpproblem += pwch_tot[i] == lp.lpSum(
                pwch[(i, t, pg)] for t in T for pg in FranjasGrupos[i]
            )
            
        # Objective Function
        lpproblem += lp.lpSum(
            pot_embalse[i] + ben_desv[i] - zl_tot[i] * PenZL - pwch_tot[i] * PenSU for i in I
        )

        # Solver
        solver_output = tempfile.NamedTemporaryFile(delete=False)
        solver = lp.GUROBI_CMD(
            gapRel=self.config.MIPGap,
            timeLimit=self.config.max_time,
            logPath=solver_output.name
        )  # Other arguments: https://coin-or.github.io/pulp/technical/solvers.html#pulp.apis.GUROBI_CMD
        lpproblem.solve(solver)
        solver_output.close()

        # Parse solver output to get history data
        with open(solver_output.name, 'r') as file:
            output = file.read()
        obj_fun_values, gap_values, time_stamps = self.parse_gurobi_output(output)
        os.remove(solver_output.name)

        def get_dam_id(name: str) -> str:
            if name.find("copy") == -1:
                return name[name.index("dam"): name.index("dam") + 4]
            else:
                return name[name.index("dam"): name.index("copy") + 4]

        # Values stored in solution
        exiting_flows = {dam_id: [var.value() for var in qs.values() if get_dam_id(var.name) == dam_id] for dam_id in I}
        powers = {dam_id: [var.value() for var in pot.values() if get_dam_id(var.name) == dam_id] for dam_id in I}
        volumes = {dam_id: [var.value() for var in vol.values() if get_dam_id(var.name) == dam_id] for dam_id in I}

        # Objective function details
        print([(get_dam_id(var.name), var.value()) for var in pot_embalse.values()])
        income_from_energy = {
            dam_id: [var.value() for var in pot_embalse.values() if get_dam_id(var.name) == dam_id][0] for dam_id in I
        }
        limit_zones = {
            dam_id: [var.value() for var in zl_tot.values() if get_dam_id(var.name) == dam_id][0] for dam_id in I
        }
        startups = {
            dam_id: [var.value() for var in pwch_tot.values() if get_dam_id(var.name) == dam_id][0] for dam_id in I
        }
        vol_exceedance = {
            dam_id: [var.value() for var in pos_desv.values() if get_dam_id(var.name) == dam_id][0] for dam_id in I
        }
        vol_shortage = {
            dam_id: [var.value() for var in neg_desv.values() if get_dam_id(var.name) == dam_id][0] for dam_id in I
        }
        values_from_vol = {
            dam_id: [var.value() for var in ben_desv.values() if get_dam_id(var.name) == dam_id][0] for dam_id in I
        }
        total_dam_incomes = {
            dam_id:
                income_from_energy[dam_id]
                - startups[dam_id] * self.config.startups_penalty
                - limit_zones[dam_id] * self.config.limit_zones_penalty
                + values_from_vol[dam_id]
            for dam_id in I
        }

        # Get datetimes of instance and solution
        start_datetime, end_datetime, _, _, _, solution_datetime = self.get_instance_solution_datetimes()

        # Save solution object
        self.solution = Solution.from_dict(
            dict(
                instance_datetimes=dict(
                    start=start_datetime,
                    end_decisions=end_datetime
                ),
                solution_datetime=solution_datetime,
                solver="MILP",
                configuration=asdict(self.config),
                objective_function=lp.value(lpproblem.objective),
                objective_history=dict(
                    objective_values_eur=obj_fun_values,
                    gap_values_pct=gap_values,
                    time_stamps_s=time_stamps,
                ),
                dams=[
                    dict(
                        id=dam_id,
                        flows=exiting_flows[dam_id],
                        power=powers[dam_id],
                        volume=volumes[dam_id],
                        objective_function_details=dict(
                            total_income_eur=total_dam_incomes[dam_id],
                            income_from_energy_eur=income_from_energy[dam_id],
                            startups=startups[dam_id],
                            limit_zones=limit_zones[dam_id],
                            volume_shortage_m3=vol_shortage[dam_id],
                            volume_exceedance_m3=vol_exceedance[dam_id]
                        )
                    )
                    for dam_id in self.instance.get_ids_of_dams()
                ],
                price=self.instance.get_all_prices(),
            )
        )

        return dict()

    def get_objective(self) -> float:

        return self.solution.get_objective_function()
