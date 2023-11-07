import numpy as np
import DataTypes as DT
import VehicleType as VT
from DivideToGroups import *
from RecursiveOptimalSolution_ChargingStations import *


class GlobalLayer:
    def __init__(self, CarsInDepots: list, EstimatedMap: DT.MapType, InitialSOC: list):
        self.NodesMap = EstimatedMap
        self.NVeciles = len(CarsInDepots)
        self.EstimatedState = []
        self.Plan = []
        for i in range(self.NVeciles):
            self.EstimatedState.append(VT.VehicleEstState(i, CarsInDepots[i], InitialSOC[i]))
            self.Plan.append(DT.Plan(EstimatedMap.N))
        self.MaxCalcTimeFromUpdate = 20.0


    def GlobalPlannerInitialize(self, EstimatedMap: DT.MapType, ClusteringMethod: str):
        # Divide the nodes to groups (Clustering):
        NodesGroups = DivideNodesToGroups(EstimatedMap, ClusteringMethod)

        for i in range(len(NodesGroups)):
            EstimatedTimeMap = EstimatedMap.MapTime[NodesGroups[i],:][:,NodesGroups[i]]
            EstimatedTimeMapCov = EstimatedMap.MapTimeCov[NodesGroups[i],:][:,NodesGroups[i]]
            EstimatedEnergyMap = EstimatedMap.MapEnergy[NodesGroups[i],:][:,NodesGroups[i]]
            EstimatedEnergyMapCov = EstimatedMap.MapEnergyCov[NodesGroups[i],:][:,NodesGroups[i]]
            CarsInDepots = [EstimatedMap.Depots[i]]
            NodesPosition = EstimatedMap.NodesPosition[NodesGroups[i],:]
            Map_i = DT.MapType(EstimatedTimeMap, EstimatedEnergyMap, EstimatedTimeMapCov, EstimatedEnergyMapCov, CarsInDepots, EstimatedMap.EnergyAlpha, EstimatedMap.TimeAlpha, NodesPosition=NodesPosition)
            CS_nodes = []
            for j in range(len(EstimatedMap.CS.Nodes)):
                if EstimatedMap.CS.Nodes[j] in NodesGroups[i]:
                    CS_nodes.append(np.argwhere(NodesGroups[i] == EstimatedMap.CS.Nodes[j]).tolist()[0][0])
            CS = DT.ChargingStationsType(CS_nodes, EstimatedMap.CS.ChargingRate)
            Map_i.CS = CS

            # Plan Tour for each vehicle:
            BestPlan = SolveParallelRecursive_ChargingStations(Map= Map_i,
                                                                        i_CurrentNode = 0, 
                                                                        TourTime = 0.0,
                                                                        TourTimeUncertainty = 0.0,
                                                                        EnergyLeft = self.EstimatedState[i].SOC,
                                                                        EnergyLeftUncertainty = 0.0,
                                                                        NodesTrajectory = [], 
                                                                        BestPlan = DT.Plan(Map_i.N),
                                                                        MaxCalcTimeFromUpdate = self.MaxCalcTimeFromUpdate)
            # Convert Plan to global nodes:
            BestPlan.NodesTrajectory = NodesGroups[i][BestPlan.NodesTrajectory].reshape((-1,))
            BestPlan.CS.Nodes = NodesGroups[i][BestPlan.CS.Nodes].reshape((-1,))

            # Update the global plan:
            self.Plan[i] = BestPlan
            print("Global Planner: Vehicle {} Plan Cost is {}, with Plan {}!".format(i+1, BestPlan.Cost, BestPlan.NodesTrajectory.T))

            self.Plan[i] = BestPlan

        print("Global Planner Initialization: Finished Planning for all vehicles!")



    def GlobalPlannerUpdate(self, EstimatedMap: DT.MapType):
        pass


