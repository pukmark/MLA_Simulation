import numpy as np
import DataTypes as DT
import VehicleType as VT
from DivideToGroups import *
from RecursiveOptimalSolution_ChargingStations import *
import copy


class GlobalLayer:
    def __init__(self, CarsInDepots: list, EstimatedMap: DT.MapType, InitialSOC: list):
        self.NodesMap = EstimatedMap
        self.NVeciles = len(CarsInDepots)
        self.EstimatedState = []
        self.Plan = []
        for i in range(self.NVeciles):
            self.EstimatedState.append(copy.copy(VT.VehicleEstState(i, CarsInDepots[i], InitialSOC[i])))
            self.Plan.append(copy.copy(DT.Plan(EstimatedMap.N)))
        self.MaxCalcTimeFromUpdate = 20.0 
        
        self.MaxCloudMapPts = 1000
        self.CloudMap = np.zeros((EstimatedMap.N, EstimatedMap.N, self.MaxCloudMapPts, 3))
        # 3 entries: time required, soc required, temperature
    
    def AddPointsToCloudMap(self, TimeCost, EnergyCost):
        # the matrices are NxN
        for i in range(np.shape(TimeCost)[0]):
            for j in range(i):
                if TimeCost[i,j] != 0: # new data point for route (i,j)
                    ind = np.squeeze(np.argwhere(self.CloudMap[i,j][:,0] == 0)[0]) # how many points do we already have
                    
                    # could implement a memory function here! 
                    self.CloudMap[i,j][ind,0] = TimeCost[i,j]
                    self.CloudMap[i,j][ind,1] = EnergyCost[i,j]
                    
                    self.CloudMap[j,i][ind,0] = TimeCost[i,j]
                    self.CloudMap[j,i][ind,1] = EnergyCost[i,j]
        

    
    def EstimateMapsFromCloudMap(self):
        # given cloud map, we want to create two estimated maps: time and energy requirements
        # for now, do not worry about the temperature thing
        N = np.shape(self.CloudMap)[0]
        
        EstimatedTimeMap = np.zeros((N,N))
        EstimatedTimeMapVar = np.zeros((N,N))
        EstimatedEnergyMap = np.zeros((N,N))
        EstimatedEnergyMapVar = np.zeros((N,N))
        
        for i in range(N):
            for j in range(N):
                # TO DO: figure out the index!!! don't want to count any "zeros"
                ind = np.squeeze(np.argwhere(self.CloudMap[i,j][:,0] == 0)[0])
                
                # mean time
                EstimatedTimeMap[i,j] = np.mean(self.CloudMap[i,j][:ind,0])
                EstimatedTimeMap[j,i] = EstimatedTimeMap[i,j]
                
                # variance time
                EstimatedTimeMapVar[i,j] = np.var(self.CloudMap[i,j][:ind,0])
                EstimatedTimeMapVar[j,i] = EstimatedTimeMapVar[i,j]
                
                # mean energy
                EstimatedEnergyMap[i,j] = np.mean(self.CloudMap[i,j][:ind,1])
                EstimatedEnergyMap[j,i] = EstimatedEnergyMap[i,j]
                
                # variance energy
                EstimatedEnergyMapVar[i,j] = np.var(self.CloudMap[i,j][:ind,1])
                EstimatedEnergyMapVar[j,i] = EstimatedEnergyMapVar[i,j]
        
        return EstimatedTimeMap, EstimatedTimeMapVar, EstimatedEnergyMap, EstimatedEnergyMapVar
                
            

    def GlobalPlannerInitialize(self, EstimatedMap: DT.MapType, ClusteringMethod: str):        
        
        # Divide the nodes to groups (Clustering), depending on estimated map:
        NodesGroups = DivideNodesToGroups(EstimatedMap, ClusteringMethod)

        for i in range(len(NodesGroups)):
            
            EstimatedTimeMap1, EstimatedTimeMapCov1, EstimatedEnergyMap1, EstimatedEnergyMapCov1 = self.EstimateMapsFromCloudMap()
            
            EstimatedTimeMap = EstimatedTimeMap1[NodesGroups[i],:][:,NodesGroups[i]]
            EstimatedTimeMapCov = EstimatedTimeMapCov1[NodesGroups[i],:][:,NodesGroups[i]]
            EstimatedEnergyMap = EstimatedEnergyMap1[NodesGroups[i],:][:,NodesGroups[i]]
            EstimatedEnergyMapCov = EstimatedEnergyMapCov1[NodesGroups[i],:][:,NodesGroups[i]]
            
            ########## select the appropriate nodes
            # EstimatedTimeMap = EstimatedMap.MapTime[NodesGroups[i],:][:,NodesGroups[i]]
            # EstimatedTimeMapCov = EstimatedMap.MapTimeCov[NodesGroups[i],:][:,NodesGroups[i]]

            # EstimatedEnergyMap = EstimatedMap.MapEnergy[NodesGroups[i],:][:,NodesGroups[i]]
            # EstimatedEnergyMapCov = EstimatedMap.MapEnergyCov[NodesGroups[i],:][:,NodesGroups[i]]

            CarsInDepots = [EstimatedMap.Depots[i]] # length of CarsInDepots is how many cars we have, and each entry corresponds to which depot
            NodesPosition = EstimatedMap.NodesPosition[NodesGroups[i],:]
            
            # sets the charging stations by reading EstimatedMap
            CS_nodes = []
            for j in range(len(EstimatedMap.CS.Nodes)):
                if EstimatedMap.CS.Nodes[j] in NodesGroups[i]:
                    CS_nodes.append(np.argwhere(NodesGroups[i] == EstimatedMap.CS.Nodes[j]).tolist()[0][0])
            CS = DT.ChargingStationsType(CS_nodes, EstimatedMap.CS.ChargingRate)
     
            Map_i = DT.MapType(EstimatedTimeMap, EstimatedEnergyMap, EstimatedTimeMapCov, EstimatedEnergyMapCov, CarsInDepots, EstimatedMap.EnergyAlpha, EstimatedMap.TimeAlpha, NodesPosition=NodesPosition)
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
            
            TourTime = np.zeros((len(BestPlan.NodesTrajectory)))
            TourTimeUncertaintyCov = np.zeros((len(BestPlan.NodesTrajectory)))
            TourEnergy = np.zeros((len(BestPlan.NodesTrajectory)))
            TourEnergyUncertaintyCov = np.zeros((len(BestPlan.NodesTrajectory)))
            TourEnergy[0] = self.EstimatedState[i].SOC # should be 100.0
            for j in range(len(BestPlan.NodesTrajectory)-1):
                j1 = BestPlan.NodesTrajectory[j]
                j2 = BestPlan.NodesTrajectory[j+1]
                TourTime[j+1] += TourTime[j] + EstimatedTimeMap1[j1,j2]
                TourTimeUncertaintyCov[j+1] = TourTimeUncertaintyCov[j] + EstimatedTimeMapCov1[j1,j2]
                TourEnergy[j+1] += TourEnergy[j] + EstimatedEnergyMap1[j1,j2]
                TourEnergyUncertaintyCov[j+1] = TourEnergyUncertaintyCov[j] + EstimatedEnergyMapCov1[j1,j2]
                
                if j2 in EstimatedMap.CS.Nodes: # is the node we are going to in the list of chargers? 
                    i_CS = np.argwhere(BestPlan.CS.Nodes == j2).tolist()[0][0]
                    TourTime[j+2] += BestPlan.CS.ChargingTime[i_CS]
                    TourEnergy[j+2] += BestPlan.CS.ChargingTime[i_CS]*BestPlan.CS.ChargingRate
            
            
            RobustTourTime = TourTime + EstimatedMap.TimeAlpha*np.sqrt(TourTimeUncertaintyCov)
            RobustTourEnergy = TourEnergy - EstimatedMap.EnergyAlpha*np.sqrt(TourEnergyUncertaintyCov)
            
            BestPlan.TimePlan = TourTime
            BestPlan.SOCPlan = TourEnergy
            
            # Update the global plan:
            self.Plan[i] = BestPlan
            print("Global Planner: Vehicle {} Plan Cost is {}, with Plan {}!".format(i+1, BestPlan.Cost, BestPlan.NodesTrajectory.T))

            self.Plan[i] = BestPlan # each plan contains the sequence of nodes for a particular group

        print("Global Planner Initialization: Finished Planning for all vehicles!")



    def GlobalPlannerUpdate(self, EstimatedMap: DT.MapType):
        
        
        
        pass


