import numpy as np
import EnvironmentDataType as EDT
import VehicleType as VT
import DataTypes as DT
from DivideToGroups import *
from GlobalLayer import *
import scipy as sp
import matplotlib.pyplot as plt




class SimDataType():
    def __init__(self, PhysicalMap: DT.MapType, EstimatedMap: DT.MapType, CarsInDepots: list, InitialSOC: list, dT_max: float = 0.01):
        self.Time = 0
        self.NVehicles = len(CarsInDepots)
        self.Layer0 = GlobalLayer(CarsInDepots, EstimatedMap, InitialSOC)
        self.Environment = EDT.EnvironmentType()
        self.Vehicle = []
        for i in range(self.NVehicles):
            self.Vehicle.append(VT.VehicleType(i, CarsInDepots[i], InitialSOC[i], PhysicalMap))
        self.Map = PhysicalMap
        self.min_TgoForNextNode = 0.1
        self.dT_max = dT_max
        self.next_dT = dT_max
        self.NumberOfCompletedTours = 0

    def Update(self, Time, dT):
        # Update Car States:
        self.next_dT = self.dT_max
        for i in range(self.NVehicles):
            if self.Vehicle[i].State.InTransit == False:
                self.Vehicle[i].Plan = self.Layer0.Plan[i]
            self.next_dT = min(self.next_dT,self.Vehicle[i].Update(Time, dT))
            if self.Vehicle[i].State.InTransit == False and self.Vehicle[i].State.Node == self.Vehicle[i].State.iDepot:
                self.Vehicle[i].ReturnedToDepot = True
                self.NumberOfCompletedTours += 1
                self.TimeReturnToDepot = Time + dT
                print("Vehicle {} Completed Tour at Time {} with Tour {}!".format(i+1, self.TimeReturnToDepot, self.Vehicle[i].State.NodesHistory))





def GenerateMaps(N: int, CarsInDepots: list):
    # Map Size
    Xmin, Xmax = -100, 100
    Ymin, Ymax = -100, 100

    M = len(CarsInDepots)

    # Platform Parameters:
    Vmax = 10 # Platform Max Speed
    MinVelReductionCoef, MaxVelReductionCoef = 0.0, 0.75 # min/max speed reduction factor for node2node travel
    VelEnergyConsumptionCoef = 0.04 # Power consumption due to velocity = VelEnergyConsumptionCoef* Vel^2
    VelConstPowerConsumption = 0.04
    ## Total Power to travel From Node i to Node J = (ConstPowerConsumption + VelEnergyConsumptionCoef* Vel^2)*Time_i2j
    MinPowerConsumptionPerTask, MaxPowerConsumptionPerTask = 2, 10
    MinTimePerTask, MaxTimePerTask = 1, 5
    # PltParams.RechargePowerPerDay = 5
    BatteryCapacity = 100.0
    MinimalSOC = 0.0*BatteryCapacity
    FullRechargeRateFactor = 0.25
    StationRechargePower = 3

    ## Randomize The Nodes Locations:
    NodesPosition = np.block([np.random.uniform(Xmin,Xmax, size=(N,1)), np.random.uniform(Ymin,Ymax, size=(N,1))])
    # NodesPosition[0,0] = 0.0; NodesPosition[0,1] = 0.0

    # Set the Nomianl Time of Travel between any 2 nodes as the distance between
    # the nodes divided by the estimated travel velocity:
    NodesVelocity = np.zeros((N,N), dtype=float)
    NodesDistance = np.zeros((N,N), dtype=float)
    NodesTimeOfTravel = np.zeros((N,N), dtype=float)
    TravelSigma = np.zeros((N,N), dtype=float)
    NodesEnergyTravel = np.zeros((N,N), dtype=float)
    NodesEnergyTravelSigma = np.zeros((N,N), dtype=float)

    for i in range(N):
        NodesVelocity[i,i+1:] = 6 #np.random.uniform(PltParams.Vmax*(1.0-PltParams.MaxVelReductionCoef), PltParams.Vmax*(1.0-PltParams.MinVelReductionCoef), size=(1,N-i-1))
        NodesVelocity[i+1:,i] = NodesVelocity[i,i+1:].T
        for j in range(i,N):
            if i==j: continue
            NodesDistance[i,j] = np.linalg.norm(np.array([NodesPosition[i,0]-NodesPosition[j,0], NodesPosition[i,1]-NodesPosition[j,1]]))
            NodesDistance[j,i] = NodesDistance[i,j]
            NodesTimeOfTravel[i,j] = NodesDistance[i,j] / NodesVelocity[i,j]
            NodesTimeOfTravel[j,i] = NodesTimeOfTravel[i,j]
            TravelSigma[i,j] = np.random.uniform(0.05*NodesTimeOfTravel[i,j], 0.3*NodesTimeOfTravel[i,j],1)
            TravelSigma[j,i] = TravelSigma[i,j]
            NodesEnergyTravel[i,j] = -NodesTimeOfTravel[i,j] * (VelConstPowerConsumption + VelEnergyConsumptionCoef*NodesVelocity[i,j]**2)
            NodesEnergyTravel[j,i] = NodesEnergyTravel[i,j]
            NodesEnergyTravelSigma[i,j] = np.abs(np.random.uniform(0.05*NodesEnergyTravel[i,j], 0.1*NodesEnergyTravel[i,j],1))
            NodesEnergyTravelSigma[j,i] = NodesEnergyTravelSigma[i,j]

    NodesEnergyTravelSigma2 = NodesEnergyTravelSigma**2
    TravelSigma2 = TravelSigma**2

    # Divide the nodes to groups (Clustering):
    SolutionProbabilityEnergyReliability = 0.999
    SolutionProbabilityTimeReliability = 0.9
    EnergyAlpha = sp.stats.norm.ppf(SolutionProbabilityEnergyReliability)
    TimeAlpha = sp.stats.norm.ppf(SolutionProbabilityTimeReliability)

    ClusteringMethod = "Max_Eigenvalue" # "Max_Eigenvalue" or "Frobenius" or "Sum_AbsEigenvalue" or "Mean_MaxRow" or "PartialMax_Eigenvalue"
    Map = DT.MapType(NodesTimeOfTravel, NodesEnergyTravel, TravelSigma2, NodesEnergyTravelSigma2, CarsInDepots, EnergyAlpha, TimeAlpha, NodesPosition=NodesPosition)
    NodesGroups = DivideNodesToGroups(Map, ClusteringMethod, isplot = False)

    ## Charging Stations:
    ChargingStations = list()
    for i in range (M):
        NumGroupChargers = min(1,int(np.ceil(NodesGroups[i].shape[0]/10)))
        GroupCharging = NodesGroups[i][list(np.random.randint(Map.NumberOfDepots,NodesGroups[i].shape[0],size=(NumGroupChargers,1)))].reshape((-1,))
        while len(np.unique(GroupCharging)) < NumGroupChargers:
            GroupCharging = NodesGroups[i][list(np.random.randint(Map.NumberOfDepots,NodesGroups[i].shape[0],size=(NumGroupChargers,1)))].reshape((-1,))
        for j in range(NumGroupChargers):
            ChargingStations.append(GroupCharging[j])
    Map.CS = DT.ChargingStationsType(ChargingStations, StationRechargePower, BatteryCapacity)
    return Map


class SavePlotData():
    def __init__(self, SimData: SimDataType) -> None:
        self.Time = [SimData.Time]
        self.VehicleState = []
        for i in range(SimData.NVehicles):
            self.VehicleState.append([SimData.Vehicle[i].State])

    def Update(self, SimData: SimDataType) -> None:
        self.Time.append(SimData.Time)
        for i in range(SimData.NVehicles):
            self.VehicleState[i].append(SimData.Vehicle[i].State)

    def PlotGraphs(self, InitPlan: DT.Plan, SimData: SimDataType) -> None:
        Map = SimData.Map
        N = Map.N
        M = Map.NumberOfCars

        if np.max(Map.NodesPosition)>0:
            col_vec = ['m','y','b','r','g','c','k']
            leg_str = list()
            plt.figure()
            plt.scatter(Map.NodesPosition[Map.Depots,0], Map.NodesPosition[Map.Depots,1], c='k', s=50)
            leg_str.append('Depot')
            for i in range(M):
                plt.scatter(Map.NodesPosition[InitPlan[i].NodesTrajectory[1:-1],0], Map.NodesPosition[InitPlan[i].NodesTrajectory[1:-1],1], s=50, c=col_vec[i%len(col_vec)])
                leg_str.append('Group '+str(i)+" Number of Nodes: {}".format(len(InitPlan[i].NodesTrajectory)-1))
            plt.scatter(Map.NodesPosition[Map.CS.Nodes,0], Map.NodesPosition[Map.CS.Nodes,1], c='c', s=15)
            leg_str.append('Charging Station')
            plt.legend(leg_str)
            for m in range(M):
                colr = col_vec[m%len(col_vec)]
                for i in range(len(InitPlan[m].NodesTrajectory)-1):
                    j1 = InitPlan[m].NodesTrajectory[i]
                    j2 = InitPlan[m].NodesTrajectory[i+1]
                    plt.arrow(Map.NodesPosition[j1,0],Map.NodesPosition[j1,1],Map.NodesPosition[j2,0]-Map.NodesPosition[j1,0],Map.NodesPosition[j2,1]-Map.NodesPosition[j1,1], width= 1, color=colr)
            plt.xlim((-100,100))
            plt.ylim((-100,100))
            plt.grid()


        # Plot the vehicles:
        TourTime = np.zeros((M,N))
        TourTimeUncertaintyCov = np.zeros((M,N))
        TourEnergy = np.zeros((M,N))
        TourEnergyUncertaintyCov = np.zeros((M,N))
        for m in range(M):
            indx = 1
            TourEnergy[m,0] = 100.0
            for i in range(len(InitPlan[m].NodesTrajectory)-1):
                j1 = InitPlan[m].NodesTrajectory[i]
                j2 = InitPlan[m].NodesTrajectory[i+1]
                TourTime[m,indx] = TourTime[m,indx-1] + Map.MapTime[j1,j2]
                TourTimeUncertaintyCov[m,indx] = TourTimeUncertaintyCov[m,indx-1] + Map.MapTimeCov[j1,j2]
                TourEnergy[m,indx] = TourEnergy[m,indx-1] + Map.MapEnergy[j1,j2]
                TourEnergyUncertaintyCov[m,indx] = TourEnergyUncertaintyCov[m,indx-1] + Map.MapEnergyCov[j1,j2]
                if j2 in Map.CS.Nodes:
                    i_CS = np.argwhere(InitPlan[m].CS.Nodes == j2).tolist()[0][0]
                    TourTime[m,indx] += InitPlan[m].CS.ChargingTime[i_CS]
                    TourEnergy[m,indx] += InitPlan[m].CS.ChargingTime[i_CS]*InitPlan[m].CS.ChargingRate
                indx += 1
            
        plt.figure()
        plt.subplot(2,1,1)
        for m in range(M):
            colr = col_vec[m%len(col_vec)]
            TourLen = len(InitPlan[m].NodesTrajectory)
            plt.plot(TourTime[m,0:TourLen], label='Vehicle '+str(m+1)+' Nominal', color=colr)
            plt.plot(TourTime[m,0:TourLen] + Map.TimeAlpha*np.sqrt(TourTimeUncertaintyCov[m,0:TourLen]), '-.', label='Vehicle '+str(m+1) + ' Uncertainty', color=colr)
            plt.grid()
            plt.legend()
            plt.title('Tour Time')
        plt.subplot(2,1,2)
        for m in range(M):
            colr = col_vec[m%len(col_vec)]
            TourLen = len(InitPlan[m].NodesTrajectory)
            plt.plot(TourEnergy[m,0:TourLen], label='Vehicle '+str(m+1)+' Nominal', color=colr)
            plt.plot(TourEnergy[m,0:TourLen] - Map.EnergyAlpha*np.sqrt(TourEnergyUncertaintyCov[m,0:TourLen]), '-.', label='Vehicle '+str(m+1) + 'Uncertainty', color=colr)
            plt.grid()
            plt.legend()
            plt.title('Tour Energy')
        plt.show()

