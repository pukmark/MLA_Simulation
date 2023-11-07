import numpy as np
import matplotlib.pyplot as plt
import SimDataTypes as SDT
import EnvironmentDataType as EDT
import VehicleType as VT
import DataTypes as DT
from DivideToGroups import *
import os

os.system('cls' if os.name == 'nt' else 'clear')



def main():
    print("Simulation Started!")

    # Initialize the simulation:
    N = 25 # Number of nodes
    CarsInDepots = [0,1] # Car in their depots
    NumberOfCars = len(CarsInDepots)
    NumberOfDepots = np.unique(CarsInDepots).size
    IniitialSOC = 100.0 * np.ones(NumberOfCars)
    MaxNumberOfNodesPerCar = int(1.2*(N-NumberOfDepots)/(np.sum(NumberOfCars))) if np.sum(NumberOfCars) > 1 else N
    MaxNodesToSolver = 13
    SolutionProbabilityTimeReliability = 0.9
    SolutionProbabilityEnergyReliability = 0.999
    MaxMissionTime = 120
    ReturnToBase = True
    MustVisitAllNodes = True
    CostFunctionType = 1 # 1: Min Sum of Time Travelled, 2: ,Min Max Time Travelled by any car
    MaxTotalTimePerVehicle  = 200.0
    RechargeModel = 'ConstantRate' # 'ExponentialRate' or 'ConstantRate'
    SolverType = 'Gurobi_NoClustering' # 'Gurobi' or 'Recursive' or 'Gurobi_NoClustering'
    ClusteringMethod = "Max_Eigenvalue" # "Max_Eigenvalue" or "Frobenius" or "Sum_AbsEigenvalue" or "Mean_MaxRow" or "PartialMax_Eigenvalue"
    MaxCalcTimeFromUpdate = 1000.0 # Max time to calculate the solution from the last update [sec]
    iplot = 1 # 0: No Plot, 1: Plot Main Cluster, 2: Plot All Clusters, 3: Plot Connected Tours
    dT_Plot = 1.0

    # Initialize the environment:
    Nmc = 1
    np.random.seed(Nmc)
    # Generate Physical Map:
    PhysicalMap = SDT.GenerateMaps(N, CarsInDepots)
    # Generate Initial Estimated Map:
    dT_max = 0.1
    MaxBiasTimeMapInit, MaxBiasCovTimeMapInit = 0.0, 0.0
    EstimatedTimeMap = PhysicalMap.MapTime + (2.0*np.random.rand(N,N)-1.0)*MaxBiasTimeMapInit
    EstimatedTimeMapCov = PhysicalMap.MapTimeCov + (2.0*np.random.rand(N,N)-1.0)*MaxBiasCovTimeMapInit
    MaxBiasEnergyMapInit, MaxBiasCovEnergyMapInit = 0.0, 0.0
    EstimatedEnergyMap = PhysicalMap.MapEnergy + (2.0*np.random.rand(N,N)-1.0)*MaxBiasEnergyMapInit
    EstimatedEnergyMapCov = PhysicalMap.MapEnergyCov + (2.0*np.random.rand(N,N)-1.0)*MaxBiasCovEnergyMapInit
    EstimatedMap = DT.MapType(EstimatedTimeMap, EstimatedEnergyMap, EstimatedTimeMapCov, EstimatedEnergyMapCov, CarsInDepots, PhysicalMap.EnergyAlpha, PhysicalMap.TimeAlpha, NodesPosition=PhysicalMap.NodesPosition)
    EstimatedMap.CS = PhysicalMap.CS
    SimData = SDT.SimDataType(PhysicalMap, EstimatedMap, CarsInDepots, IniitialSOC, dT_max)
    
    # Run the simulation:
    MaxTime = 1000.0
    Layer0UpdateRate = 5.0
    VehiclePlannerUpdateRate = 0.5
    SimData.Layer0.GlobalPlannerInitialize(EstimatedMap, ClusteringMethod)
    InitPlan = SimData.Layer0.Plan

    TimePlot = 0.0
    PlotData = SDT.SavePlotData(SimData)
    while (SimData.Time <= MaxTime):
        
        # Update the global layer:
        if SimData.Time % Layer0UpdateRate == 0:
            SimData.Layer0.GlobalPlannerUpdate(EstimatedMap)
        
        # Update the environment:
        SimData.Environment.EnvParams.Update(SimData.Time)
        
        # Update the vehicles:
        dT = SimData.next_dT
        if SimData.Time % VehiclePlannerUpdateRate == 0:
            for i in range(SimData.NVehicles):
                SimData.Vehicle[i].UpdateSelfPlanner(SimData.Time)
        SimData.Update(SimData.Time, dT)

        # Update the simulation time:
        SimData.Time += dT

        # Save the data for plotting:
        if SimData.Time >= TimePlot + dT_Plot:
            PlotData.Update(SimData)
            TimePlot += dT_Plot

        # Check if the simulation is over:
        if SimData.NumberOfCompletedTours == SimData.NVehicles:
            break

    # Plot the results:
    if iplot > 0:
        PlotData.PlotGraphs(InitPlan, SimData)

if __name__ == "__main__":
    main()
