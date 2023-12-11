import numpy as np
import matplotlib.pyplot as plt
import SimDataTypes as SDT
import EnvironmentDataType as EDT
import VehicleType as VT
import DataTypes as DT
from DivideToGroups import *
import os
import copy

os.system('cls' if os.name == 'nt' else 'clear')



def main():
    print("Simulation Started!")
    
    MaxDays = 2
    Day = 0
    
    # Initialize the simulation:
    N = 25 # Number of nodes
    N = 8
    CarsInDepots = [0,1] # Car in their depots. Len(array) is numCars, and each entry says which depot we are in
    NumberOfCars = len(CarsInDepots)
    NumberOfDepots = np.unique(CarsInDepots).size
    InitialSOC = 100.0 * np.ones(NumberOfCars) # Everything initialized to full state of charge
    
    # Do the following parameters have any effect? Where are they used?
    MaxNumberOfNodesPerCar = int(1.2*(N-NumberOfDepots)/(np.sum(NumberOfCars))) if np.sum(NumberOfCars) > 1 else N # max number of nodes each car will solve
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
    dT_Plot = 1.0 # what resolution to plot with? 
    dT_max = 0.1
    
    MaxTime = 1000.0 # this is max time for one day
    Layer0UpdateRate = 5.0
    VehiclePlannerUpdateRate = 0.1
    
    InitialModelEstimate = [0.04, 0.06]
    
    SimDataCollection = []
    
    # Initialize the environment:
    Nmc = 1
    np.random.seed(Nmc)
    
    # Generate Physical Map, including location of chargers in a way that ends up being nice. Can see if we want to keep this. 
    PhysicalMap = SDT.GenerateMaps(N, CarsInDepots)
    
    # -------------------------------------------------------------------------------------------------------------------
    
    # PREPARING THE FIRST MAP FOR THE FIRST DAY
    # These lines are only here so that we don't have to change the data type for estimated map right now. they are not used for planning. 
    MaxBiasTimeMapInit, MaxBiasCovTimeMapInit = 0.0, 0.0
    EstimatedTimeMap = PhysicalMap.MapTime + (2.0*np.random.rand(N,N)-1.0)*MaxBiasTimeMapInit # Exactly the same so far
    EstimatedTimeMapCov = PhysicalMap.MapTimeCov + (2.0*np.random.rand(N,N)-1.0)*MaxBiasCovTimeMapInit
    MaxBiasEnergyMapInit, MaxBiasCovEnergyMapInit = 0.0, 0.0
    EstimatedEnergyMap = PhysicalMap.MapEnergy + (2.0*np.random.rand(N,N)-1.0)*MaxBiasEnergyMapInit
    EstimatedEnergyMapCov = PhysicalMap.MapEnergyCov + (2.0*np.random.rand(N,N)-1.0)*MaxBiasCovEnergyMapInit
    EstimatedMap = DT.MapType(EstimatedTimeMap, EstimatedEnergyMap, EstimatedTimeMapCov, EstimatedEnergyMapCov, CarsInDepots, PhysicalMap.EnergyAlpha, PhysicalMap.TimeAlpha, NodesPosition=PhysicalMap.NodesPosition)
    EstimatedMap.CS = PhysicalMap.CS # We know exactly which ones the charger ndoes are
    
    ############## CREATE INITIAL CLOUDMAP DATASET FOR HIGH-LEVEL PLANNER HERE
    CloudMap = np.zeros((N, N, 1000, 3))
    # Use PhysicalMap.MapTime, PhysicalMap.MapEnergy
    # Initialize with 10 samples? The samples will be deviations from MapTime, etc.
    for i in range(N):
        for j in range(N):
            # create 50 things
            energies = np.random.normal(EstimatedEnergyMap[i,j], 0.1*EstimatedEnergyMapCov[i,j], 10)
            times = np.random.normal(EstimatedTimeMap[i,j], 0.1*EstimatedTimeMapCov[i,j], 10)
            CloudMap[i,j,:10,0] = times
            CloudMap[i,j,:10,1] = energies
    # check signs
    if np.any(CloudMap[:,:,:,0] < 0) or np.any(CloudMap[:,:,:,1] > 0):
        print('problem creating cloud map')
        return
    
    
    
    
    
    while Day < MaxDays:
        
        # Data structure to store information about this day's route
        SimData = SDT.SimDataType(PhysicalMap, EstimatedMap, CarsInDepots, InitialSOC, dT_max)
        
        # initialize the model or transfer previous vehicle information as necessary
        if Day == 0:
            SimData.Layer0.CloudMap = copy.deepcopy(CloudMap)
            for i in range(SimData.NVehicles):
                SimData.Vehicle[i].DM.Model = InitialModelEstimate # initial estimate of the vehicle's battery parameters
                SimData.Vehicle[i].SA.Model = InitialModelEstimate # initial estimate of the vehicle's battery parameters
        if Day > 0:
            # 1. deep copy Layer 0
            SimData.Layer0 = copy.deepcopy(SimDataCollection[Day-1].Layer0)
            # 2. Update the cloud map with old SA's estimates
            for i in range(SimData.NVehicles):
                SimData.Vehicle[i] = copy.deepcopy(SimDataCollection[Day-1].Vehicle[i]) # this should keep all the previous experience right?
                SimData.Vehicle[i].ReturnedToDepot = False # have to set this!!! otherwise it thinks we're already done with the task
                SimData.Vehicle[i].CurrentDayTrajectory = [copy.copy(SimData.Vehicle[i].State)] # reset the current day trajectory
                SimData.Vehicle[i].CurrentDayInputTrajectory = [] # reset the current day trajectory
                SimData.Vehicle[i].Temperature = [copy.copy(SimData.Vehicle[i].Temperature[-1])] # not sure if this is what we want to initialize it to. I guess the current temperature, actually
    
        # Create the first plan based on the initial map estimate
        SimData.Layer0.GlobalPlannerInitialize(EstimatedMap, ClusteringMethod)
        for i in range(len(SimData.Layer0.Plan)):
            if SimData.Layer0.Plan[i].NodesTrajectory[0] != SimData.Layer0.Plan[i].NodesTrajectory[-1]:
                SimData.Layer0.Plan[i].NodesTrajectory = np.hstack((SimData.Layer0.Plan[i].NodesTrajectory[-1], SimData.Layer0.Plan[i].NodesTrajectory[0:-1]))
        InitPlan = SimData.Layer0.Plan
        
        # Send the new plan to the vehicles
        for i in range(len(InitPlan)):
            SimData.Vehicle[i].Plan = InitPlan[i] # give them the plan
            SimData.Vehicle[i].State.Pos = SimData.Vehicle[i].Map.NodesPosition[SimData.Vehicle[i].State.Node,:] # set their position to the depot position 
            SimData.Vehicle[i].DM.Pi1 = InitPlan[i] # set the estimated state to be equal to the true state (position and node)
            SimData.Vehicle[i].DM.EstimatedState = copy.deepcopy(SimData.Vehicle[i].State)
            SimData.Vehicle[i].SA.EstimatedState = copy.deepcopy(SimData.Vehicle[i].State)
            
        
        # Run the simulation for the day
        PlotData = SDT.SavePlotData(SimData)
        while (SimData.Time <= MaxTime):
            print(SimData.Time)
            
            # Update the global layer:
            if SimData.Time % Layer0UpdateRate == 0:
                SimData.Layer0.GlobalPlannerUpdate(EstimatedMap)
            
            # Update the environment: 
            SimData.Environment.EnvParams.Update(SimData.Time)
            
            # Update the vehicle DM:
            dT = SimData.next_dT
            if (SimData.Time % VehiclePlannerUpdateRate < 1e-6) or (SimData.Time % VehiclePlannerUpdateRate > (VehiclePlannerUpdateRate - 1e-6)): # == 0:
                print('Updating DM loop')
                for i in range(SimData.NVehicles): 
                    SimData.Vehicle[i].UpdateDM(SimData.Time, dT) 
            else:
                print('Skipped the re-planning loop')
            
            # Update the vehicle states
            if (SimData.Time % VehiclePlannerUpdateRate < 1e-6) or (SimData.Time % VehiclePlannerUpdateRate > (VehiclePlannerUpdateRate - 1e-6)): # == 0:
                for i in range(SimData.NVehicles): 
                    SimData.Vehicle[i].UpdateVehicle(SimData.Time, dT, SimData.Environment.EnvParams.Temperature)
            else:
                print('Skipped vehicle action loop')
        
            
            # Update the simulation time:
            SimData.Update(SimData.Time, dT)
            SimData.Time += dT
        
            # Update the vehicle SA:
            if (SimData.Time % VehiclePlannerUpdateRate < 1e-6) or (SimData.Time % VehiclePlannerUpdateRate > (VehiclePlannerUpdateRate - 1e-6)): # == 0:
                for i in range(SimData.NVehicles): 
                    SimData.Vehicle[i].UpdateSA(SimData.Time, dT)
            else:
                print('Skipped vehicle action loop')
                
        
            PlotData.Update(SimData)
                        
            # Check if the simulation is over:
            if SimData.NumberOfCompletedTours == SimData.NVehicles:
                for i in range(SimData.NVehicles): # here we see if the plan changes
                    #SimData.Vehicle[i].updateModel()
                    TimeCost, EnergyCost = SimData.Vehicle[i].ConsolidateRouteInfo()
                    SimData.Layer0.AddPointsToCloudMap(TimeCost, EnergyCost)
                break
        
        # Plot the results:
        if iplot > 0:
            PlotData.PlotGraphs(InitPlan, SimData)
            PlotData.PlotUnicycleGraphs(InitPlan, SimData)

        
        # Save the day's data here
        SimDataCollection.append(copy.deepcopy(SimData))
        
        Day += 1
        
    # Plotting!! 
    # Plot one: the routes the vehicles took
    plt.figure()
    for i in range(len(SimDataCollection)):
        Map = SimDataCollection[i].Map
        N = Map.N
        M = Map.NumberOfCars
        if np.max(Map.NodesPosition)>0:
            col_vec = ['m','y','b','r','g','c','k']
            for m in range(M):
                plt.scatter(Map.NodesPosition[SimDataCollection[i].Layer0.Plan[m].NodesTrajectory[0:-1],0], Map.NodesPosition[SimDataCollection[i].Layer0.Plan[m].NodesTrajectory[0:-1],1], s=50, c=col_vec[m%len(col_vec)])
                x = []
                y = []
                for j in range(1,len(SimDataCollection[i].Vehicle[m].CurrentDayTrajectory)):
                    x.append(SimDataCollection[i].Vehicle[m].CurrentDayTrajectory[j].Pos[0])
                    y.append(SimDataCollection[i].Vehicle[m].CurrentDayTrajectory[j].Pos[1])
                plt.plot(x,y,c=col_vec[m%len(col_vec)])
            plt.scatter(Map.NodesPosition[Map.CS.Nodes,0], Map.NodesPosition[Map.CS.Nodes,1], c='c', s=15)
            plt.scatter(Map.NodesPosition[Map.Depots,0], Map.NodesPosition[Map.Depots,1], c='k', s=15)
    plt.grid()
    
    # Plot two: time to complete each route per vehicle
    # potential problem: are we not doing the vehicle stuff correctly anymore? 
    # are we not re-setting the trajectory when we create new vehicles? 
    plt.figure()
    leg_str = list()
    for i in range(len(SimDataCollection)):
        leg_str = list()
        for m in range(SimDataCollection[i].Map.NumberOfCars):
            col_vec = ['m','y','b','r','g','c','k']
            m_time = len(SimDataCollection[i].Vehicle[m].CurrentDayTrajectory)*0.1
            plt.scatter(i, m_time, c=col_vec[m%len(col_vec)])
            leg_str.append('Vehicle ' + str(m))
    plt.legend(leg_str)
    plt.xlabel('Day')
    plt.ylabel('Total Route Duration [s]')
    
    # Plot three: vehicles' SOC as a function of time
    # One plot, two vehicles, "Day" lines.
    plt.figure()
    leg_str = list()
    for i in range(len(SimDataCollection)):
        for m in range(SimDataCollection[i].Map.NumberOfCars):
            col_vec = ['m','y','b','r','g','c','k']
            soc = []
            for j in range(1,len(SimDataCollection[i].Vehicle[m].CurrentDayTrajectory)):
                soc.append(SimDataCollection[i].Vehicle[m].CurrentDayTrajectory[j].SOC)
            plt.plot(soc)
            #plt.plot(soc,c=col_vec[m%len(col_vec)])
            leg_str.append('Vehicle ' + str(m) + ', Day ' + str(i))
    plt.legend(leg_str)
    plt.xlabel('Time')
    plt.ylabel('SOC [s]')
            

if __name__ == "__main__":
    main()
    
