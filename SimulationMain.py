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

################## Overview of what is running ########
# 1. Global layer is initialized with a mean/variance estimate of Time and SOC costs for each node segment. The mean/variance are estimated from a CloudMap of (T, SOC) cost points.
# 2. Global layer calculates a plan for each vehicle. A plan consists of a sequence of nodes the vehicle must visit, along with expectations for a) at what time the vehicle should
#       reach the node and b) how much charge the vehicle should have left when it reaches the node. The plan also contains information about when the vehicle must charge.  
# 3. Given instructions for a) what node to travel to next, b) how quickly to get there, and c) the minimum charge left at arrival, the vehicle's Decision Manager (DM) calculates a reference 
#       trajectory to the goal node that satisfies the time and SOC requirements. This is calculated via batch formulation, and is calculated ONCE whenever the vehicle reaches
#       a new node. The reference trajectory is stored. 
# 4. The vehicle (unicycle model) updates its control input at a frequency of 10 Hz. The control input at any time point is calculated using LMPC, with the DM's batch reference trajectory as the 
#       the safe set. The cost function is a minimum time formulation. The LMPC is allowed to outperform the SOC dynamics given by the DM's batch reference trajectory. This results 
#       in a low-level controller that satisfies (and may improve upon) the time and soc constraints set forth by the Global layer.
# 5. Once the vehicle has traveled 70% of the distance between two nodes, a forward reachability analysis is undertaken by the DM Module. The Global Layer dictates a (T, SOC) in which the 
#       vehicle must reach the next node. The reachability analysis calculates an estimate of what other (T, SOC) costs the vehicle could have incurred on the current node segment. This is 
#       calculated using forward reachable sets and seeing in how many time steps / with how much remaining SOC we could have reached the goal node from the current position. The results
#       of the reachability analysis are stored in the vector ReachSetInfo, which lists [start_node, end_node, [array of possible (time_cost, soc_cost)]]. 
#       The reachability results are not currently used, but are available for the Global Layer to incorporate in the next day's planning. Note we can change the 70% to be whatever value we want, 
#       based on how much we want to explore. 
#######################################################

################## Current Assumptions ################
# 1. Vehicle model is known exactly   
# 2. No variability in the SOC / Time costs for different road segments -> exact reproducibility
# 3. No exogenous disturbances (e.g. temperature variations, etc)
# These assumptions will be relaxed in upcoming iterations of the work
#######################################################

################## TO DO ##############################
# 1. Update Global Layer calculation to allow for multiple vehicles / incomplete jobs 
# 2. Update Global Layer calculations to use the exploration data
#######################################################

def main():
    print("Simulation Started!")
    
    MaxDays = 2
    Day = 0
    
    # Initialize the simulation:
    N = 15 # Number of nodes
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
    
    InitialModelEstimate = [0.04, 0.06] # not used at the moment
    
    SimDataCollection = []
    
    # Initialize the environment:
    Nmc = 1
    np.random.seed(Nmc)
    
    # Generate Physical Map, including location of chargers in a way that ends up being nice. Can see if we want to keep this. 
    PhysicalMap = SDT.GenerateMaps(N, CarsInDepots)
    
    # -------------------------------------------------------------------------------------------------------------------
    
    MaxBiasTimeMapInit, MaxBiasCovTimeMapInit = 0.0, 0.0
    EstimatedTimeMap = PhysicalMap.MapTime + (2.0*np.random.rand(N,N)-1.0)*MaxBiasTimeMapInit # Exactly the same so far
    EstimatedTimeMapCov = PhysicalMap.MapTimeCov + (2.0*np.random.rand(N,N)-1.0)*MaxBiasCovTimeMapInit
    MaxBiasEnergyMapInit, MaxBiasCovEnergyMapInit = 0.0, 0.0
    EstimatedEnergyMap = PhysicalMap.MapEnergy + (2.0*np.random.rand(N,N)-1.0)*MaxBiasEnergyMapInit
    EstimatedEnergyMapCov = PhysicalMap.MapEnergyCov + (2.0*np.random.rand(N,N)-1.0)*MaxBiasCovEnergyMapInit
    EstimatedMap = DT.MapType(EstimatedTimeMap, EstimatedEnergyMap, EstimatedTimeMapCov, EstimatedEnergyMapCov, CarsInDepots, PhysicalMap.EnergyAlpha, PhysicalMap.TimeAlpha, NodesPosition=PhysicalMap.NodesPosition)
    EstimatedMap.CS = PhysicalMap.CS # We know exactly which ones the charger ndoes are
    
    ############## Initial Cloudmap for High-Level Planner
    CloudMap = np.zeros((N, N, 1000, 3))
    for i in range(N):
        for j in range(N):
            energies = np.random.normal(EstimatedEnergyMap[i,j], 0.1*EstimatedEnergyMapCov[i,j], 10)
            times = np.random.normal(EstimatedTimeMap[i,j], 0.1*EstimatedTimeMapCov[i,j], 10)
            CloudMap[i,j,:10,0] = times
            CloudMap[i,j,:10,1] = energies
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
            SimData.Layer0 = copy.deepcopy(SimDataCollection[Day-1].Layer0)
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
                for i in range(SimData.NVehicles): 
                    #SimData.Vehicle[i].updateModel()
                    TimeCost, EnergyCost = SimData.Vehicle[i].ConsolidateRouteInfo()
                    SimData.Layer0.AddPointsToCloudMap(TimeCost, EnergyCost)
                break
        
        # Plot the results:
        if iplot > 0:
            PlotData.PlotGraphs(InitPlan, SimData)
            PlotData.PlotUnicycleGraphs(InitPlan, SimData)
        
        # Save the day's data 
        SimDataCollection.append(copy.deepcopy(SimData))
        
        Day += 1
        
    # Plots
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
    
