import numpy as np
import SimDataTypes as DataTypes
import matplotlib.pyplot as plt
import DataTypes as DT


def CreateGroupMatrix(Group, NodesTimeOfTravel):
    GroupMatrix = np.zeros((len(Group),len(Group)))
    for i in range(len(Group)):
        for j in range(len(Group)):
            GroupMatrix[i,j] = NodesTimeOfTravel[Group[i],Group[j]]
    return GroupMatrix

def CalcEntropy(NodesGroups, NodesTimeOfTravel, Method):
    NumberOfCars = len(NodesGroups)
    TimeEntropy_i = np.zeros(NumberOfCars)
    if Method == "Max_Eigenvalue":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            w, _ = np.linalg.eig(GroupTimeMatrix)
            TimeEntropy_i[iGroup] = np.max(np.abs(w))**2 * len(w)
    elif Method == "Frobenius":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            TimeEntropy_i[iGroup] = np.linalg.norm(GroupTimeMatrix, 'fro')
    elif Method == "Mean_MaxRow":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            TimeEntropy_i[iGroup] = np.sum(np.max(GroupTimeMatrix, axis=1))**2
    elif Method == "Sum_AbsEigenvalue":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            w, v = np.linalg.eig(GroupTimeMatrix)
            TimeEntropy_i[iGroup] = np.sum(np.abs(w)**2)
    elif Method == "PartialMax_Eigenvalue":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            w, v = np.linalg.eig(GroupTimeMatrix)
            w = np.sort(np.abs(w))[::-1]
            ind = 3
            weights = np.array(range(1,ind+1))[::-1]
            TimeEntropy_i[iGroup] = len(w) * np.sum((weights*np.abs(w[0:ind]))**2)
    else:
        print("Error: Unknown method for calculating Entropy")
    
    TimeEntropy = np.sum(TimeEntropy_i)
    return TimeEntropy

def DivideNodesToGroups(Map: DT.MapType , 
                        Method = "Max_Eigenvalue", 
                        MaximizeGroupSize: bool=False,
                        MustIncludeNodeZero: bool = True, 
                        ChargingStations: list = [],
                        MaxGroupSize: int = 12,
                        isplot: bool = False):

    N = Map.N
    M = Map.NumberOfCars
    NodesGroups = list()
    MaxGroupSize = max(np.ceil(N/M), MaxGroupSize)

    i1 = 0 if MustIncludeNodeZero==False else 1
    NodesTimeOfTravel = Map.MapTime + Map.TimeAlpha * np.sqrt(Map.MapTimeCov)
    for i in range(len(Map.Depots)):
        NodesGroups.append(np.array([Map.Depots[i]]))

    # Initialize the groups by Closest Neighbor:
    for iNode in range(Map.NumberOfDepots,Map.N):
        TimeToNode = NodesTimeOfTravel[Map.Depots,iNode]
        Group_MinTime = np.argsort(TimeToNode)
        for iGroup in Group_MinTime:
            if NodesGroups[iGroup].shape[0] < MaxGroupSize:
                NodesGroups[iGroup] = np.append(NodesGroups[iGroup], iNode)
                break

    # Minimize "Entropy"
    print("Initial Entropy: ", CalcEntropy(NodesGroups, NodesTimeOfTravel, Method))
    for iter in range(100):
        CurNodesGroups = NodesGroups.copy()
        GroupChanged = False
        for iGroup in range(M):
            Entropy = np.zeros((M,))
            for i in range(i1,len(CurNodesGroups[iGroup])):
                CurNode = CurNodesGroups[iGroup][i]
                for jGroup in range(M):
                    arg_iGroup = np.argwhere(NodesGroups[iGroup]==CurNode)
                    NodesGroups[iGroup] = np.delete(NodesGroups[iGroup], arg_iGroup)
                    NodesGroups[jGroup] = np.append(NodesGroups[jGroup], CurNode)
                    if len(NodesGroups[jGroup]) > MaxGroupSize:# and len(CurNodesGroups[jGroup]) > len(NodesGroups[jGroup]):
                        Entropy[jGroup] = 1e10
                    else:
                        Entropy[jGroup] = CalcEntropy(NodesGroups, NodesTimeOfTravel, Method)
                    arg_jGroup = np.argwhere(NodesGroups[jGroup]==CurNode)
                    NodesGroups[jGroup] = np.delete(NodesGroups[jGroup], arg_jGroup)
                    NodesGroups[iGroup] = np.append(NodesGroups[iGroup], CurNode)
                    
                Group_MinEntropy = np.argmin(Entropy)
                arg_iGroup = np.argwhere(NodesGroups[iGroup]==CurNode)
                NodesGroups[iGroup] = np.delete(NodesGroups[iGroup], arg_iGroup)
                NodesGroups[Group_MinEntropy] = np.append(NodesGroups[Group_MinEntropy], CurNode)
                if Group_MinEntropy != iGroup:
                    GroupChanged = True
            NodesGroups[iGroup] = np.sort(NodesGroups[iGroup])
        print("iteration = "+str(iter)+", Entropy = {:}".format(Entropy[Group_MinEntropy]))
        if not GroupChanged:
            break
        
    # Try Switch Nodes between groups:
    for iter in range(100):
        GroupChanged = False
        CurNodesGroups = NodesGroups.copy()
        Entropy = CalcEntropy(NodesGroups, NodesTimeOfTravel, Method)
        for iGroup in range(M):
            for jGroup in range(iGroup+1,M):
                if iGroup==jGroup: continue
                for iNode in NodesGroups[iGroup][i1:]:
                    for jNode in NodesGroups[jGroup][i1:]:
                        if iNode==0 or jNode==0: continue
                        arg_iNode = np.argwhere(CurNodesGroups[iGroup]==iNode)
                        arg_jNode = np.argwhere(CurNodesGroups[jGroup]==jNode)
                        CurNodesGroups[iGroup] = np.delete(CurNodesGroups[iGroup], arg_iNode)
                        CurNodesGroups[jGroup] = np.delete(CurNodesGroups[jGroup], arg_jNode)
                        CurNodesGroups[iGroup] = np.append(CurNodesGroups[iGroup], jNode)
                        CurNodesGroups[iGroup] = np.sort(CurNodesGroups[iGroup])
                        CurNodesGroups[jGroup] = np.append(CurNodesGroups[jGroup], iNode)
                        CurNodesGroups[jGroup] = np.sort(CurNodesGroups[jGroup])
                        CurEntropy = CalcEntropy(CurNodesGroups, NodesTimeOfTravel, Method)
                        if CurEntropy >= Entropy:
                            CurNodesGroups = NodesGroups.copy()
                        else:
                            Entropy = CurEntropy
                            GroupChanged = True
                            for i in range(M):
                                CurNodesGroups[i] = np.sort(CurNodesGroups[i])
                            NodesGroups = CurNodesGroups.copy()
        print("iteration = "+str(iter)+", Entropy = {:}".format(Entropy))
        if not GroupChanged:
            break

    # # Make sure that the first group has the depot:
    # if NodesGroups[0][0] > Map.NumberOfDepots:
    #     for i in range(1,M):
    #         if NodesGroups[i][0] == 0:
    #             Temp = NodesGroups[0].copy()
    #             NodesGroups[0] = NodesGroups[i].copy()
    #             NodesGroups[i] = Temp.copy()
    #             break
    # Organize the groups:
    CurGroupIntegration = NodesGroups[0]
    for i in range(1,M-1):
        Entropy = np.zeros((M,))+np.inf
        for j in range(M-i):
            Group = np.append(CurGroupIntegration, NodesGroups[j+i])
            Entropy[i+j] = CalcEntropy([Group], NodesTimeOfTravel, Method)
        Group_MinEntropy = np.argmin(Entropy)
        # Set Group_MinEntropy as Group number i:
        Temp = NodesGroups[i].copy()
        NodesGroups[i] = NodesGroups[Group_MinEntropy].copy()
        NodesGroups[Group_MinEntropy] = Temp.copy()
        # Update CurGroupIntegration:
        CurGroupIntegration = np.append(CurGroupIntegration, NodesGroups[i])
            
    # Final Enthropy:
    for i in range(M):
        NodesGroups_i = []
        NodesGroups_i.append(NodesGroups[i])
        print("Final Entropy Group", i,": ", CalcEntropy(NodesGroups_i, NodesTimeOfTravel, Method))    

    # Print summary:
    print("Final Entropy: ", CalcEntropy(NodesGroups, NodesTimeOfTravel, Method))    
    for i in range(M):
        GroupTimeMatrix = CreateGroupMatrix(NodesGroups[i], NodesTimeOfTravel)
        w, v = np.linalg.eig(GroupTimeMatrix)
        w = np.sort(np.abs(w))[::-1]
        print("Group {:} - Number of Nodes: {:}, Entropy: {:}, Max Eigenvalue: {:}".format(i, len(NodesGroups[i]), CalcEntropy([NodesGroups[i]], NodesTimeOfTravel, Method), np.abs(w[0:3])))

# Plot the groups
    if np.max(Map.NodesPosition)>0 and isplot==True:
        col_vec = ['m','y','b','r','g','c','k']
        leg_str = list()
        plt.figure()
        if MustIncludeNodeZero==True:
            plt.scatter(Map.NodesPosition[Map.Depots,0], Map.NodesPosition[Map.Depots,1], c='k', s=50)
            leg_str.append('Depot')
        for i in range(M):
            plt.scatter(Map.NodesPosition[NodesGroups[i][i1:],0], Map.NodesPosition[NodesGroups[i][i1:],1], s=50, c=col_vec[i%len(col_vec)])
            leg_str.append('Group '+str(i)+" Number of Nodes: {}".format(len(NodesGroups[i])-1))
        for i in ChargingStations:
            plt.scatter(Map.NodesPosition[i,0], Map.NodesPosition[i,1], c='c', s=15)
        leg_str.append('Charging Station')
        plt.legend(leg_str)
        for i in range(N):
            colr = 'r' if i in Map.Depots else 'c'
            colr = 'k' if i in ChargingStations else colr
            plt.text(Map.NodesPosition[i,0]+1,Map.NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=30)
        plt.xlim((-100,100))
        plt.ylim((-100,100))
        plt.grid()
        plt.show()

    return NodesGroups


# def ConnectSubGroups(PltParams: DataTypes.PlatformParams, NominalPlan: DataTypes.NominalPlanning, NodesTrajectoryGroup, NodesTrajectorySubGroup, isplot: bool = False):

#     # Create the charging stations data
#     ChargingStations = np.array([],dtype=int)
#     for i in NominalPlan.ChargingStations:
#         if i in NodesTrajectoryGroup or i in NodesTrajectorySubGroup:
#             ChargingStations = np.append(ChargingStations,i)
#     if ChargingStations.shape[0] == 0:
#         ChargingStations = np.array([0])
#     BestChargingStationsData = DataTypes.ChargingStations(ChargingStations)

#     # Connect the subgroups
#     CostGroup = np.inf
#     BestNodesTrajectoryGroup = []
#     NodesTrajectorySubGroup = NodesTrajectorySubGroup[0:-1]
    
#     for reverse_i in range(2):
#         if reverse_i == 1:
#             NodesTrajectoryGroup = NodesTrajectoryGroup[::-1]
#         for i in range(1,len(NodesTrajectoryGroup)):
#             for j in range(len(NodesTrajectorySubGroup)):
#                 for reverse in range(2):
#                     PotentialTrajectory = []
#                     for k in range(len(NodesTrajectoryGroup)+len(NodesTrajectorySubGroup)-1):
#                         if k < i:
#                             PotentialTrajectory.append(NodesTrajectoryGroup[k])
#                         elif k < i+len(NodesTrajectorySubGroup):
#                             if reverse == 0:
#                                 PotentialTrajectory.append(NodesTrajectorySubGroup[(k-i+j)%len(NodesTrajectorySubGroup)])
#                             elif reverse == 1:
#                                 PotentialTrajectory.append(NodesTrajectorySubGroup[(i+len(NodesTrajectorySubGroup)+j-k)%len(NodesTrajectorySubGroup)])
#                         else:
#                             PotentialTrajectory.append(NodesTrajectoryGroup[k-len(NodesTrajectorySubGroup)])
#                     PotentialTrajectory.append(0)

#                     #Calculate the cost of the potential trajectory:
#                     TrajMeanTime = 0.0
#                     TrajMeanEnergy = NominalPlan.InitialChargeStage
#                     TrajSigmaTime2 = 0.0
#                     TrajSigmaEnergy2 = 0.0
#                     ChargingStationsData = DataTypes.ChargingStations(ChargingStations)
#                     for iNode in range(len(PotentialTrajectory)-1):
#                         i1 = PotentialTrajectory[iNode]
#                         i2 = PotentialTrajectory[iNode+1]
#                         TrajMeanTime += NominalPlan.NodesTimeOfTravel[i1,i2]
#                         TrajSigmaTime2 += NominalPlan.TravelSigma2[i1,i2]
#                         TrajMeanEnergy += NominalPlan.NodesEnergyTravel[i1,i2]
#                         TrajSigmaEnergy2 += NominalPlan.NodesEnergyTravelSigma2[i1,i2]
#                         # Check if the trajectory is feasible:
#                         if TrajMeanEnergy + ChargingStationsData.MaxChargingPotential - NominalPlan.EnergyAlpha*np.sqrt(TrajSigmaEnergy2) < 0.0:
#                             TrajMeanTime = np.inf
#                             break
#                         # Update the charging stations data:
#                         if np.any(i2 == ChargingStationsData.ChargingStationsNodes):
#                             arg_i = np.argwhere(i2 == ChargingStationsData.ChargingStationsNodes)[0][0]
#                             ChargingStationsData.MaxChargingPotential += PltParams.BatteryCapacity - (TrajMeanEnergy+ChargingStationsData.MaxChargingPotential)
#                             ChargingStationsData.EnergyEntered[arg_i] = TrajMeanEnergy

                    
#                     ChargeTime = max(0.0,-(TrajMeanEnergy - NominalPlan.EnergyAlpha*np.sqrt(TrajSigmaEnergy2))/NominalPlan.StationRechargePower)
#                     Cost = TrajMeanTime + ChargeTime + NominalPlan.TimeAlpha*np.sqrt(TrajSigmaTime2)
#                     if Cost < CostGroup:
#                         ChargeNeeded = ChargeTime*NominalPlan.StationRechargePower
#                         iChargingStation = 0
#                         ChargingNodeSquence = np.argsort(ChargingStationsData.EnergyEntered)[::-1]
#                         while iChargingStation < len(ChargingStationsData.ChargingStationsNodes):
#                             iCharge = ChargingNodeSquence[iChargingStation]
#                             ChargingStationsData.EnergyExited[iCharge] = min(PltParams.BatteryCapacity*0.95,ChargingStationsData.EnergyEntered[iCharge] + ChargeNeeded)
#                             ChargingStationsData.ChargingTime[iCharge] = (ChargingStationsData.EnergyExited[iCharge] - ChargingStationsData.EnergyEntered[iCharge])/NominalPlan.StationRechargePower
#                             for ii in range(iChargingStation+1,len(ChargingStationsData.ChargingStationsNodes)):
#                                 ChargingStationsData.EnergyEntered[ChargingNodeSquence[ii]] += ChargingStationsData.ChargingTime[iCharge]*NominalPlan.StationRechargePower
#                             ChargeNeeded -= ChargingStationsData.EnergyExited[iCharge] - ChargingStationsData.EnergyEntered[iCharge]
#                             iChargingStation += 1
#                         CostGroup = Cost
#                         BestNodesTrajectoryGroup = PotentialTrajectory
#                         BestChargingStationsData = ChargingStationsData
#                         print("Connected Traj With Cost = {:}".format(Cost))
#                         if isplot == True:
#                             PlotSubGroups(NominalPlan, NodesTrajectoryGroup, NodesTrajectorySubGroup, PotentialTrajectory)



#     return BestNodesTrajectoryGroup, CostGroup, BestChargingStationsData



# def PlotSubGroups(NominalPlan: DataTypes.NominalPlanning, Group1, Group2, Group3 = []):

#     plt.figure()
#     leg_str = []
#     leg_str.append('Group 1 - '+str(Group1))
#     leg_str.append('Group 2 - '+str(Group2))
#     leg_str.append('Group 3 - '+str(Group3))
#     for i in Group1:
#         plt.plot(NominalPlan.NodesPosition[i,0].T,NominalPlan.NodesPosition[i,1].T,'o',linewidth=10, color='r')
#     for i in Group2:
#         plt.plot(NominalPlan.NodesPosition[i,0].T,NominalPlan.NodesPosition[i,1].T,'o',linewidth=10, color='b')
#     plt.grid('on')
#     # plt.xlim((-100,100))
#     # plt.ylim((-100,100))
#     for i in range(len(Group1)-1):
#         plt.arrow(NominalPlan.NodesPosition[Group1[i],0],NominalPlan.NodesPosition[Group1[i],1],NominalPlan.NodesPosition[Group1[i+1],0]-NominalPlan.NodesPosition[Group1[i],0],NominalPlan.NodesPosition[Group1[i+1],1]-NominalPlan.NodesPosition[Group1[i],1], width= 1, color='r')
#     for i in range(len(Group2)-1):
#         plt.arrow(NominalPlan.NodesPosition[Group2[i],0],NominalPlan.NodesPosition[Group2[i],1],NominalPlan.NodesPosition[Group2[i+1],0]-NominalPlan.NodesPosition[Group2[i],0],NominalPlan.NodesPosition[Group2[i+1],1]-NominalPlan.NodesPosition[Group2[i],1], width= 1, color='g')
#     if len(Group3)>0:
#         for i in range(len(Group3)-1):
#             plt.arrow(NominalPlan.NodesPosition[Group3[i],0],NominalPlan.NodesPosition[Group3[i],1],NominalPlan.NodesPosition[Group3[i+1],0]-NominalPlan.NodesPosition[Group3[i],0],NominalPlan.NodesPosition[Group3[i+1],1]-NominalPlan.NodesPosition[Group3[i],1], width= 0.1, color='b')
#     for i in Group1:
#         colr = 'r' if i==0 else 'c'
#         colr = 'k' if i in NominalPlan.ChargingStations else colr
#         plt.text(NominalPlan.NodesPosition[i,0]+1,NominalPlan.NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=30)
#     for i in Group2:
#         colr = 'r' if i==0 else 'c'
#         colr = 'k' if i in NominalPlan.ChargingStations else colr
#         plt.text(NominalPlan.NodesPosition[i,0]+1,NominalPlan.NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=30)
#     plt.legend(leg_str)

#     plt.show()