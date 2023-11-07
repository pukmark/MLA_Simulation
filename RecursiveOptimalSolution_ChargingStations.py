import numpy as np
import DataTypes as DT
from copy import deepcopy
import time
from multiprocessing import Pool, Lock, Value

SharedBestCost = Value('d', 1.0e6)
TimeLastSolutionFound = Value('d', 0.0)
NumberOfTrajExplored = Value('i', 0)
StartTime = Value('d', 0)
StopProgram = Value('b', 0)
DeltaTimeToStop = Value('d', 0)

def SolveRecursive_ChargingStations(Map: DT.MapType, 
                                    i_CurrentNode, 
                                    TourTime,
                                    TourTimeUncertainty,
                                    EnergyLeft,
                                    EnergyLeftUncertainty,
                                    NodesTrajectory, 
                                    BestPlan: DT.Plan):

    # append current node to trajectory:
    NodesTrajectory.append(i_CurrentNode)
    
    # Check if current node is has a optimal or feasible potential:
    Nodes2Go = list(set(range(Map.N)) - set(NodesTrajectory))
    if len(Nodes2Go) >= 1:
        EstMinTimeToGo = np.min(Map.MapTime[i_CurrentNode,Nodes2Go]) + np.sum(np.sort(np.min(Map.MapTime[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(Map.MapTime[Nodes2Go,0])
        EstMinEnergyToGo = np.max(Map.MapEnergy[i_CurrentNode,Nodes2Go]) + np.sum(np.sort(np.max(Map.MapEnergy[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.max(Map.MapEnergy[Nodes2Go,0])
        EstMinTimeToGoUncertainty = np.sqrt(np.min(Map.MapTimeCov[i_CurrentNode,Nodes2Go])+np.sum(np.sort(np.min(Map.MapTimeCov[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(Map.MapTimeCov[Nodes2Go,0]) + TourTimeUncertainty**2)
        EstMinEnergyToGoUncertainty = np.sqrt(np.min(Map.MapEnergyCov[i_CurrentNode,Nodes2Go])+np.sum(np.sort(np.min(Map.MapEnergyCov[Nodes2Go,:][:,Nodes2Go],axis=0))[:-1]) + np.min(Map.MapEnergyCov[Nodes2Go,0]) + EnergyLeftUncertainty**2)        

        # Current Trajectory not feasible or not better than best plan:
        EstMaxChargingPotential = Map.CS.BatteryCapacity*len(np.where(BestPlan.CS.SOC_Entered==0.0)[0])
        if (EnergyLeft + EstMinEnergyToGo + EstMaxChargingPotential + BestPlan.CS.PotentialCharging - Map.EnergyAlpha*EstMinEnergyToGoUncertainty < 0.0) or (EstMinTimeToGo + TourTime + Map.TimeAlpha*EstMinTimeToGoUncertainty >= SharedBestCost.value):
            BestPlan.Cost = np.inf
            BestPlan.NodesTrajectory = NodesTrajectory.copy()
            return BestPlan

    # Update ChargingStationsData:
    if np.any(i_CurrentNode == Map.CS.Nodes):
        BestPlan.CS.PotentialCharging += Map.CS.BatteryCapacity - (EnergyLeft+BestPlan.CS.PotentialCharging)
        arg = np.argwhere(Map.CS.Nodes == i_CurrentNode)
        BestPlan.CS.SOC_Entered[arg] = EnergyLeft
    
    # Chaeck if all nodes are visited:
    if (len(NodesTrajectory) == Map.N) and (Map.MustVisitAllNodes==True):
         # Go back to Depot if Energy Allows:
        EnergyLeftUncertaintyToDepot = np.sqrt(EnergyLeftUncertainty**2 + Map.MapEnergyCov[i_CurrentNode,0])
        EnergyLeftToDepot = EnergyLeft + Map.MapEnergy[i_CurrentNode,0]
        if EnergyLeftToDepot + BestPlan.CS.PotentialCharging - Map.EnergyAlpha*EnergyLeftUncertaintyToDepot < 0.0:
            BestPlan.Cost = np.inf
            BestPlan.NodesTrajectory = NodesTrajectory.copy()
            return BestPlan
        else:
            NodesTrajectory.append(0)
            Cost = TourTime + Map.MapTime[i_CurrentNode,0] + Map.TimeAlpha*np.sqrt(TourTimeUncertainty**2+Map.MapTimeCov[i_CurrentNode,0])
            if EnergyLeftToDepot - Map.EnergyAlpha*EnergyLeftUncertaintyToDepot < 0.0:
                EnergyForRecharge = -(EnergyLeftToDepot - Map.EnergyAlpha*EnergyLeftUncertaintyToDepot)
                Cost += EnergyForRecharge/Map.CS.ChargingRate
                if Cost >= SharedBestCost.value:
                    BestPlan.Cost = np.inf
                    BestPlan.NodesTrajectory = NodesTrajectory.copy()
                    return BestPlan
                ChargingNodeSquence = np.argsort(BestPlan.CS.SOC_Entered)[::-1]
                for i in range(len(BestPlan.CS.Nodes)):
                    iCharge = ChargingNodeSquence[i]
                    i_MaxEnergyToRecharge = Map.CS.BatteryCapacity - BestPlan.CS.SOC_Entered[iCharge]
                    BestPlan.CS.SOC_Exited[iCharge] =BestPlan.CS.SOC_Entered[iCharge] + min(i_MaxEnergyToRecharge, EnergyForRecharge)
                    BestPlan.CS.ChargingTime[iCharge] = (BestPlan.CS.SOC_Exited[iCharge]-BestPlan.CS.SOC_Entered[iCharge])/BestPlan.CS.ChargingRate
                    for ii in range(i+1,len(BestPlan.CS.Nodes)):
                            BestPlan.CS.SOC_Entered[ChargingNodeSquence[ii]] += BestPlan.CS.ChargingTime[iCharge]*BestPlan.CS.ChargingRate
                    EnergyForRecharge -= min(i_MaxEnergyToRecharge, EnergyForRecharge)
            BestPlan.Cost = Cost
            BestPlan.NodesTrajectory = NodesTrajectory.copy()
            return BestPlan
    elif (len(NodesTrajectory) == Map.N) and (Map.MustVisitAllNodes==False):
        BestPlan.Cost = TourTime + Map.TimeAlpha*TourTimeUncertainty
        BestPlan.NodesTrajectory = NodesTrajectory.copy()
        return BestPlan
             
    # Move To next node:
    # i_array = (Map.NodesTimeOfTravel[i_CurrentNode,:] + Map.TimeAlpha*Map.TravelSigma[i_CurrentNode,:]).argsort()
    i_array = Map.MapTime[i_CurrentNode,:].argsort()
    for i in range(len(NodesTrajectory)):
        i_array = np.delete(i_array,np.where(i_array ==NodesTrajectory[i]))
    
    if i_CurrentNode == 0:
        t = time.time()
    for iNode in i_array:
        if np.any(np.array([iNode]) == NodesTrajectory): # Node already visited
            continue
        if StopProgram.value == False:
            if TimeLastSolutionFound.value > 0.0:  
                if time.time()-TimeLastSolutionFound.value > DeltaTimeToStop.value and SharedBestCost.value<np.inf: # 2 hours without improvement
                    print("No Improvement for 5 Min. Stopping... Stopping at:", NodesTrajectory)
                    StopProgram.value = True
                if time.time()-TimeLastSolutionFound.value > DeltaTimeToStop.value*10 and SharedBestCost.value==np.inf: # 2 hours without improvement
                    print("No Feasiable Solutions Found for 10 Min. Stopping...")
                    StopProgram.value = True
        else:
            break

        EnergyLeftNext = EnergyLeft + Map.MapEnergy[i_CurrentNode,iNode]
        EnergyLeftUncertaintyNext = np.sqrt(EnergyLeftUncertainty**2 + Map.MapEnergyCov[i_CurrentNode,iNode])
        TourTimeNext = TourTime + Map.MapTime[i_CurrentNode,iNode]
        TourTimeUncertaintyNext = np.sqrt(TourTimeUncertainty**2 + Map.MapTimeCov[i_CurrentNode,iNode])
        if EnergyLeftNext + BestPlan.CS.PotentialCharging - Map.EnergyAlpha*EnergyLeftUncertaintyNext < 0.0 or TourTimeNext + Map.TimeAlpha*TourTimeUncertaintyNext>=SharedBestCost.value:
            continue
        
        Plan = SolveRecursive_ChargingStations(Map= Map, 
                                                        i_CurrentNode= iNode, 
                                                        TourTime= TourTimeNext,
                                                        TourTimeUncertainty= TourTimeUncertaintyNext,
                                                        EnergyLeft= EnergyLeftNext,
                                                        EnergyLeftUncertainty= EnergyLeftUncertaintyNext,
                                                        NodesTrajectory= NodesTrajectory.copy(), 
                                                        BestPlan=deepcopy(BestPlan))
        if ((len(Plan.NodesTrajectory) == Map.N+1 and Map.MustVisitAllNodes==True) or (len(Plan.NodesTrajectory) == Map.N and Map.MustVisitAllNodes==False)) and (Plan.Cost < BestPlan.Cost):
            BestPlan = Plan
            if BestPlan.Cost < SharedBestCost.value:
                # print('New Best Plan Found: ', BestPlan.NodesTrajectory, BestPlan.Cost)
                TimeLastSolutionFound.value = time.time()
                SharedBestCost.value = BestPlan.Cost

    # if len(NodesTrajectory) == 3 and StopProgram.value == False:
    #     NumberOfTrajExplored.value += 1
    #     Explored = NumberOfTrajExplored.value/((Map.N-2)*(Map.N-1))*100
    #     print("Done!",NodesTrajectory," Trajectories Explored[%]:", "{:.3}".format(Explored), "TimeLeft: ", "{:.3}".format((time.time()-StartTime.value)*(100-Explored)/Explored/60.0), "[min]")
    return BestPlan

def SolveParallelRecursive_ChargingStations(Map: DT.MapType, 
                                    i_CurrentNode, 
                                    TourTime,
                                    TourTimeUncertainty,
                                    EnergyLeft,
                                    EnergyLeftUncertainty,
                                    NodesTrajectory, 
                                    BestPlan: DT.Plan,
                                    MaxCalcTimeFromUpdate: float = 60.0):
    
    SharedBestCost.value = 1.0e6
    TimeLastSolutionFound.value = time.time()
    StartTime.value = time.time()
    NumberOfTrajExplored.value = 0
    StopProgram.value = False
    DeltaTimeToStop.value = MaxCalcTimeFromUpdate
    BestPlan.CS = Map.CS

    # append current node to trajectory:
    if Map.N <= 11:
        BestPlan = SolveRecursive_ChargingStations(Map,
                                                    i_CurrentNode = i_CurrentNode, 
                                                    TourTime = TourTime,
                                                    TourTimeUncertainty = TourTimeUncertainty,
                                                    EnergyLeft = EnergyLeft,
                                                    EnergyLeftUncertainty = EnergyLeftUncertainty,
                                                    NodesTrajectory = NodesTrajectory.copy(), 
                                                    BestPlan=deepcopy(BestPlan))
        return BestPlan
    args = []
    i_array = Map.MapTime[i_CurrentNode,:].argsort()
    i_array = i_array[i_array != i_CurrentNode]

    Times = np.zeros(((Map.N-1)*(Map.N-2),1))
    indxes = np.zeros(((Map.N-1)*(Map.N-2),2), dtype=int)
    k=0
    for i in range(1,Map.N):
        for j in range(1,Map.N):
            if i==j:
                continue
            TourTime = Map.MapTime[i_CurrentNode,i] + Map.MapTime[i,j]
            TourTimeUncertainty = np.sqrt(Map.MapTimeCov[i_CurrentNode,i] + Map.MapTimeCov[i,j])
            Times[k] = TourTime + TourTimeUncertainty*Map.TimeAlpha
            indxes[k,:] = [i,j]
            k+=1
    indxes = indxes[np.argsort(Times[:,0]),:]

    for indx in indxes:
        i = indx[0]
        j = indx[1]
        TourTime_ij = TourTime +Map.MapTime[0,i] + Map.MapTime[i,j]
        TourTimeUncertainty_ij = np.sqrt(TourTimeUncertainty**2 + np.sqrt(Map.MapTimeCov[0,i] + Map.MapTimeCov[i,j]))
        EnergyLeft_ij = EnergyLeft + Map.MapEnergy[0,i] + Map.MapEnergy[i,j]
        EnergyLeftUncertainty_ij = np.sqrt(EnergyLeftUncertainty**2 + Map.MapEnergyCov[0,i] + Map.MapEnergyCov[i,j])
        NodesTrajectory_i = NodesTrajectory.copy()
        NodesTrajectory_i.append(i)
        args.append((Map, j, TourTime_ij, TourTimeUncertainty_ij, EnergyLeft_ij, EnergyLeftUncertainty_ij, NodesTrajectory_i.copy(), deepcopy(BestPlan)))


    with Pool(14) as pool:
        results = pool.starmap(SolveRecursive_ChargingStations, args)
        # pool.close()
        # pool.join()

    Cost = np.inf
    for result in results:
        if result.Cost < Cost and result.Cost > 0.0:
            BestPlan = result
            Cost = result.Cost
    
    # print('Final Best Plan Found: ', BestPlan.NodesTrajectory, BestPlan.Cost)
    
    return BestPlan
