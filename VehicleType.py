import numpy as np
from DataTypes import *

class VehicleEstState:
    def __init__(self, iCar: int, iDepot: int, InitialSOC: float):
        self.iCar = iCar
        self.SOC = InitialSOC
        self.Node = iDepot
        self.Velocity = 0.0
        self.iDepot = iDepot
        self.NodesHistory = [iDepot]
        self.CurrentNode = iDepot
        self.InTransit = False
        self.ChargingState = False
        self.InTransition_StartNode = -1
        self.InTransition_TargetNode = -1
        self.SOC_rate = 0.0
    
    def Update(self, iNode: int, SOC: float):
        self.Node = iNode
        self.SOC = SOC
    

        

class VehicleState:
    def __init__(self, iCar: int, iDepot: int, InitialSOC: float):
        self.iCar = iCar
        self.SOC = InitialSOC
        self.Node = iDepot
        self.Velocity = 0.0
        self.iDepot = iDepot
        self.NodesHistory = [iDepot]
        self.CurrentNode = iDepot
        self.InTransit = False
        self.ChargingState = False
        self.InTransition_StartNode = -1
        self.InTransition_TargetNode = -1
        self.TgoForNextPhase = 0.0
        self.SOC_rate = 0.0
        self.TimeReturnToDepot = 0.0


class VehicleType():
    def __init__(self, i: int, iDepot: int, InitialState: float, Map: MapType):
        self.ID = i
        self.EstimatedState = VehicleEstState(i, iDepot, InitialState)
        self.State = VehicleState(i, iDepot, InitialState)
        self.Plan = []
        self.Map = Map

        self.ReturnedToDepot = False

    def Update(self, Time: float, dT: float):
        if self.State.InTransit == False and self.State.ChargingState == False:
            CurPlan_i = np.argwhere(self.Plan.NodesTrajectory == self.State.Node).tolist()[0][0]
            NextNode = self.Plan.NodesTrajectory[CurPlan_i+1]
            self.State.InTransit = True
            self.State.InTransition_StartNode = self.State.Node
            self.State.InTransition_TargetNode = NextNode
            self.State.CurrentNode = -1
            self.State.TgoForNextPhase = self.Map.MapTime[self.State.InTransition_StartNode, self.State.InTransition_TargetNode] + np.random.normal(0,1)*np.sqrt(self.Map.MapTimeCov[self.State.InTransition_StartNode, self.State.InTransition_TargetNode])
            self.SOC_rate = (self.Map.MapEnergy[self.State.InTransition_StartNode, self.State.InTransition_TargetNode] + np.random.normal(0,1)*np.sqrt(self.Map.MapEnergyCov[self.State.InTransition_StartNode, self.State.InTransition_TargetNode]))/self.State.TgoForNextPhase
        
        # Integrate the state:
        self.State.TgoForNextPhase -= dT # Time Integration
        self.State.SOC += dT*self.SOC_rate
        # Phase Transition:
        if self.State.TgoForNextPhase < 1.0e-8:
            if self.State.ChargingState == True:
                self.State.ChargingState = False
            else:
                self.State.InTransit = False
                self.State.CurrentNode = self.State.InTransition_TargetNode
                self.State.NodesHistory.append(self.State.CurrentNode)
                self.State.Node = self.State.CurrentNode
                self.State.InTransition_StartNode = -1
                self.State.InTransition_TargetNode = -1
                self.State.TgoForNextPhase = 0.0
                # Check if the vehicle is in a charging station:
                if self.State.Node in self.Plan.CS.Nodes:
                    i_CS = np.argwhere(self.Plan.CS.Nodes == self.State.Node).tolist()[0][0]
                    if self.Plan.CS.ChargingTime[i_CS] > 0.0:
                        self.State.TgoForNextPhase = self.Plan.CS.ChargingTime[i_CS]
                        self.State.ChargingState = True
                        self.SOC_rate = self.Map.CS.ChargingRate
            

        
        return self.State.TgoForNextPhase if self.State.TgoForNextPhase > 0.0 else np.inf

            

    def UpdateSelfPlanner(self, Time: float):
        pass
        