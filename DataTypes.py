import numpy as np

class MapType():
    def __init__(self, MapTime, MapEnergy, TimeMapCov, EnergyMapCov, Depots, EnergyAlpha, TimeAlpha, NodesPosition: np.ndarray = None, MustVisitAllNodes: bool = True):
        self.N = len(MapTime)
        self.NumberOfCars = len(Depots)
        self.Depots = Depots
        self.NumberOfDepots = len(np.unique(Depots).tolist())
        self.MapTime = MapTime
        self.MapEnergy = MapEnergy
        self.MapTimeCov = TimeMapCov
        self.MapEnergyCov = EnergyMapCov
        self.NodesVisited = np.zeros(self.N, dtype=bool)
        self.EnergyAlpha = EnergyAlpha
        self.TimeAlpha = TimeAlpha
        self.NodesPosition = NodesPosition
        self.MustVisitAllNodes = MustVisitAllNodes
        self.CS = ChargingStationsType()

class VehicleEstState():
    def __init__(self, InitialCharge, InitialNode, InitialMapTime, InitialMapEnergy, InitialMapTimeCov, InitialMapEnergyCov):
        self.N = len(InitialMapTime)
        self.MapTime = InitialMapTime
        self.MapEnergy = InitialMapEnergy
        self.MapTimeCov = InitialMapTimeCov
        self.MapEnergyCov = InitialMapEnergyCov
        self.SOC = InitialCharge
        self.Node = InitialNode
        self.NodesHistory = np.zeros(self.N, dtype=int)

class ChargingStationsType():
    def __init__(self, Nodes: list = [], Rate: float = 0.0, BatteryCapacity: float = 100.0):
        self.N = len(Nodes)
        self.Nodes = Nodes
        self.ChargingRate = Rate
        self.BatteryCapacity = BatteryCapacity
        self.PotentialCharging = 0.0
        self.ChargingTime = np.zeros((self.N,))
        self.SOC_Entered = np.zeros((self.N,))
        self.SOC_Exited = np.zeros((self.N,))

        
class Plan():
    def __init__(self,N):
        self.Cost = np.inf
        self.NodesTrajectory = np.zeros((N,1), dtype=int)
        self.CS = ChargingStationsType()