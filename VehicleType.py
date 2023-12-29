import numpy as np
from DataTypes import *
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.dae import *
import copy
import matplotlib.pyplot as plt
from scipy import spatial
from ReachSets import *
import polytope as pt
import gurobipy
from DecisionManager import * 
from SituationalAwareness import * 

# estimated state
class VehicleEstState:
    def __init__(self, iCar: int, iDepot: int, InitialSOC: float):
        self.iCar = iCar
        self.SOC = InitialSOC
        self.Node = iDepot
        self.iDepot = iDepot
        self.NodesHistory = [iDepot]
        self.CurrentNode = iDepot
        self.InTransit = False
        self.ChargingState = False
        self.InTransition_StartNode = -1
        self.InTransition_TargetNode = -1
        self.SOC_rate = 0.0
        
        # added new unicycle state information below, velocity was same as above.
        self.Pos = [0.0,0.0] #x, y position
        self.Vel = 0.0 # m/s, velocity
        self.Theta = 0.0 # this will be steering angle
    
    def Update(self, iNode: int, SOC: float):
        self.Node = iNode
        self.SOC = SOC
    

class VehicleState:
    def __init__(self, iCar: int, iDepot: int, InitialSOC: float):
        self.iCar = iCar
        self.SOC = InitialSOC
        self.Node = iDepot
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
        
        # added new unicycle state information below, velocity was same as above.
        self.Pos = [0.0,0.0] #x, y position
        self.Theta = 0.0 # this will be steering angle
        self.Vel = 0.0
        
        
class VehicleType():
    def __init__(self, i: int, iDepot: int, InitialState: float, Map: MapType):
        self.ID = i
        self.State = VehicleState(i, iDepot, InitialState) # this will be the real state
        self.Plan = [] # This will be "Pi1"
        self.Map = Map
        self.ReturnedToDepot = False
        self.Trajectory = [copy.copy(self.State)]
        self.CurrentDayTrajectory = [copy.copy(self.State)]
        self.InputTrajectory = []
        self.CurrentDayInputTrajectory = []
        self.DM = DM(self.Plan, VehicleState(i, iDepot, InitialState), Map, i) # these will be classes in the Vehicle Type I guess! 
        self.SA = SA( VehicleState(i, iDepot, InitialState), Map, i) # these will be classes in the Vehicle Type I guess! 
        self.index = 0
        self.State.Time = 0.0
        self.Model = [0.04, 0.06]
        self.Temperature = [298] # Kelvin
        self.ReachabilityFlag = False
        self.ReachabilityAnalysisComplete = False
        self.ReachableSets = []

    def UpdateVehicle(self, Time: float, dT: float, Temp):
        # This function updates the vehicle's state
        
        if self.ReturnedToDepot == True:
            return np.inf
        
        if self.State.InTransit == True:
            if np.linalg.norm(self.State.Pos - self.Map.NodesPosition[self.State.InTransition_TargetNode,:]) < 0.002: # we are very close to the reference goal already:
                self.State.InTransit = False
                self.State.CurrentNode = self.State.InTransition_TargetNode
                self.State.NodesHistory.append(self.State.CurrentNode)
                self.State.Node = self.State.CurrentNode
                self.State.InTransition_StartNode = -1
                self.State.InTransition_TargetNode = -1
                
                if self.State.Node in self.Plan.CS.Nodes:
                    i_CS = np.argwhere(self.Plan.CS.Nodes == self.State.Node).tolist()[0][0]
                    if self.Plan.CS.ChargingTime[i_CS] > 0.0:
                        self.State.TgoForNextPhase = self.Plan.CS.ChargingTime[i_CS]
                        self.State.ChargingState = True
                        self.SOC_rate = self.Map.CS.ChargingRate
            else:
                print('Reference tracking Vehicle ' + str(self.ID))
                x0 = np.hstack((self.State.Pos, self.State.Theta, self.State.SOC, self.State.Vel))
                N = 10
                
                input_sequence, ol_state_sequence = self.solve_linear_lmpc(x0, N, dT)
                a_apply = input_sequence[0,0]
                w_apply = input_sequence[1,0]
                 
                self.State.Pos = [self.State.Pos[0] + dT * (self.State.Vel*np.cos(self.State.Theta)), self.State.Pos[1] + dT * (self.State.Vel*np.sin(self.State.Theta))]
                self.State.Theta = self.State.Theta + dT * w_apply
                #self.State.SOC = self.State.SOC - (0.0002*(Temp[-1] - 285)**4) * (dT *(self.Model[0] + np.random.normal(0, 0.005))*v_apply - dT * (self.Model[1] + np.random.normal(0, 0.005)))
                self.State.SOC = self.State.SOC - dT*self.Model[0]*self.State.Vel # - dT*self.Model[1]
                self.State.Vel = self.State.Vel + dT * a_apply
                self.State.TrackingCounter += 1
                self.index += 1 
                
                self.Trajectory.append(copy.copy(self.State))
                self.CurrentDayTrajectory.append(copy.copy(self.State))
                self.InputTrajectory.append(np.vstack((a_apply, w_apply)))
                self.CurrentDayInputTrajectory.append(np.vstack((a_apply, w_apply)))
                self.Temperature.append(np.squeeze(Temp[-1]))
                
                if self.ReachabilityFlag == True:
                    # do the forward reachability analysis to determine in what kind of (T, SOC) we could have reached the upcoming node
                    xN = [self.DM.Pi0[1][0][-1], self.DM.Pi0[1][1][-1]]
                    df = np.sqrt((self.State.Pos[0] - xN[0])**2 + (self.State.Pos[1] - xN[1])**2)
                    vf = 0.0
                    d0 = 0.0
                    v0 = self.State.Vel
                    soc0 = 100.0
                    x0 = np.array([[d0], [v0], [soc0]])
                    
                    Kmax = int(self.DM.Travel_Distance * self.DM.ReachabilityMarker / 3 / dT * 3) # this is somewhat arbitrary
                    set_list = self.DM.forward_reach_sets(x0, Kmax, dT)
                    
                    T_possible = []
                    soc_bounds = []
                    for i in range(1,len(set_list)):
                        print(i)
                        consider_set = set_list[i]
                        eps = 0.001
                        Xf = pt.Polytope(np.array([[1.0, 0, 0],
                                                  [-1.0, 0, 0],
                                                  [0, 1.0, 0],
                                                  [0, -1.0, 0],
                                                  [0, 0, 1.0],
                                                  [0, 0, -1.0]]),
                                        np.array([[df + eps],
                                                  [-df + eps],
                                                  [eps],
                                                  [eps],
                                                  [100],
                                                  [0.0]]))
                        C = consider_set.intersect(Xf)
                        if C.volume > 0.0:
                            print('Adding set ' + str(i))
                            T_possible.append((Time - self.DM.NodeStartTime) + i*dT)
                            f = C.project([3]).b
                            energy_costs = np.array([100 - f[0], 100 + f[1]])
                            soc_bounds.append(self.DM.NodeStartSOC - self.State.SOC + energy_costs.reshape((2,1)))

                    reachable_sets = []
                    for i in range(len(T_possible)):
                        reachable_sets.append([[T_possible[i]],[soc_bounds[i][0][0]]])
                        reachable_sets.append([[T_possible[i]],[soc_bounds[i][1][0]]])
                    
                    self.ReachableSets.append(reachable_sets)
                    self.ReachabilityFlag = False
                    self.ReachabilityAnalysisComplete = True
                
            
        if self.State.ChargingState == True:
            print('Charging! Vehicle ' + str(self.ID))
            if self.State.TgoForNextPhase > 1e-8:
                self.State.TgoForNextPhase -= dT 
                self.State.SOC += dT*self.SOC_rate 
                self.State.SOC = max(self.State.SOC, 100.0)
            else:
                self.State.ChargingState = False
        
        self.State.Time += dT
        return np.inf
    
    
    def solve_linear_lmpc(self, x0, N, Ts):
        
        ref_traj = self.DM.Pi0[1]
        Q_fun = np.arange(np.shape(ref_traj)[1]-1,-1,-1)
        
        x0 = np.squeeze(x0)
        
        theta_refs = self.DM.lmpc_theta

        soc_alpha = self.Model[0]
        model = pyo.ConcreteModel()
        model.nIDX = pyo.Set(initialize=range(0, N+1)) 
        model.nLambda = pyo.Set(initialize=range(0,np.shape(ref_traj)[1]))
        
        model.x = pyo.Var(model.nIDX) 
        model.y = pyo.Var(model.nIDX) 
        
        def fb1(model, i):
            return x0[2]
        model.theta = pyo.Var(model.nIDX, initialize=fb1)
        model.soc = pyo.Var(model.nIDX) 
        model.v = pyo.Var(model.nIDX)
        
        model.lam  = pyo.Var(model.nLambda, domain=pyo.Binary)

        model.eps = pyo.Var(pyo.Set(initialize=range(0,4)))
        
        model.a = pyo.Var(model.nIDX) 
        def fb(model, i):
            return 0
        model.w = pyo.Var(model.nIDX, initialize = fb)
        
       
        # 1. initialization
        model.constraint01 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.x[t] == x0[0] if t < 1 else pyo.Constraint.Skip )
        model.constraint02 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.y[t] == x0[1] if t < 1 else pyo.Constraint.Skip )
        model.constraint03 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.theta[t] == x0[2] if t < 1 else pyo.Constraint.Skip)
        model.constraint04 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.soc[t] == x0[3] if t < 1 else pyo.Constraint.Skip )
        model.constraint05 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.v[t] == x0[4] if t < 1 else pyo.Constraint.Skip )
        
    
        # 2. dynamics - NOTE THAT PYOMO HAS THETA IN DEGREES!!!!!!!!!!!
        model.constraint11 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.x[t+1] == model.x[t] + Ts * (model.v[t]*pyo.cos(theta_refs[t])) # pyomo is in degrees!!!!!!!!!
                                       if t < N else pyo.Constraint.Skip)
        model.constraint12 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.y[t+1] == model.y[t] + Ts * (model.v[t]*pyo.sin(theta_refs[t]))
                                       if t < N else pyo.Constraint.Skip)
        model.constraint13 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.theta[t+1] == model.theta[t] + Ts * model.w[t]
                                       if t < N else pyo.Constraint.Skip)
        model.constraint14 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.soc[t+1] == model.soc[t] - Ts * soc_alpha*model.v[t] # - Ts * soc_const
                                       if t < N else pyo.Constraint.Skip)
        model.constraint15 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.v[t+1] == model.v[t] + Ts * model.a[t] 
                                       if t < N else pyo.Constraint.Skip)
                                        
        # 5. velocity limits
        model.constraint41 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.v[t] <= 10.0)
        model.constraint42 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.v[t] >= 0.0)
        
        # theta limits
        model.constraint411 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.theta[t] <= np.pi)
        model.constraint421 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.theta[t] >= -np.pi)
        
        # 6. steering limits
        model.constraint61 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.w[t] <= 40)
        model.constraint62 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.w[t] >= -40)
        
        model.constraint81 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.a[t] <= 3)
        model.constraint82 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.a[t] >= -3)

        
        # constrain lambdas to sum to 1
        model.constraint83 = pyo.Constraint(expr = sum(model.lam[i] for i in model.nLambda) == 1.0)

        # terminal constraints: system must come to a stop
        model.constraint84 = pyo.Constraint(expr = sum(model.lam[i]*ref_traj[0,i] for i in model.nLambda) == model.x[N] + model.eps[0])
        model.constraint85 = pyo.Constraint(expr = sum(model.lam[i]*ref_traj[1,i] for i in model.nLambda) == model.y[N] + model.eps[1])
        model.constraint86 = pyo.Constraint(expr = sum(model.lam[i]*ref_traj[2,i] for i in model.nLambda) == model.theta[N] + model.eps[2])
        model.constraint87 = pyo.Constraint(expr = sum(model.lam[i]*ref_traj[3,i] for i in model.nLambda) <= model.soc[N])
        model.constraint88 = pyo.Constraint(expr = sum(model.lam[i]*ref_traj[4,i] for i in model.nLambda) == model.v[N] + model.eps[3])
               
        model.cost = pyo.Objective(expr = sum(model.lam[i]*Q_fun[i] for i in model.nLambda) + sum((0.01*model.a[t]**2 + 0.5*model.w[t]**2) for t in model.nIDX) + 1000*sum(model.eps[i]**2 for i in range(0,4)), sense=pyo.minimize)

        # solve
        solver = SolverFactory('gurobi')
        results = solver.solve(model,tee=False)   
        
                    
        if (results.solver.termination_condition != pyo.TerminationCondition.optimal):
            print('solver was infeasible!')
        
        # extract variables
        x = [pyo.value(model.x[0])]
        y = [pyo.value(model.y[0])]
        theta = [pyo.value(model.theta[0])]
        soc = [pyo.value(model.soc[0])]
        v = [pyo.value(model.v[0])]     
        
        a = [pyo.value(model.a[0])]
        w = [pyo.value(model.w[0])]
        
        for t in range(N):
            if t < N-1:
                x.append(pyo.value(model.x[t+1]))
                y.append(pyo.value(model.y[t+1]))
                theta.append(pyo.value(model.theta[t+1]))
                soc.append(pyo.value(model.soc[t+1]))
                v.append(pyo.value(model.v[t+1]))
            if t < N-2:
                a.append(pyo.value(model.a[t+1]))
                w.append(pyo.value(model.w[t+1]))
        
        lam = [pyo.value(model.lam[0])]
        for t in range(len(Q_fun)-1):
            lam.append(pyo.value(model.lam[t+1]))
        
        eps = [pyo.value(model.eps[0])]
        for t in range(len(model.eps)-1):
            eps.append(pyo.value(model.eps[t+1]))
    
        input_sequence = np.vstack((a,w))
        ol_state_sequence = np.vstack((x, y, theta, soc, v))
        
        self.DM.lmpc_theta = np.hstack((theta[1:], theta[-1])) 

        # plot predicted OL trajectory
        plt.figure()
        plt.plot(ref_traj[0,:], ref_traj[1,:],'k')
        plt.plot(x, y,'r')
        plt.show()

        return input_sequence, ol_state_sequence
    
    def ConsolidateRouteInfo(self):
        TimeCost, EnergyCost = self.SA.consolidate(self.ReachableSets)
        return TimeCost, EnergyCost

    def UpdateSA(self, Time: float, dT: float):
        if self.ReturnedToDepot == True:
            5
        else:
            self.SA.update(self, Time, dT)
            self.DM.EstimatedState = copy.deepcopy(self.SA.EstimatedState) # this will transfer the position info, soc info
            self.DM.Model = copy.deepcopy(self.SA.Model) # this transfers the model 

    def UpdateDM(self, Time: float, dT: float):
        if self.ReturnedToDepot == True:
            5
        
        elif self.State.InTransit == False and self.State.ChargingState == False:
          
            self.DM.EstimatedState = copy.copy(self.SA.EstimatedState)
            self.DM.update(self, Time, dT)
     
            self.State.TrackingCounter = 0
            self.State.InTransit = True
            self.State.InTransition_StartNode = self.DM.EstimatedState.InTransition_StartNode
            self.State.InTransition_TargetNode = self.DM.EstimatedState.InTransition_TargetNode
            self.State.CurrentNode = -1
            self.State.Theta = self.DM.Pi0[1][2][0]
            self.index = 0
            self.ReachabilityAnalysisComplete = False
        
        elif self.State.InTransit == True and self.ReachabilityAnalysisComplete == False:            
            x0 = self.State.Pos
            xN = [self.DM.Pi0[1][0][-1], self.DM.Pi0[1][1][-1]]
            cur_dist = np.sqrt((x0[0] - xN[0])**2 + (x0[1] - xN[1])**2)
            if cur_dist/self.DM.Travel_Distance <= self.DM.ReachabilityMarker:
                self.ReachabilityFlag = True
                
    