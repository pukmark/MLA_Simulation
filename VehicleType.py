import numpy as np
from DataTypes import *
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.dae import *
import copy
import matplotlib.pyplot as plt
from scipy import spatial

# Implementation steps: 
# Step one: update the "true" vehicle model to be a unicycle
# Step two: add the Level 1 controller

# Journaling
# Do I have to implement both at the same time? 
# Maybe it's easier to implement the controller first actually, and just have it "pretend" calculate an input.
# In order to do the controller, we need to change some stuff around so that the Global Layer sends a time and SOC that it expects the vehicle to have.

# Option 1: Change the "plan" structure to output not just nodes, but also the time it will be there? 
# Option 2: Ask Mark to do this, and just assume we have it for now. Can always (to be safe) just use 10% as the SOC we must have by the time we get there.

# What's the high-level organization here? 
# Each vehicle will get a DM/SA thing, so it seems appropriate to put that here. 
# Currently, each vehicle has a "plan" from above that says what the order of nodes is. 
# Is one step to just do it on the real vehicle as opposed to the estimated vehicle state? sure. 
# So: "Vehicle.Plan" will refer to high-level plan. Plan = Pi1.

# Every time we reach a node, it's time to make a new Pi0 


# estimated state
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
        
        # added new unicycle state information below, velocity was same as above.
        self.Pos = [0.0,0.0] #x, y position
        self.Theta = 0.0 # this will be steering angle
        
        # added energy model
        # self.mhat0 = [0.04, 0.06]
    
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
        self.TgoForNextPhase = 0.0 # THIS IS DIFFERENT
        self.SOC_rate = 0.0
        self.TimeReturnToDepot = 0.0 # THIS IS DIFFERENT, and maybe not used?
        
        # added new unicycle state information below, velocity was same as above.
        self.Pos = [0.0,0.0] #x, y position
        self.Theta = 0.0 # this will be steering angle
        # added energy model
        # self.mhat0 = [0.04, 0.06]
        
class DM():
    def __init__(self, Pi1, EstState, Map, i):
        self.Pi1 = Pi1 # this will be nodes, energy, time, type "Vehicle.Plan"
        self.EstimatedState = EstState
        self.Map = Map # map of the world (i guess this is technically an estimate, but we'll run with it. just used for node positions, which we'll assume we know)
        self.ID = i # vehicle ID
        self.Model = [0.04, 0.06]
        
        
    def update(self, vehicle, Time, dT):
        # This function updates whenever we have reached a new node that is NOT a charging node! 
        # The function creates a trajectory to drive to the next node.
        
        # for when it's time: current estimated temperature can be accessed at vehicle.Temperature[-1]
        
        if vehicle.ReturnedToDepot == True:
            5
        
        elif self.EstimatedState.InTransit == False and self.EstimatedState.ChargingState == False:
            print('Updating Pi0 for Vehicle ' + str(self.ID))
            CurPlan_i = np.argwhere(self.Pi1.NodesTrajectory == self.EstimatedState.Node).tolist()[0][0] # at what index of the plan are we?
            Cur_Pos = self.EstimatedState.Pos
            
            NextNode = self.Pi1.NodesTrajectory[CurPlan_i+1] # what node are we going to? 
            Goal_Pos = self.Map.NodesPosition[NextNode,:]
            
            Goal_Time = self.Pi1.TimePlan[CurPlan_i+1] - Time # how much time we have to get to the next node
            Goal_Charge = self.Pi1.SOCPlan[CurPlan_i+1] # how much charge do we need to have left at the next node?
            
            x0 = np.hstack((Cur_Pos, self.EstimatedState.Theta, self.EstimatedState.SOC))
            
            # now we add the solve_batch function that calculates the optimal input sequence to apply to get from current node to next node
            input_sequence, ol_state_sequence = self.solve_batch(x0, Goal_Charge, Goal_Time, Goal_Pos, Time, dT)
            
            self.Pi0 = [input_sequence, ol_state_sequence]
            
            self.EstimatedState.TrackingCounter = 0
            self.EstimatedState.InTransit = True
            self.EstimatedState.InTransition_StartNode = self.EstimatedState.Node
            self.EstimatedState.InTransition_TargetNode = NextNode
            self.EstimatedState.CurrentNode = -1 # does this mean we are between nodes?
            
            
    def solve_batch(self, x0, goal_charge, goal_time, goal_loc, Time, Ts):
            # goal_charge = self.goal_charge
            # goal_time = self.goal_time
            # goal_loc = self.goal_loc
            # x0 = np.squeeze(self.xhat0)
            
            # TO DO: UPDATE THESE TO BE THE ESTIMATED MODEL!! 
            #soc_alpha = self.EstimatedState.mhat0[0]
            soc_alpha = self.Model[0]
            #soc_const = self.EstimatedState.mhat0[1]
            soc_const = self.Model[1]
            
            model = pyo.ConcreteModel()
            #N = 10*int(np.ceil(10*goal_time)/10) # 0.1s
            N = int(10*goal_time)
            model.nIDX = pyo.Set(initialize=range(0, N+1)) 
            
            model.x = pyo.Var(model.nIDX) 
            model.y = pyo.Var(model.nIDX) 
            model.theta = pyo.Var(model.nIDX)
            model.soc = pyo.Var(model.nIDX) 
            
            model.v = pyo.Var(model.nIDX) 
            model.w = pyo.Var(model.nIDX)
            
           
            # 1. initialization
            model.constraint01 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.x[t] == x0[0] if t < 1 else pyo.Constraint.Skip )
            model.constraint02 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.y[t] == x0[1] if t < 1 else pyo.Constraint.Skip )
            #model.constraint03 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.theta[t] == x0[2] if t < 1 else pyo.Constraint.Skip )
            model.constraint04 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.soc[t] == x0[3] if t < 1 else pyo.Constraint.Skip )
            
            
            model.constraint05 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.v[t] <= 5.0 * Ts if t < 1 else pyo.Constraint.Skip )
            model.constraint06 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.w[t] <= 100.0 * Ts if t < 1 else pyo.Constraint.Skip )
            model.constraint07 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.w[t] >= -100.0 * Ts if t < 1 else pyo.Constraint.Skip )
        
        
            # 2. dynamics - NOTE THAT PYOMO HAS THETA IN DEGREES!!!!!!!!!!!
            model.constraint11 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.x[t+1] == model.x[t] + Ts * (model.v[t]*pyo.cos(model.theta[t])) # pyomo is in degrees!!!!!!!!!
                                           if t < N else pyo.Constraint.Skip)
            model.constraint12 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.y[t+1] == model.y[t] + Ts * (model.v[t]*pyo.sin(model.theta[t]))
                                           if t < N else pyo.Constraint.Skip)
            model.constraint13 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.theta[t+1] == model.theta[t] + Ts * model.w[t]
                                           if t < N else pyo.Constraint.Skip)
            model.constraint14 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.soc[t+1] == model.soc[t] - Ts * soc_alpha*model.v[t] # - Ts * soc_const
                                           if t < N else pyo.Constraint.Skip)
            
            # 4. SOC charge constraint
            model.constraint31 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.soc[t] >= goal_charge)
                                            
            
            # 5. velocity limits
            model.constraint41 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.v[t] <= 10.0)
            model.constraint42 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.v[t] >= 0.0)
            
            # theta limits
            model.constraint411 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.theta[t] <= 180)
            model.constraint421 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.theta[t] >= -180)
            
            
            
            # 5b. velocity rate limits
            model.constraint43 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.v[t+1] - model.v[t] <= 5.0 * Ts
                                                if t < N else pyo.Constraint.Skip)
            model.constraint44 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.v[t+1] - model.v[t] >= - 5.0 * Ts
                                                if t < N else pyo.Constraint.Skip)
            
            # 6. steering limits
            model.constraint61 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.w[t] <= 40)
            model.constraint62 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.w[t] >= -40)
            
            # 6b. steering rate limits
            model.constraint63 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.w[t+1] - model.w[t] <= 100.0 * Ts
                                                if t < N else pyo.Constraint.Skip)
            model.constraint64 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.w[t+1] - model.w[t] >= -100.0 * Ts
                                                if t < N else pyo.Constraint.Skip)
            
            # terminal constraint, system must come to a stop
            model.constraint71 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.v[t] == 0.0 if t >= N-1 else pyo.Constraint.Skip)
            model.constraint72 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.x[t] == goal_loc[0] if t >= N else pyo.Constraint.Skip)
            model.constraint73 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.y[t] == goal_loc[1] if t >= N else pyo.Constraint.Skip)
            
            # cost: minimize squared norm to lref
            model.cost = pyo.Objective(expr = sum(((goal_loc[0] - model.x[t])**2 + (goal_loc[1] - model.y[t])**2 + 0.1*model.v[t]**2 + 0.1*model.w[t]**2) for t in model.nIDX if t < N), sense=pyo.minimize)
            
            # solve
            solver = SolverFactory('ipopt')
            solver.options["print_level"] = 5
            results = solver.solve(model,tee=False)   
            
            if (results.solver.termination_condition == pyo.TerminationCondition.infeasible):
                print('solver was infeasible!')
                # we'll use this later for the re-planning! 
            
            # extract inputs
            x = [pyo.value(model.x[0])]
            y = [pyo.value(model.y[0])]
            theta = [pyo.value(model.theta[0])]
            soc = [pyo.value(model.soc[0])]
            v = [pyo.value(model.v[0])]
            w = [pyo.value(model.w[0])]
            time_vec = [Time]
            
            for t in range(N):
                if t < N:
                    x.append(pyo.value(model.x[t+1]))
                    y.append(pyo.value(model.y[t+1]))
                    theta.append(pyo.value(model.theta[t+1]))
                    soc.append(pyo.value(model.soc[t+1]))
                    time_vec.append(time_vec[-1] + Ts)
                if t < N-1:
                    v.append(pyo.value(model.v[t+1]))
                    w.append(pyo.value(model.w[t+1]))
            
            
            # plt.plot(x,y,'r')
            # plt.scatter(goal_loc[0], goal_loc[1])
        
            # calculate distance
            dist = np.sqrt((goal_loc[0] - x)**2 + (goal_loc[1] - y)**2)
            if min(abs(dist)) < 0.1:
                stop_index = np.where(abs(dist)<0.1)
                stop_index = stop_index[0][0]
            else:
                print('did not get within 0.05m of goal')
                stop_index = -2
            
            
            input_sequence = np.vstack((v[:stop_index],w[:stop_index]))
            ol_state_sequence = np.vstack((x[:stop_index+1], y[:stop_index+1], theta[:stop_index+1], soc[:stop_index+1], time_vec[:stop_index+1]))
            # plt.plot(x[:stop_index+1], y[:stop_index+1],'r')
            # plt.scatter(goal_loc[0], goal_loc[1])
            
            return input_sequence, ol_state_sequence
            

        
class SA():
    def __init__(self, EstState, Map, i):
        self.EstimatedState = EstState
        self.Map = Map # map of the world (i guess this is technically an estimate, but we'll run with it. just used for node positions, which we'll assume we know)
        self.ID = i # vehicle ID
        self.History = []
        self.InputHistory = []
        self.TemperatureHistory= []
        self.UpdateModelFlag = False
        self.TempBuckets = [265, 270, 280, 290, 300]
        self.Model = [0.04, 0.06]
        self.Model_L = np.zeros(len(self.TempBuckets))
        # later on, we will also add a data set that we use to update our estimate of mhat
        # later on, we will have to write add a pihat 
    
    
    def update(self, vehicle, Time, dT):
        # for now, our "observer" will just read the actual states
        self.EstimatedState = copy.copy(vehicle.State)
      
        # maybe for now the right thing to do is to just transfer the entire state and input history
        # the vehicle tracks the state and input history, and feeds it to the SA here in the "observer"
        self.DayHistory = copy.copy(vehicle.CurrentDayTrajectory)
        self.DayInputHistory = copy.copy(vehicle.CurrentDayInputTrajectory)
        self.DayTemperatureHistory = copy.copy(vehicle.Temperature) # temperature of this particular day
        
        
        if self.UpdateModelFlag:
            # also check that we have a meaningful amount of data
            if np.shape(np.reshape(self.DayInputHistory, ((2,-1))))[1] >= 10:
                self.updateModel()
            else:
                self.UpdateModelFlag = False
                
                
    def consolidate(self):
        # first, add today's thing to our saved route
        self.History.append(copy.copy(self.DayHistory))
        self.InputHistory.append(copy.copy(self.DayInputHistory))
        self.TemperatureHistory.append(copy.copy(self.DayTemperatureHistory))
        
        # now, do some stuff to send up again
        TimeCost = np.zeros((self.Map.N, self.Map.N))
        EnergyCost = np.zeros((self.Map.N, self.Map.N))
        
        x = []
        y = []
        soc = []
        time = []
        start_node = []
        end_node = []
        
        for i in range(1,len(self.DayHistory)):
            if self.DayHistory[i].ChargingState == False:
                x.append(copy.copy(self.DayHistory[i].Pos[0]))
                y.append(copy.copy(self.DayHistory[i].Pos[1]))
                soc.append(copy.copy(self.DayHistory[i-1].SOC)) # do to some unfortunate things, we have to do it like this
                time.append(copy.copy(self.DayHistory[i].Time))
                start_node.append(copy.copy(self.DayHistory[i].InTransition_StartNode))
                end_node.append(copy.copy(self.DayHistory[i].InTransition_TargetNode))
                
        # have to do some set stuff now i guess. 
        starts = [start_node[0]]
        for i in range(1,len(start_node)):
            if start_node[i] != start_node[i-1]:
                starts.append(start_node[i])
        ends = [end_node[0]]
        for i in range(1,len(end_node)):
            if end_node[i] != end_node[i-1]:
                ends.append(end_node[i])

        for i in range(len(starts)):
            indices = []
            for j in range(len(x)):
                if start_node[j] == starts[i] and end_node[j] == ends[i]:
                    indices.append(j)
            soc_lost = soc[indices[-1]] - soc[indices[0]]
            time_lost = time[indices[-1]] - time[indices[0]]           
            
            # now we set the corresponding indices in the TimeCost and EnergyCost matrices
            TimeCost[starts[i], ends[i]] = time_lost
            TimeCost[ends[i], starts[i]] = time_lost
            EnergyCost[starts[i], ends[i]] = soc_lost
            EnergyCost[ends[i], starts[i]] = soc_lost
        
        self.TimeCost = TimeCost
        self.EnergyCost = EnergyCost
        # okay cool. This gives us a dataset for transferring energy and time costs to the upper level planner. 
        return TimeCost, EnergyCost
                    

        
    def updateModel(self):
        
        # update the parameter estimate (EstimatedState.)
        ts = 0.1
        
        # get vector of velocities
        v_data = np.squeeze(self.InputHistory)[:,0]
        
        # get vector of SOCs
        soc_data = []
        for i in range(len(self.History)):
            soc_data.append(self.History[i].SOC)
        
        temp_data = self.TemperatureHistory
        
        # do some k-means clustering of this data? we want temperature to be the independent variable. 
        # so what do we really want? 
        
        # First, solve the general problem for alpha and constant:
        model = pyo.ConcreteModel()
        N = len(v_data) # 0.05s
        model.nIDX = pyo.Set(initialize=range(0, N)) 
        model.alpha = pyo.Var() 
        model.constant = pyo.Var() 
        model.d = pyo.Var()
        model.o = pyo.Var()
        model.cost = pyo.Objective(expr = sum(((soc_data[t+1] - soc_data[t] + ts*v_data[t]*model.alpha + ts*model.constant)**2) for t in model.nIDX if t < N), sense=pyo.minimize)
        solver = SolverFactory('ipopt')
        solver.solve(model,tee=False)   
        self.Model = [pyo.value(model.alpha), pyo.value(model.constant)]
        
        
        # we have a vector with a bunch of temperatures
        entries = spatial.KDTree(self.TempBuckets).query(temp_data.reshape((-1,1)))[1]
        
        # Then, split the data into buckets. 
        # For each bucket:
        for bucket in range(len(self.TempBuckets)):
            # add all the indices that are closest to that one
            indices = np.argwhere(bucket == entries)
            # reduce dataset
            v = v_data[indices]
            soc = soc_data[indices]
            temp = temp_data[indices]
            # solve for L
            model = pyo.ConcreteModel()
            N = len(v) # 0.05s
            model.nIDX = pyo.Set(initialize=range(0, N)) 
            model.L = pyo.Var() 
            model.cost = pyo.Objective(expr = sum(((soc_data[t+1] - soc_data[t] + model.L*(ts*v_data[t]*self.Model[0] + ts*self.Model[1]))**2) for t in model.nIDX if t < N), sense=pyo.minimize)
            solver = SolverFactory('ipopt')
            solver.solve(model,tee=False)   
            
            # store this somewhere now
            self.Model_L[bucket] = pyo.value(model.L)
            
        
        self.UpdateModelFlag = False
        
        print('Updated the SA model for Vehicle ', str(self.ID))
       
        


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

    def UpdateVehicle(self, Time: float, dT: float, Temp):
        # This function updates the true dynamics
        
        if self.ReturnedToDepot == True:
            return np.inf
        
        # There will be two kinds of updates:
            # 1. We need to be moving from one node to the other, in which case we recursively track the open-loop reference trajectory in Pi0
            # 2. We have reached the node and we need to be charging! This time step we can just do in batch I guess. 
        
        if self.State.InTransit == True:
            # TO DO: rewrite this in terms of the Pi1 goal! 
            if np.linalg.norm(self.State.Pos - self.Map.NodesPosition[self.State.InTransition_TargetNode,:]) < 0.5: # we are very close to the reference goal already:
                self.State.InTransit = False
                self.State.CurrentNode = self.State.InTransition_TargetNode
                self.State.NodesHistory.append(self.State.CurrentNode)
                self.State.Node = self.State.CurrentNode
                self.State.InTransition_StartNode = -1
                self.State.InTransition_TargetNode = -1
                
                # Check if the vehicle is in a charging station: If so, count how long we are charging for.
                if self.State.Node in self.Plan.CS.Nodes:
                    i_CS = np.argwhere(self.Plan.CS.Nodes == self.State.Node).tolist()[0][0]
                    if self.Plan.CS.ChargingTime[i_CS] > 0.0:
                        self.State.TgoForNextPhase = self.Plan.CS.ChargingTime[i_CS]
                        self.State.ChargingState = True
                        self.SOC_rate = self.Map.CS.ChargingRate
            else:
                print('Reference tracking Vehicle ' + str(self.ID))
                # if we are moving, then keep reference-tracking.                 
                # new way of calculating reference
                i = np.max((np.argmin(np.linalg.norm(np.reshape(self.State.Pos,(-1,1)) - self.DM.Pi0[1][0:2,:], axis=0)), self.index))
                # if i is 0, we are at the beginning - turn yourself for now. 
                if i == 0:
                    self.State.Theta = self.DM.Pi0[1][2,0]
                self.index = i
                
                x0 = np.hstack((self.State.Pos, self.State.Theta, self.State.SOC))
                N = 10 # arbitrarily set, but can fix this later
                
                ref = np.vstack((self.DM.Pi0[1][0, i : i+N], self.DM.Pi0[1][1, i : i+N]))
                input_sequence, ol_state_sequence = self.solve_cftoc(x0, ref, dT)
               
                v_apply = input_sequence[0,0]
                w_apply = input_sequence[1,0]
                
                self.State.Pos = [self.State.Pos[0] + dT * (v_apply*np.cos(self.State.Theta)), self.State.Pos[1] + dT * (v_apply*np.sin(self.State.Theta))]
                self.State.Theta = self.State.Theta + dT * w_apply
                #self.State.SOC = self.State.SOC - (0.0002*(Temp[-1] - 285)**4) * (dT *(self.Model[0] + np.random.normal(0, 0.005))*v_apply - dT * (self.Model[1] + np.random.normal(0, 0.005)))
                self.State.SOC = self.State.SOC - dT*self.Model[0]*v_apply # - dT*self.Model[1]
                self.State.TrackingCounter += 1
                
                self.Trajectory.append(copy.copy(self.State))
                self.CurrentDayTrajectory.append(copy.copy(self.State))
                self.InputTrajectory.append(np.vstack((v_apply, w_apply)))
                self.CurrentDayInputTrajectory.append(np.vstack((v_apply, w_apply)))
                self.Temperature.append(np.squeeze(Temp[-1]))
                
            
            
        if self.State.ChargingState == True:
            print('Charging! Vehicle ' + str(self.ID))
            if self.State.TgoForNextPhase > 1e-8:
                # else we are charging and the only state that updates is the charge
                self.State.TgoForNextPhase -= dT # Time Integration. # how much time is remaining after this time step until we reach the node
                self.State.SOC += dT*self.SOC_rate # how much SOC do we have left at this point
                # The other states will stay the same.
                self.State.SOC = max(self.State.SOC, 100.0)
            else:
                self.State.ChargingState = False
                # this should clear us up for the next UpdateSelfPlanner run.
                # check that after we enter this, we re-plan
        
        self.State.Time += dT

        return np.inf
    
    def ConsolidateRouteInfo(self):
        TimeCost, EnergyCost = self.SA.consolidate()
        
        return TimeCost, EnergyCost

    def UpdateSA(self, Time: float, dT: float):
        if self.ReturnedToDepot == True:
            5
        else:
            self.SA.update(self, Time, dT)

            # send new info to the DM
            self.DM.EstimatedState = copy.deepcopy(self.SA.EstimatedState) # this will transfer the position info, soc info
            self.DM.Model = copy.deepcopy(self.SA.Model) # this transfers the model 

    def UpdateDM(self, Time: float, dT: float):
        # this has to be fancier, only sometimes we do the update of the DM. only if we're at a special place.
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
            
            # sad, but have to do it
            # think of it as steering before we begin the drive! 
            self.State.Theta = self.DM.Pi0[1][2][0]
            self.index = 0
            
            # self.State.InTransition_StartNode = self.State.Node
            # self.State.InTransition_TargetNode = NextNode
            # self.State.CurrentNode = -1 # does this mean we are between nodes?

    
    def solve_cftoc(self, x0, ref, Ts): 
        x0 = np.squeeze(x0)
        model = pyo.ConcreteModel()
        N = np.shape(ref)[1]
        
        model.nIDX = pyo.Set(initialize=range(0, N)) 
        model.nUDX = pyo.Set(initialize=range(0, N-1))
        
        model.x = pyo.Var(model.nIDX) 
        model.y = pyo.Var(model.nIDX) 
        model.theta = pyo.Var(model.nIDX)
        model.soc = pyo.Var(model.nIDX) 
        
        soc_alpha = self.Model[0]
        soc_const = self.Model[1]
        
        model.v = pyo.Var(model.nIDX) 
        model.w = pyo.Var(model.nIDX)
       
        # 1. initialization
        model.constraint01 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.x[t] == x0[0] if t < 1 else pyo.Constraint.Skip )
        model.constraint02 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.y[t] == x0[1] if t < 1 else pyo.Constraint.Skip )
        model.constraint03 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.theta[t] == x0[2] if t < 1 else pyo.Constraint.Skip )
        model.constraint04 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.soc[t] == x0[3] if t < 1 else pyo.Constraint.Skip )
    
    
        # 2. dynamics
        model.constraint11 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.x[t+1] == model.x[t] + Ts * (model.v[t]*pyo.cos(model.theta[t]))
                                       if t < N-1 else pyo.Constraint.Skip)
        model.constraint12 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.y[t+1] == model.y[t] + Ts * (model.v[t]*pyo.sin(model.theta[t]))
                                       if t < N-1 else pyo.Constraint.Skip)
        model.constraint13 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.theta[t+1] == model.theta[t] + Ts * model.w[t]
                                       if t < N-1 else pyo.Constraint.Skip)
        model.constraint14 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.soc[t+1] == model.soc[t] - Ts * soc_alpha*model.v[t]  #- Ts * soc_const
                                       if t < N-1 else pyo.Constraint.Skip)
                                        
        
        # 5. velocity limits
        model.constraint41 = pyo.Constraint(model.nUDX, rule=lambda model, t: model.v[t] <= 10.0)
        model.constraint42 = pyo.Constraint(model.nUDX, rule=lambda model, t: model.v[t] >= 0.0)
        # theta limits
        model.constraint411 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.theta[t] <= 180)
        model.constraint421 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.theta[t] >= -180)
        
        
        
        # 5b. velocity rate limits
        model.constraint43 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.v[t+1] - model.v[t] <= 5.0 * Ts
                                            if t < N-2 else pyo.Constraint.Skip)
        model.constraint44 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.v[t+1] - model.v[t] >= - 5.0 * Ts
                                            if t < N-2 else pyo.Constraint.Skip)
        
        # 6. steering limits
        model.constraint61 = pyo.Constraint(model.nUDX, rule=lambda model, t: model.w[t] <= 40)
        model.constraint62 = pyo.Constraint(model.nUDX, rule=lambda model, t: model.w[t] >= -40)
        
        # 6b. steering rate limits
        model.constraint63 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.w[t+1] - model.w[t] <= 100.0 * Ts
                                            if t < N-2 else pyo.Constraint.Skip)
        model.constraint64 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.w[t+1] - model.w[t] >= -100.0 * Ts
                                            if t < N-2 else pyo.Constraint.Skip)

        
        # cost: minimize squared norm to lref
        model.cost = pyo.Objective(expr = sum(( (model.x[t] - ref[0,t])**2 + (model.y[t] - ref[1,t])**2 + 0.05*model.v[t]**2 + 0.05*model.w[t]**2) for t in model.nIDX), sense=pyo.minimize)
        
        # solve
        solver = SolverFactory('ipopt')
        solver.options["print_level"] = 5
        results = solver.solve(model,tee=False)   
        
        x = [pyo.value(model.x[0])]
        y = [pyo.value(model.y[0])]
        theta = [pyo.value(model.theta[0])]
        soc = [pyo.value(model.soc[0])]
        v = [pyo.value(model.v[0])]
        w = [pyo.value(model.w[0])]
        
        for t in range(N):
            if t < N-1:
                x.append(pyo.value(model.x[t+1]))
                y.append(pyo.value(model.y[t+1]))
                theta.append(pyo.value(model.theta[t+1]))
                soc.append(pyo.value(model.soc[t+1]))
            if t < N-2:
                v.append(pyo.value(model.v[t+1]))
                w.append(pyo.value(model.w[t+1]))
        
    
        input_sequence = np.vstack((v,w))
        ol_state_sequence = np.vstack((x, y, theta, soc))
        
        return input_sequence, ol_state_sequence
    
    