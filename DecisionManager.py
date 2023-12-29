#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:22:47 2023

@author: vallon2
"""

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


class DM():
    def __init__(self, Pi1, EstState, Map, i):
        self.Pi1 = Pi1 
        self.EstimatedState = EstState
        self.Map = Map 
        self.ID = i # vehicle ID
        self.Model = [0.04, 0.06]
        self.ReachabilityMarker = 0.3 
        
        
    def update(self, vehicle, Time, dT):
        # This function updates whenever we have reached a new node that is not a charging node. Creates a trajectory to drive to the next node.
        
        if vehicle.ReturnedToDepot == True:
            # don't update anymore
            5
        
        elif self.EstimatedState.InTransit == False and self.EstimatedState.ChargingState == False:
            print('Updating Pi0 for Vehicle ' + str(self.ID))
            CurPlan_i = np.argwhere(self.Pi1.NodesTrajectory == self.EstimatedState.Node).tolist()[0][0] # at what index of the plan are we?
            Cur_Pos = self.EstimatedState.Pos
            
            NextNode = self.Pi1.NodesTrajectory[CurPlan_i+1] # what node are we going to? 
            Goal_Pos = self.Map.NodesPosition[NextNode,:]
            
            self.Travel_Distance = np.sqrt((Goal_Pos[0] - Cur_Pos[0])**2 + (Goal_Pos[1] - Cur_Pos[1])**2)
            
            Goal_Time = self.Pi1.TimePlan[CurPlan_i+1] - Time # how much time we have to get to the next node
            Goal_Charge = self.Pi1.SOCPlan[CurPlan_i+1] # how much charge do we need to have left at the next node?
            
            x0 = np.hstack((Cur_Pos, self.EstimatedState.Theta, self.EstimatedState.SOC, 0))
            
            # now we add the solve_batch function that calculates the optimal input sequence to apply to get from current node to next node
            input_sequence, ol_state_sequence = self.solve_batch(x0, Goal_Charge, Goal_Time, Goal_Pos, Time, dT)
        
            # append the goal location to this
            Goal_State = np.array([[Goal_Pos[0]], [Goal_Pos[1]], [ol_state_sequence[2,-1]], [Goal_Charge], [0], [ol_state_sequence[5,-1] + dT]])
            
            self.Pi0 = [np.hstack((input_sequence, np.array([[0],[0]]))), np.hstack((ol_state_sequence, Goal_State))]
            
            self.lmpc_theta = self.Pi0[1][2]
            self.EstimatedState.Theta = self.Pi0[1][2][0]
            self.EstimatedState.TrackingCounter = 0
            self.EstimatedState.InTransit = True
            self.EstimatedState.InTransition_StartNode = self.EstimatedState.Node
            self.EstimatedState.InTransition_TargetNode = NextNode
            self.EstimatedState.CurrentNode = -1 
            self.NodeStartTime = Time # Time at which we began journeying to the next node
            self.NodeStartSOC = self.EstimatedState.SOC
            
    def forward_reach_sets(self, x0, Kmax, dT):
        vmax = 10
        
        # x = [d, v, soc]
        X = pt.Polytope(np.array([[1.0, 0, 0],
                                  [-1.0, 0, 0],
                                  [0, 1.0, 0],
                                  [0, -1.0, 0],
                                  [0, 0, 1.0],
                                  [0, 0, -1.0]]),
                        np.array([[100],
                                  [0],
                                  [vmax],
                                  [0],
                                  [100],
                                  [0]]))       
        A = np.array([[1.0, dT, 0],
                      [0, 1.0, 0],
                      [0, -self.Model[0]*dT, 1.0]])

        U = pt.Polytope(np.array([1.0, -1.0]).reshape(2,1),
                        np.array([3.0, 3.0]).reshape(2,1))
        B = np.array([[0], [dT], [0]])

        set_list = ForwardReachSet(x0, A, B, X, U, Kmax)
        return set_list
          
    def solve_batch(self, x0, goal_charge, goal_time, goal_loc, Time, Ts):
            
            soc_alpha = self.Model[0]
            
            reference_theta = np.arctan2(goal_loc[1] - x0[1], goal_loc[0] - x0[0])
            
            model = pyo.ConcreteModel()
            N = int(10*goal_time)
            model.nIDX = pyo.Set(initialize=range(0, N+1)) 
            
            model.x = pyo.Var(model.nIDX) 
            model.y = pyo.Var(model.nIDX) 
            model.theta = pyo.Var(model.nIDX)
            model.soc = pyo.Var(model.nIDX) 
            model.v = pyo.Var(model.nIDX)
            
            model.a = pyo.Var(model.nIDX) 
            model.w = pyo.Var(model.nIDX)
            
           
            # 1. initialization
            model.constraint01 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.x[t] == x0[0] if t < 1 else pyo.Constraint.Skip )
            model.constraint02 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.y[t] == x0[1] if t < 1 else pyo.Constraint.Skip )
            model.constraint04 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.soc[t] == x0[3] if t < 1 else pyo.Constraint.Skip )
            model.constraint05 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.v[t] == x0[4] if t < 1 else pyo.Constraint.Skip )
        
        
            # 2. dynamics - NOTE THAT PYOMO HAS THETA IN DEGREES!!!!!!!!!!!
            model.constraint11 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.x[t+1] == model.x[t] + Ts * (model.v[t]*pyo.cos(model.theta[t])) # pyomo is in degrees!!!!!!!!!
                                           if t < N else pyo.Constraint.Skip)
            model.constraint12 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.y[t+1] == model.y[t] + Ts * (model.v[t]*pyo.sin(model.theta[t]))
                                           if t < N else pyo.Constraint.Skip)
            model.constraint13 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.theta[t+1] == model.theta[t] + Ts * model.w[t]
                                           if t < N else pyo.Constraint.Skip)
            model.constraint14 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.soc[t+1] == model.soc[t] - Ts * soc_alpha*model.v[t] # - Ts * soc_const
                                           if t < N else pyo.Constraint.Skip)
            model.constraint15 = pyo.Constraint(model.nIDX, rule=lambda model, t: model.v[t+1] == model.v[t] + Ts * model.a[t]
                                           if t < N else pyo.Constraint.Skip)
            
            # 4. SOC charge constraint
            model.constraint31 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.soc[t] >= goal_charge)
                                            
            
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

            
            # terminal constraint, system must come to a stop
            model.constraint71 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.v[t] == 0.0 if t >= N else pyo.Constraint.Skip)
            model.constraint72 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.x[t] == goal_loc[0] if t >= N else pyo.Constraint.Skip)
            model.constraint73 = pyo.Constraint(model.nIDX, rule = lambda model, t: model.y[t] == goal_loc[1] if t >= N else pyo.Constraint.Skip)
            
            model.cost = pyo.Objective(expr = sum((goal_loc[0] - model.x[t])**2 + (goal_loc[1] - model.y[t])**2 + 10*(reference_theta - model.theta[t])**2 + 0.5*model.a[t]**2 + 0.5*model.w[t]**2 for t in model.nIDX if t <= N), sense=pyo.minimize)

            
            # solve
            solver = SolverFactory('ipopt')
            solver.options["print_level"] = 5
            results = solver.solve(model,tee=False)   
            
            if (results.solver.termination_condition == pyo.TerminationCondition.infeasible):
                print('solver was infeasible!')
            
            # extract inputs
            x = [pyo.value(model.x[0])]
            y = [pyo.value(model.y[0])]
            theta = [pyo.value(model.theta[0])]
            soc = [pyo.value(model.soc[0])]
            v = [pyo.value(model.v[0])]
            a = [pyo.value(model.a[0])]
            w = [pyo.value(model.w[0])]
            time_vec = [Time]
            
            for t in range(N):
                if t < N:
                    x.append(pyo.value(model.x[t+1]))
                    y.append(pyo.value(model.y[t+1]))
                    theta.append(pyo.value(model.theta[t+1]))
                    soc.append(pyo.value(model.soc[t+1]))
                    v.append(pyo.value(model.v[t+1]))
                    time_vec.append(time_vec[-1] + Ts)
                if t < N-1:
                    a.append(pyo.value(model.a[t+1]))
                    w.append(pyo.value(model.w[t+1]))
            
            
            plt.plot(x,y,'r')
            plt.scatter(goal_loc[0], goal_loc[1])
        
            # calculate distance
            dist = np.sqrt((goal_loc[0] - x)**2 + (goal_loc[1] - y)**2)
            if min(abs(dist)) < 0.001:
                stop_index = np.where(abs(dist)<0.001)
                stop_index = stop_index[0][0]
            else:
                print('did not get within 0.001m of goal')
                stop_index = -2
            
            input_sequence = np.vstack((a[:stop_index],w[:stop_index]))
            ol_state_sequence = np.vstack((x[:stop_index+1], y[:stop_index+1], theta[:stop_index+1], soc[:stop_index+1], v[:stop_index+1], time_vec[:stop_index+1]))

            return input_sequence, ol_state_sequence