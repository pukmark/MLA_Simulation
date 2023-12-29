#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:24:22 2023

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


class SA():
    def __init__(self, EstState, Map, i):
        self.EstimatedState = EstState
        self.Map = Map
        self.ID = i # vehicle ID
        self.History = []
        self.InputHistory = []
        self.TemperatureHistory= []
        self.UpdateModelFlag = False
        self.TempBuckets = [265, 270, 280, 290, 300] # unnecessary for now 
        self.Model = [0.04, 0.06] # unnecessary for now 
        self.Model_L = np.zeros(len(self.TempBuckets)) # unnecessary for now 
    
    
    def update(self, vehicle, Time, dT):
        # for now, our "observer" will just read the actual states
        self.EstimatedState = copy.copy(vehicle.State)
      
        # the vehicle tracks the state and input history, and feeds it to the SA here in the "observer"
        self.DayHistory = copy.copy(vehicle.CurrentDayTrajectory)
        self.DayInputHistory = copy.copy(vehicle.CurrentDayInputTrajectory)
        self.DayTemperatureHistory = copy.copy(vehicle.Temperature) # temperature of this particular day
        
        if self.UpdateModelFlag:
            if np.shape(np.reshape(self.DayInputHistory, ((2,-1))))[1] >= 10:
                self.updateModel()
            else:
                self.UpdateModelFlag = False
                
                
    def consolidate(self, reach_sets):
        self.History.append(copy.copy(self.DayHistory))
        self.InputHistory.append(copy.copy(self.DayInputHistory))
        self.TemperatureHistory.append(copy.copy(self.DayTemperatureHistory))
        
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
                soc.append(copy.copy(self.DayHistory[i-1].SOC))
                time.append(copy.copy(self.DayHistory[i].Time))
                start_node.append(copy.copy(self.DayHistory[i].InTransition_StartNode))
                end_node.append(copy.copy(self.DayHistory[i].InTransition_TargetNode))
                
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
        
            TimeCost[starts[i], ends[i]] = time_lost
            TimeCost[ends[i], starts[i]] = time_lost
            EnergyCost[starts[i], ends[i]] = soc_lost
            EnergyCost[ends[i], starts[i]] = soc_lost

        self.TimeCost = TimeCost
        self.EnergyCost = EnergyCost
        
        ReachSetInfo = []
        for i in range(len(reach_sets)):
            ReachSetInfo.append([[starts[i]], [ends[i]], [reach_sets[i]]])
        self.ReachSetInfo = ReachSetInfo
        
        return TimeCost, EnergyCost
                    

        
    def updateModel(self):
        # This is not used at the moment, but I wanted to leave it in here in case we want it later
        
        ts = 0.1
        
        v_data = np.squeeze(self.InputHistory)[:,0]
        soc_data = []
        for i in range(len(self.History)):
            soc_data.append(self.History[i].SOC)
        
        temp_data = self.TemperatureHistory
        
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

        entries = spatial.KDTree(self.TempBuckets).query(temp_data.reshape((-1,1)))[1]

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
            
            self.Model_L[bucket] = pyo.value(model.L)
        
        self.UpdateModelFlag = False
        
        print('Updated the SA model for Vehicle ', str(self.ID))
       