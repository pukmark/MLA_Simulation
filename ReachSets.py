#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:43:47 2023

@author: vallon2
"""

import polytope as pt
import matplotlib.pyplot as plt
import numpy as np

def minkowski_sum(X, Y):

    # Minkowski sum between two polytopes based on
    # vertex enumeration. So, it's not fast for the
    # high dimensional polytopes with lots of vertices.
    V_sum = []
    if isinstance(X, pt.Polytope):
        V1 = pt.extreme(X)
    else:
        # assuming vertices are in (N x d) shape. N # of vertices, d dimension
        V1 = X

    if isinstance(Y, pt.Polytope):
        V2 = pt.extreme(Y)
    else:
        V2 = Y

    for i in range(V1.shape[0]):
        for j in range(V2.shape[0]):
            V_sum.append(V1[i,:] + V2[j,:])
    return pt.qhull(np.asarray(V_sum))


def precursor(Xset, A, Uset=pt.Polytope(), B=np.array([])):
        if not B.any():
            return pt.Polytope(Xset.A @ A, Xset.b)
        else:
            tmp  = minkowski_sum( Xset, pt.extreme(Uset) @ -B.T )
        return pt.Polytope(tmp.A @ A, tmp.b)
    
    
def successor(Xset, A, Uset=pt.Polytope(), B=np.array([]), Wset=pt.Polytope()):

    # Xset, Uset shoud be polytope

    # autonomous case: xdot = Ax
    if not B.any():
        if not pt.is_empty(Wset):
            return minkowski_sum(pt.qhull(pt.extreme(Xset) @ A.T), Wset)
        else:
            return pt.qhull(pt.extreme(Xset) @ A.T)

    # controlled case: xdot = Ax+Bu
    if not pt.is_empty(Wset):
        return minkowski_sum(minkowski_sum(pt.extreme(Xset) @ A.T,
                                           pt.extreme(Uset) @ B.T), Wset)
    else:
        return minkowski_sum(pt.extreme(Xset) @ A.T,
                     pt.extreme(Uset) @ B.T)
    

def KReachSet(X, A, U, B, Xf, K):
    C = {}
    C[0] = Xf
    PreS = precursor(Xf, A, U, B)
    print('Beginning reachability')
    for j in range(1,K+1):
        print(K-j)
        C[j]= PreS.intersect(X)
        PreS = precursor(C[j], A, U, B)
    print('Finished reachability')
    return C


def ForwardReachSet(x0, A, B, X, U, Kmax):
    C = {}
    
    eps = 0.001
    # need to turn x0 into a Polytope via eps
    X0 = pt.Polytope(np.array([[1.0, 0, 0],
                              [-1.0, 0, 0],
                              [0, 1.0, 0],
                              [0, -1.0, 0],
                              [0, 0, 1.0],
                              [0, 0, -1.0]]),
                    np.array([[x0[0] + eps],
                              [-x0[0] + eps],
                              [x0[1] + eps],
                              [-x0[1] + eps],
                              [x0[2] + eps],
                              [-x0[2] + eps]]))
    
    
    C[0] = X0
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # C[0].project([1,2]).plot(ax)
    # ax.autoscale_view()
    # plt.show()
    
    for j in range(1, Kmax):
        print(Kmax-j)
        next_set = successor(C[j-1], A, U, B)
        C[j] = next_set.intersect(X)
        
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # C[j].project([1,2]).plot(ax)
        # ax.autoscale_view()
        # plt.show()
        
    return C
    
    