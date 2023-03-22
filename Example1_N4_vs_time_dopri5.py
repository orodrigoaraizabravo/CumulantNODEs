# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:25:16 2023

@author: Rodrigo Araiza Bravo
"""
import torch 
from CNODEs import CumulantNN
from julia import Main
from time import time
import sys 

def Example1(N, platform = 'Python', h=0.5): 
    s=0
    if platform not in ['Python', 'python', 'p']: s+=1
    Gs = [['x', 'x', i,i+1] for i in range(s,N-1+s)]
    Fs = [['z', i] for i in range(s,N+s)]
    Hout = [[1.0, 'x', 'x', i,i+1] for i in range(s,N-1+s)]
    Hout+= [[h, 'z', i] for i in range(s,N-1+s)]
    return Gs, Fs, Hout

def compare_grads(m1,m2): 
    g1=[p.grad for p in m1.parameters()]
    g2=[p.grad for p in m2.parameters()]
    g = [torch.sum(abs(g1[i]-g2[i])/(g2[i]+1e-4))\
         for i in range(len(g1))]
    return sum(g).item()
n_basis=5

Ns = range(3,10)
Times = [[],[]]
orders=[2,3]
order = orders[int(sys.argv[1])]

method='dropi5'
ts =torch.linspace(0,1.0,11)
Nreps=10

for N in Ns:
    Gs_j, Fs_j, H_j = Example1(N,platform='Julia')
    cvqe = CumulantNN(N, Gs_j, Fs_j, order, H_j, Main, recall=True, method=method)
    cvqe.exps_to_vect([[0.0,0.0,1.] for i in range(N)])

    t0=time()
    for i in range(Nreps):
        loss_cvqe = cvqe.loss(ts)
    t1=time()
    print((t1-t0)/Nreps)
    Times.append((t1-t0)/Nreps)

import pickle
dic= {'Ns': Ns, 'order': order, 'Times': Times}
with open('Times_N4_Example1_dropi5_order'+str(order)+'.pt', 'wb') as ofile:
    pickle.dump(dic,ofile)