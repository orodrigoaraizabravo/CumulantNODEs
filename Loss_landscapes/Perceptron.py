# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 08:58:41 2023

@author: Rodrigo Araiza Bravo, Jorge Garcia Ponce
"""
#You may have to change the path before calling cnodes
import torch 
from CNODES import EDNN

def Perceptron_ansatz(N, h=0.5): 
    Gs = [['z', 'z', i,N-1] for i in range(N-2)]
    Fs = [['x', i] for i in range(N)]
    Fs += [['z', i] for i in range(N)]
    Hout = [[1.0, 'z', 'z', i,i+1] for i in range(N-1)]
    Hout+= [[h, 'x', i] for i in range(N)]
    return Gs, Fs, Hout

N=5
times=torch.linspace(0,1,20)

Gs, Fs, Hout= Perceptron_ansatz(N)
perceptron=EDNN(N, Gs, Fs, Hout, constraint=False)
loss = perceptron.loss(times)



