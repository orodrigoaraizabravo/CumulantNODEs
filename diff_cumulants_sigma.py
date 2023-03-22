# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:41:45 2023

@author: HP
"""

import torch 
from torch import nn, pi, optim
from torchdiffeq import odeint, odeint_adjoint
from torch.special import legendre_polynomial_p as lp
import numpy as np

def call_operators(N, Graph, Fields, Main): 
    Main.eval("""σ(i, axis) = Sigma(h,Symbol(:σ_,i), axis, i)""")
    Main.eval("σx(i) = σ(i, 1)")
    Main.eval("σy(i) = σ(i, 2)")
    Main.eval("σz(i) = σ(i, 3)")
    Jops, hops = [], []
    for c in Graph: 
        oplen = len(c)//2
        Jops.append(''.join(["σ"+c[i]+"("+str(c[oplen+i])+")*" for i in range(oplen)])[:-1])
    for f in Fields: hops.append('σ'+f[0]+"("+str(f[1])+")")
    return Jops, hops

def call_constants(N,Graph,Fields,Main):
    '''Graph has entries (A,B,i,j) where A, B are directions, and i,j are sites
    Fields has entries (A, i) where A is a direction and i is a site. 
    A, B are strings, and i,j are integers <=N'''
    Main.eval("""J(A,B,i,j) = cnumber(Symbol("J^{$A$B}_{$i$j}"))""")
    Main.eval("""Δ(A,i) = cnumber(Symbol("h^{$A}_{$i}"))""")
    Main.eval("""M(A,B,i,j) = cnumber(Symbol("M^{$A$B}_{$i$j}"))""")
    Main.eval("""t(A,i) = cnumber(Symbol("t^{$A}_{$i}"))""")
    Js, hs = [], [] 
    
    for c in Graph: Js.append("J('"+c[0]+"','"+c[1]+"',"+str(c[2])+","+str(c[3])+")")
    for f in Fields: hs.append("Δ('"+f[0]+"',"+str(f[1])+")")

    return Js, hs

def call_Hout(Hout): 
    ops = []
    for c in Hout: 
        #c = [coeff, A, i] or [coeff, A, B,i,j] or [coeff, A, B, C, i, j, k]....
        cop = c[1:]
        oplen = len(cop)//2
        ops.append(''.join(["σ"+cop[i]+"("+str(cop[oplen+i])+")*" for i in range(oplen)])[:-1])
    return ops

def call_Main(N, Graph, Fields, Hout, order,Main):
    Main.using("QuantumCumulants")
    Main.using("ModelingToolkit")
    Main.using("Symbolics")
    Main.eval("N="+str(N))
    Main.eval("order="+str(order))
    Main.eval("h = ⊗([SpinSpace(Symbol(:Spin,i)) for i=1:N]...)")
    Jops, hops = call_operators(N,Graph, Fields,Main)
    Outops = call_Hout(Hout)

    
    Js, hs = call_constants(N, Graph, Fields,Main)
    
    sH='H='
    for i in range(len(Graph)): sH+=Js[i]+'*'+Jops[i]+'+'
    for i in range(len(Fields)): sH+=hs[i]+'*'+hops[i]+'+'
    
    sHout = 'Hout='
    sops = 'ops=['
    for i in range(len(Hout)): 
        sHout+=str(Hout[i][0])+'*'+Outops[i]+'+'
        sops +=Outops[i]+','
    sops=sops[:-1]+']'
    Main.eval(sH[:-1])
    Main.eval(sHout[:-1])
    Main.eval(sops)
    Main.eval("""eqs = meanfield(ops, H; order=order)""")
    Main.eval("""eqs = complete(eqs, order=order)""")
    Main.eval("""sys=ODESystem(eqs; name=:cumeqs)""")
    Main.eval("""avgout = cumulant_expansion(average(Hout),order)""")
    return Jops, hops

class CumulantNN(nn.Module):
    def __init__(self, N, Graph, Fields, order, Hout, Main, recall=False, T=1.0, n_basis = 5, basis='Fourier', x0=None, adjoint=False, Js=None, vs=None,method='euler'):
        super(CumulantNN, self).__init__()
        self.Main=Main
        if recall:
            _=call_Main(N, Graph, Fields,Hout, order, Main)
        self.order=order
        self.n_qubits=N
        self.n_basis= n_basis
        self.basis = basis
        self.method=method
        self.T=T
        self.varmap = {str(self.Main.sys.states[i])[15:-1]:"x"+"["+str(i)+"]"\
                       for i in range(len(self.Main.sys.states))}
        self.get_varmap_out()
        #self.complete_varmap_out()
        #self.set_output()
        
        self.parmap = {str(self.Main.sys.ps[i])[15:-1]:'' for i in range(len(self.Main.sys.ps))}
        
        cntJ=0
        cnth=0
        for key in self.parmap.keys(): 
            if 'J' in key: 
                self.parmap[key]='self.J('+str(cntJ)+')'
                cntJ+=1
            else: 
                self.parmap[key]='self.p('+str(cnth)+',t)'
                cnth+=1
        
        self.Graph   = Graph
        self.Fields  = Fields
        
        if Js is None: self.Js  = nn.Parameter(torch.rand(len(self.Graph)))
        else: self.Js = nn.Parameter(Js)
        
        if vs is None: self.vs =  nn.Parameter(torch.rand([len(self.Fields), n_basis]))
        else: self.vs = nn.Parameter(vs)
        
        if x0 is None: self.x0 = torch.zeros(len(self.Main.sys.states))
        else: self.x0=x0
        self.odeint = odeint_adjoint if adjoint else odeint
        
        if basis not in ['Legendre', 'Fourier', 'poly']:
            raise ValueError('Basis ' + basis + ' not supported')
        
        self.get_equations()
        self.get_string_out()
        
    def get_varmap_out(self):
        self.varmap_out = {}
        for key in self.varmap:
            newkey = key.replace("var","").replace("(t)","")[1:-1]
            self.varmap_out[newkey] = self.varmap[key]
        return 
    
    def J(self,i):
        return (2*torch.sigmoid(self.Js[i])-1)
    
    def p(self,i,t):
        """function p_i(t) for H_i
        """
        if self.basis == 'Fourier':
            u = torch.dot(self.vs[i,:],torch.cos(2*pi*(t/self.T)*torch.arange(self.n_basis)))
        elif self.basis == 'poly':
            u = torch.dot(self.vs[i,:],(t/self.T)**torch.arange(self.n_basis))
        elif self.basis == 'Legendre':
            u = torch.dot(self.vs[i,:],lp(t/self.T, torch.arange(self.n_basis)))
            
        return (2*torch.sigmoid(u)-1)
    
    
    def parse_string(self,s):
        for x in [" "+str(integer) for integer in range(1,10)]: s=s.replace(x,x+'*').replace(x[1:]+'var', x[1:]+'*var')
        s=s.replace("im","").replace(" ","").replace(")J",")*J")
        for key in self.varmap: 
            s=s.replace(key, self.varmap[key])
        for key in self.parmap:
            s=s.replace(key, self.parmap[key])
        return s.replace("⟨","").replace("⟩","").replace("^","**").replace("*)",")").replace(')self',')*self')
    
    def parse_string_out(self):
        s=str(self.Main.avgout)[15:-1]
        for x in [" "+str(integer) for integer in range(1,10)]: s=s.replace(x,x+'.*').replace(x[1:]+'var', x[1:]+'*var').replace(x[1:]+'⟨', x[1:]+'*⟨')
        for key in self.varmap_out: 
            s=s.replace(key, self.varmap_out[key])
        return s.replace("⟨","").replace("⟩","").replace("^","**").replace("*)",")").replace(')self',')*self')
    
    def _equation(self, i):
        s=str(self.Main.sys.eqs[i].rhs)[15:-1]
        return self.parse_string(s)
    
    def get_equations(self):
        self.equations= [] 
        for i in range(len(self.varmap)):
            self.equations.append(eval('lambda self, t, x: '+self._equation(i)))
        return 
    
    def get_string_out(self):
        self.s = eval('lambda x: '+self.parse_string_out())
        return 
    
    def func(self,t,x):
        dx_dt = torch.zeros(x.size()[0])
        #ps = [self.p(i,t) for i in range(len(self.Fields))]
        for i in range(x.size()[0]): dx_dt[i] = self.equations[i](self,t,x) #function of time
        return dx_dt
    
    def forward(self,times):
        return self.odeint(self.func,self.x0,times,method=self.method)[-1]
    
    def loss(self, times):
        x = self.forward(times)
        return self.s(x)
    
    def exps_to_vect(self,single_exp):
        #single_exp is of shape [ [sx,sy,sz] for i in range(N) ]
        self.x0 = torch.zeros(len(self.varmap))
        axes = {'x':0, 'y':1, 'z': 2}
        for i in range(len(self.varmap)):
            p=1.0
            s = list(self.varmap_out.keys())[i][1:-1]
            for l in range(s.count('σ')):
                indexes= []
                for w in ['x','y','z']:
                    try:
                        s.index(w)
                        indexes.append(s.index(w))
                    except ValueError: indexes.append(len(s)+1)
                aind=min(indexes)
                a, site = axes[s[aind]], int(s[s.index("_")+1:aind])-1
                s = s[aind+1:]
                p*=single_exp[site][a]
                if p==0.0: break
            self.x0[i]=p
        return
    
    def train_me(self, Epochs, times,optimizer):
        losses=[]
        
        for e in range(Epochs):
            #x = self.forward(times)
            l = self.loss(times)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(l.detach())
            losses.append(l.detach())
            
        return losses
    
    
class EDNN(nn.Module):
    def __init__(self, N, Graph, Fields, Hout, T=1.0, n_basis = 5, basis='Fourier', x0=None, Js=None, vs=None, method='euler'):
        super(EDNN,self).__init__()
        self.N=N
        self.Gs=Graph
        self.Fs=Fields
        self.Hout=Hout
        self.T=T
        self.n_basis=n_basis
        self.basis=basis
        self.method=method
        
        if Js is None: self.Js  = nn.Parameter(torch.rand(len(self.Graph)))
        else: self.Js = nn.Parameter(Js)
        
        if vs is None: self.vs =  nn.Parameter(torch.rand([len(self.Fields), n_basis]))
        else: self.vs = nn.Parameter(vs)
        
        if x0 is None: 
            self.x0 = torch.zeros(2**self.N, dtype=torch.cfloat)
            self.x0[0] = 1.0
        else: self.x0=x0
        
        self.sx = torch.tensor([[0j,1.],[1.,0j]])
        self.sy = torch.tensor([[0.,-1j],[1j,0.]])
        self.sz = torch.tensor([[1.,0j],[0,-1.]])
        self.I2 = torch.tensor([[1.0, 0j],[0,1.]])
        
        self.measurment_H()
        self._evolving_H()
        
    def add_op(self, ops, sites, full=False): 
        if not full: 
            l= [self.I2 for i in range(self.N)] 
            for s in range(len(sites)): l[sites[s]]=ops[s]
        else: l = ops
        m=l[0]
        for s in range(1,self.N): m=torch.kron(m,l[s])
        return m
    
    def get_op(self, s):
        if s in ['x','X']: return self.sx
        elif s in ['y','Y']: return self.sy
        elif s in ['z','Z']: return self.sz
        
    def measurment_H(self): 
        self.H = 0.0
        for term in self.Hout:
            l = (len(term)-1)//2
            ops = [self.get_op(term[i]) for i in range(1,l+1)]
            sites = term[l+1:]
            self.H+=term[0]*self.add_op(ops, sites)
        return
        
    def _evolving_H(self): 
        self.Gops=[]
        for term in self.Gs: 
            ops = [self.get_op(term[i]) for i in range(len(term)//2)]
            sites = term[len(term)//2:]
            self.Gops.append(self.add_op(ops, sites))
        self.Fops=[]
        for term in self.Fs: 
            op, site = [self.get_op(term[0])], term[1:]
            self.Fops.append(self.add_op(op, site))
        return
            
    def evolving_H(self, t):
        h = self.Gops[0]*self.generate_p_constant(0)(t)
        for i in range(1,len(self.Gops)): h+=self.Gops[i]*self.generate_p_constant(i)(t)
        for i in range(0,len(self.Fops)): h+=self.Fops[i]*self.generate_p(i)(t)
        return h
    
    def generate_p_constant(self, i):
        """Function for couplings (constant)"""
        def p(t):
            return (2*torch.sigmoid(self.Js[i])-1)
        return p
    
    def generate_p(self,i):
        """Generate the function p_i(t) for H_i
        Args:
            i: index of the H_i.
            coefficient of shape [nqubits, 2] 
        Returns:
            p: function p_i(t).
        """
        def p(t):
            if self.basis == 'Fourier':
                u = torch.dot(self.vs[i,:],torch.cos(2*pi*(t/self.T)*torch.arange(self.n_basis)))
            elif self.basis == 'poly':
                u = torch.dot(self.vs[i,:],(t/self.T)**torch.arange(self.n_basis))
            elif self.basis == 'Legendre':
                u = torch.dot(self.vs[i,:],lp(t/self.T, torch.arange(self.n_basis)))
                
            return (2*torch.sigmoid(u)-1)
        
        return p
    
    def func(self, t,psi):
        return torch.matmul(-1j*self.evolving_H(t), psi)
    
    def forward(self,ts):
        return odeint(self.func,self.x0,ts, method=self.method)[-1]
    
    def loss(self,times):
        x  = self.forward(times)
        return torch.real(torch.dot(x.conj(), torch.matmul(self.H,x)))
    
    def train_me(self, Epochs, times,optimizer):
        losses=[]
        
        for e in range(Epochs):
            #x = self.forward(times)
            l = self.loss(times)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(l.detach())
            losses.append(l.detach())
            
        return losses