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
import pickle 

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

def output_as_string(N,order,Hout,varmap_out):
    from julia import Main
    Main.using("QuantumCumulants")
    Main.using("ModelingToolkit")
    Main.using("Symbolics")
    Main.eval("""σ(i, axis) = Sigma(h,Symbol(:σ_,i), axis, i)""")
    Main.eval("σx(i) = σ(i, 1)")
    Main.eval("σy(i) = σ(i, 2)")
    Main.eval("σz(i) = σ(i, 3)")
    Main.eval("N="+str(N))
    Main.eval("order="+str(order))
    Main.eval("h = ⊗([SpinSpace(Symbol(:Spin,i)) for i=1:N]...)")
    Outops = call_Hout(Hout)
    
    sHout = 'Hout='
    for i in range(len(Hout)): sHout+=str(Hout[i][0])+'*'+Outops[i]+'+'
        
    Main.eval(sHout[:-1])
    Main.eval("""avgout = cumulant_expansion(average(Hout),order)""")
    
    s=str(Main.avgout)[15:-1]
    for x in [str(integer) for integer in range(1,10)]: s=s.replace(x+'var', x+'*var').replace(x+'⟨', x+'*⟨')
    for key in varmap_out: 
            s=s.replace(key, varmap_out[key])
    for x in [str(integer) for integer in range(0,10)]: s=s.replace(x+'x', x+'*x')
    return s.replace("⟨","").replace("⟩","").replace("^","**").replace("*)",")").replace(')self',')*self')

def call_Main(N, Graph, Fields, Hout, order,Main):
    Main.using("QuantumCumulants")
    Main.using("ModelingToolkit")
    Main.using("Symbolics")
    Main.eval("N="+str(N))
    Main.eval("order="+str(order))
    Main.eval("h = ⊗([SpinSpace(Symbol(:Spin,i)) for i=1:N]...)")
    Jops, hops = call_operators(N,Graph, Fields,Main)

    
    Js, hs = call_constants(N, Graph, Fields,Main)
    
    Outops = call_Hout(Hout)
    sops = 'ops=['
    for i in range(len(Hout)): sops +=Outops[i]+','
        
    sops=sops[:-1]+']'
    Main.eval(sops)
    
    sH='H='
    for i in range(len(Graph)): sH+=Js[i]+'*'+Jops[i]+'+'
    for i in range(len(Fields)): sH+=hs[i]+'*'+hops[i]+'+'
    
    Main.eval(sH[:-1])
    Main.eval("""eqs = meanfield(ops, H; order=order)""")
    Main.eval("""eqs = complete(eqs, order=order)""")
    Main.eval("""sys=ODESystem(eqs; name=:cumeqs)""")
    return

def Make_Ansatz(N, Gs, Fs, Ops, order, name):
    """This is a very important function which takes in the graph and fields of a 
    Hamiltonian ansatz (Gs,Fs) of a certain size (N) and creates the cumulant 
    expansion of a certain order (order). Ops contains information about the kind of 
    expectation values one wishes to be expressed in the ODEs. 
    
    We advise that the Ansatz of a certain geometry and order is only made once 
    since this procedure has a runtime of O(N!). Thereafter, the ansatz is stored in a 
    file called name.pt. 
    The Ansatz is then loaded into the Cumulant NN. In this sense, the loading should only take 
    a runtime of O(N^k)."""
    from julia import Main
    #The following line builds the ansatz in Julia language
    call_Main(N, Gs, Fs, Ops, order, Main)
    
    #Now, we need to parse it and produce a dictionary 
    
    #Get variable and parameter maps
    varmap = {str(Main.sys.states[i])[15:-1]:"x"+"["+str(i)+"]"\
                   for i in range(len(Main.sys.states))}
    parmap = {str(Main.sys.ps[i])[15:-1]:'' for i in range(len(Main.sys.ps))}
    
    cntJ=0
    cnth=0
    for key in parmap.keys(): 
        if 'J' in key: 
            parmap[key]='self.J('+str(cntJ)+')'
            cntJ+=1
        else: 
            parmap[key]='self.p('+str(cnth)+',t)'
            cnth+=1
            
    #Getting the variable map for the output
    varmap_out = {}
    for key in varmap:
            newkey = key.replace("var","").replace("(t)","")[1:-1]
            varmap_out[newkey] = varmap[key]
    
    #Parsing the equations to make them into python-compatible strings
    def parse_string(s):
        s=s.replace("im","0").replace(" ","").replace(")J",")*J")
        s=s.replace(')var',')*var')
        for key in varmap.keys(): 
            s=s.replace(key, varmap[key])
        for key in parmap.keys():
            s=s.replace(key, parmap[key])
            
        for x in [str(integer) for integer in range(1,10)]: 
            s=s.replace(x+'self',x+'*self').replace(x+'x', x+'*x')
        return s.replace("⟨","").replace("⟩","").replace("^","**").replace("*)",")").replace(')self',')*self')
    
    equations_as_strings = [parse_string(str(Main.sys.eqs[i].rhs)[15:-1]) for i in range(len(Main.sys.eqs))]
    
    dic = {'Variable map': varmap, 
           'Parameter map': parmap, 
           'Variable map for output': varmap_out,
           'Equations': equations_as_strings, 
           'Operator base': Ops,
           'Graph': Gs,
           'Fields': Fs,
           'N': N, 
           'order': order}
    
    with open(name, 'wb') as outfile:
        pickle.dump(dic, outfile)
    print('Ansatz saved!')
    
    return 
    
class CumulantNN(nn.Module):
    def __init__(self, N, file, Hout, T=1.0, n_basis = 5, basis='Fourier', x0=None, adjoint=False, Js=None, vs=None,method='euler'):
        super(CumulantNN, self).__init__()
        with open(file, 'br') as infile:
            self.dic = pickle.load(infile)
    
        self.n_qubits=self.dic['N']
        self.order=self.dic['order']
        self.varmap = self.dic['Variable map']
        self.varmap_out= self.dic['Variable map for output']
        self.parmap = self.dic['Parameter map']
        self.Graph   = self.dic['Graph']
        self.Fields  = self.dic['Fields']
        self.equations_as_strings = self.dic['Equations']
        
        self.n_basis= n_basis
        self.basis = basis
        self.method=method
        self.T=T
        
        if Js is None: self.Js  = nn.Parameter(torch.rand(len(self.Graph)))
        else: self.Js = nn.Parameter(Js)
        
        if vs is None: self.vs =  nn.Parameter(torch.rand([len(self.Fields), n_basis]))
        else: self.vs = nn.Parameter(vs)
        
        if x0 is None: self.exps_to_vect([[0.0,0.0,1.0] for i in range(self.n_qubits)])
        else: self.x0=x0
        
        self.odeint = odeint_adjoint if adjoint else odeint
        
        if basis not in ['Legendre', 'Fourier', 'poly']: raise ValueError('Basis ' + basis + ' not supported')
        
        self.get_equations()
        self.s = eval('lambda x:' + output_as_string(self.n_qubits,self.order,Hout,self.varmap_out))
        
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
    
    def get_equations(self):
        self.equations= [] 
        for i in range(len(self.varmap)):
            self.equations.append(eval('lambda self, t, x: '+self.equations_as_strings[i].replace('j','')))
        return 
    
    def func(self,t,x):
        dx_dt = torch.zeros(x.size()[0])
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
        self.Graph=Graph
        self.Fields=Fields
        self.Hout=Hout
        self.T=T
        self.n_basis=n_basis
        self.basis=basis
        self.method=method
        
        if Js is None: self.Js  = nn.Parameter(torch.rand(len(self.Graph)))
        else: self.Js = nn.Parameter(Js,requires_grad=True)
        
        if vs is None: self.vs =  nn.Parameter(torch.rand([len(self.Fields), n_basis]))
        else: self.vs = nn.Parameter(vs,requires_grad=True)
        
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
        for term in self.Graph: 
            ops = [self.get_op(term[i]) for i in range(len(term)//2)]
            sites = term[len(term)//2:]
            self.Gops.append(self.add_op(ops, sites))
        self.Fops=[]
        for term in self.Fields: 
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