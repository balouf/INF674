# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:12:15 2015

@author: Fabien
"""

"""
Notes:
- I use cell writing to write the solutions of TPs
Cell should be executed in order
Cf https://pythonhosted.org/spyder/editor.html
- The code proposed may not be optimal, but it is working (3.5)
"""


import numpy as np
# Numpy is used to access some nice math primitives
import matplotlib.pyplot as plt
# Matplotlib draws nice plots


"""
Question 1: the link between Pext and pk's is that a process eventually dies
iff all its sons evnetually die. This leads to
Pext = \sum_k p_k P_ext^k
"""


#%% Question 2

#==============================================================================
# Pext exact computation in the case of bimodal distribution
#==============================================================================

def pext_bim_exact(mu):
    '''gives Pext in case of Poisson distribution'''
    return np.minimum(1,2/mu-1)

# Test on a simple example
mu=np.arange(0,2.05,.05) # Values of mu
p_exact = pext_bim_exact(mu)
print(p_exact) # Extinction (exact)

#%% Question 3, part 1

#==============================================================================
# Simulation: Pext bimodal after time t using n_trials
#==============================================================================

def pext_bim_trials(mu,t=1000,n=1000):
    '''for p=[1-mu/2,0,mu/2], gives the probability to have an extinction
    after t steps using n probes'''
    
    pext=np.zeros(mu.size) # initiate extinction vector
    for i in range(mu.size): # iterate over values of mu
        pext[i]=0
        for j in range(n):
            p = 1+(2*(np.random.rand(t) < mu[i]/2)-1).cumsum()
            if (p==0).any():
                pext[i]+=1
    return pext/n


#%% Question 3, part 2 (Display)

t = 100
n = 100

p_trials = pext_bim_trials(mu, t, n) # Extinction (sim)

plt.figure(1)
plt.clf()
plt.plot(mu,p_trials, 'r',label='$P_{ext}$ after '+str(t)+' steps ('+str(n)+' trials)')
plt.plot(mu,p_exact, 'k',label='$P_{ext}$')
plt.ylim(0,1)
plt.xlabel('$\mu$')
plt.ylabel('$P_{ext}$')
plt.legend(loc=3)
plt.show()

#%% Question 4, part 1

def pop_after_t(p=[.5, 0, .5],t=1000):
    '''gives the distribution of the alive population after t single
    spawn attempts; extinction is implicit (result has a sum <=1)'''

    pop=np.array([1]) # At t=0, probability to have 1 member is 1
    for i in range(t): # Running through time
        pop=np.convolve(pop,p)[1:] # A spawn is made; extinct situations are discarded
    return pop


def pext_bim_distrib(mu,t=1000):
    '''for p=[1-mu/2,0,mu/2], gives the probability to have an extinction
    after t steps'''
    
    pext=np.zeros(mu.size) # initiate extinction vector
    for i in range(mu.size): # iterate over values of mu
        # compute extinction
        pext[i]=1-pop_after_t([1-mu[i]/2, 0, mu[i]/2],t).sum() 
    return pext



#%% Question 4, part 2 (Display)

t = 100
n = 100

p_trials = pext_bim_trials(mu, t, n) # Extinction (trials)
p_distrib = pext_bim_distrib(mu, t)  # (distribution after t)

plt.figure(1)
plt.clf()
plt.plot(mu,p_trials, 'r',label='$P_{ext}$ after '+str(t)+' steps ('+str(n)+' trials)')
plt.plot(mu,p_distrib, 'b',label='$P_{ext}$ after '+str(t)+' steps')
plt.plot(mu,p_exact, 'k',label='$P_{ext}$')
plt.ylim(0,1)
plt.xlabel('$\mu$')
plt.ylabel('$P_{ext}$')
plt.legend(loc=3)
plt.show()

#%% Question 5, part 1

#==============================================================================
# Pext exact computation in the case of geometric distribution
#==============================================================================

def pext_geo_exact(mu):
    '''gives Pext in case of Geometric distribution'''
    return np.minimum(1,1/mu)

#==============================================================================
# Simulation: Pext geometric after time t, using n trials
#==============================================================================
def pext_geo_trials(mu, t=1000, n=10000,):
    '''simulates Pext after t in case of Poisson distribution'''
    pext=np.zeros(mu.size)
    for i in range(mu.size):
        pext[i]=0
        for j in range(n):
            p=1+(np.random.geometric(1/(mu[i]+1),t)-2).cumsum()
            if (p==0).any():
                pext[i]+=1
    return pext/n

#==============================================================================
# Simulation: Pext geometric after time t, using truncated distribution
#==============================================================================
def pext_geo_distrib(mu,t=1000,pmax=50):
    '''Simulates Pext after t in case of Poisson distribution, using truncated Poisson'''
    pext=np.zeros(mu.size)
    for i in range(mu.size):
        a = mu[i]/(1+mu[i])
        p=np.array(list(map(lambda x: (1-a)*a**x,range(pmax))))
        p[-1]=1-p.sum()
        pop=pop_after_t(p,t)
        pext[i]=1-pop.sum()
    return pext

#%% Question 5, part 2 (display)

t = 100
n = 100

p_exact = pext_geo_exact(mu)
p_trials = pext_geo_trials(mu, t, n) # Extinction (trials)
p_distrib = pext_geo_distrib(mu, t)  # (distribution after t)

plt.figure(1)
plt.clf()
plt.plot(mu,p_trials, 'r',label='$P_{ext}$ after '+str(t)+' steps ('+str(n)+' trials)')
plt.plot(mu,p_distrib, 'b',label='$P_{ext}$ after '+str(t)+' steps')
plt.plot(mu,p_exact, 'k',label='$P_{ext}$')
plt.ylim(0,1)
plt.xlabel('$\mu$')
plt.ylabel('$P_{ext}$')
plt.legend(loc=3)
plt.show()

#%% Question 6, part 1

#==============================================================================
# Pext exact computation in the case of Poisson distribution
#==============================================================================

def pext_poisson_exact(mu):
    '''gives Pext in case of Poisson distribution'''
    
    pext=np.zeros(mu.size)
    for i in range(mu.size):
        if mu[i]<=1:
            pext[i]=1
        else:
            y=np.exp(mu[i]*(pext[i]-1))
            while y!=pext[i]:
                pext[i]=y
                y=np.exp(mu[i]*(pext[i]-1))
    return pext


#==============================================================================
# Simulation: Pext Poisson after time t, using n trials
#==============================================================================

def pext_poisson_trials(mu, t=1000, n=1000):
    '''simulates Pext after t in case of Poisson distribution'''
    pext=np.zeros(mu.size)
    for i in range(mu.size):
        pext[i]=0
        for j in range(n):
            p=1+(np.random.poisson(mu[i],t)-1).cumsum()
            if (p==0).any():
                pext[i]+=1
    return pext/n


#==============================================================================
# Simulation: Pext Poisson after time t, using truncated distribution
#==============================================================================

def pext_poisson_distrib(mu,t=1000,pmax=25):
    '''Simulates Pext after t in case of Poisson distribution, using truncated Poisson'''
    pext=np.zeros(mu.size)
    for i in range(mu.size):
        p=np.array(list(map(lambda x: np.exp(-mu[i])*mu[i]**x/np.math.factorial(x),range(pmax))))
        p[-1]=1-p.sum()
        pop=pop_after_t(p,t)
        pext[i]=1-pop.sum()
    return pext


#%% Question 6, part 2 (Display)

t = 100
n = 100

p_exact = pext_poisson_exact(mu)
p_trials = pext_poisson_trials(mu, t, n) # Extinction (trials)
p_distrib = pext_poisson_distrib(mu, t)  # (distribution after t)

plt.figure(1)
plt.clf()
plt.plot(mu,p_trials, 'r',label='$P_{ext}$ after '+str(t)+' steps ('+str(n)+' trials)')
plt.plot(mu,p_distrib, 'b',label='$P_{ext}$ after '+str(t)+' steps')
plt.plot(mu,p_exact, 'k',label='$P_{ext}$')
plt.ylim(0,1)
plt.xlabel('$\mu$')
plt.ylabel('$P_{ext}$')
plt.legend(loc=3)
plt.show()

#%% Not a question: compare the 3 cases (exact)

p_bim = pext_bim_exact(mu)
p_geo = pext_geo_exact(mu)
p_poisson = pext_poisson_exact(mu)

plt.figure(1)
plt.clf()
plt.plot(mu,p_geo, 'b',label='$P_{ext}$ (Geometric)')
plt.plot(mu,p_poisson, 'k',label='$P_{ext}$ (Poisson)')
plt.plot(mu,p_bim, 'r',label='$P_{ext}$ (Bimodal)')
plt.ylim(0,1)
plt.xlabel('$\mu$')
plt.ylabel('$P_{ext}$')
plt.legend(loc=3)
plt.show()

#%% Question 7: it is a bonus, isn't it?