from pyOpt import Optimization
from pyOpt import PSQP
from pyOpt import SLSQP
from pyOpt import CONMIN
from pyOpt import COBYLA
from pyOpt import SOLVOPT
from pyOpt import KSOPT
from pyOpt import NSGA2
from pyOpt import ALGENCAN
from pyOpt import FILTERSD
import numpy as np
import random

np.random.seed(1)

# Create random Reachable sets

c = 5 # cache capacity
k = 5 # each user should be in at most K reachable sets
m = 2 # at most M items can be initially recommended to each user
L = 5 # the total number of items that the user will receive either through 
# direct recommendation or indirectly (because it is in reachable sets of users to whom items 
#have been recommended) should not be more than L
number_of_items = 10
number_of_users = 12
items= np.array(range(0,number_of_items)) # videos
users = np.array(range(0,number_of_users)) # users
# initialize x0 for the optimization
x = np.full(len(items),0.5)
y = np.full((len(items), len(users)),0.5)
x0 = np.concatenate([x.flatten(), y.flatten()])
# Random Reachable sets
reachable_sets = np.array(np.array([[np.array(random.sample(list(users), random.choice(users))) for user in users] for i in items]))
I = np.zeros((len(items), len(users), len(users)))
for i in items:
    for u in users:
        for v in reachable_sets[i][u]:
            I[i][u][v] = 1

intersections = np.zeros((len(items),len(users),len(users)))
for i in items:
    for u in users:
        for v in users:
            if v!=u:
                intersections[i][u][v] = len(set(reachable_sets[i][u])&set(reachable_sets[i][v]))

size_of_sets = np.vectorize(len)
reachable_sets_len = size_of_sets(reachable_sets)

# Objective function
def objective(z):


    #x = z[:len(items)]
    #y = z[len(items):].reshape(len(items),len(users))

    x = z['x']
    y = z['y'].reshape(len(items),len(users))
    
    #The first term in (1) counts the users to which an item is 
    #initially recommended and the users in the ensuing reachable sets
    #first_term = sum([(1+len(reachable_sets[i][u])) * y[i][u] for u in users])
    #The second term in (1) is due to the overlap of reachable sets Riu,Riv
    #second_term = 0.5*sum([sum([len(set(reachable_sets[i][u])&set(reachable_sets[i][v]))*y[i][u]*y[i][v] for v in users if v!=u]) for u in users])
    # Ekfonisi
    #return -sum([x[i]*(sum([(1+len(reachable_sets[i][u])) * y[i][u] for u in users]) - 0.5*sum([sum([len(set(reachable_sets[i][u])&set(reachable_sets[i][v]))*y[i][u]*y[i][v] for v in users if v!=u]) for u in users])) for i in items])
    # Paper

    # Constraints
    g = [0.0,np.zeros(len(users))]
   
    g[0] = sum([x[i] for i in items]) - c
    g[1] = [sum([y[i][u] for i in items]) + sum([sum([I[i][u][v]*y[i][u] for v in users]) for i in items])-L for u in users]
    fail=0

    return -sum([x[i]*sum([(1+len(reachable_sets[i][u])) * y[i][u] - 0.5*sum([len(set(reachable_sets[i][u])&set(reachable_sets[i][v]))*y[i][u]*y[i][v] for v in users if v!=u]) for u in users]) for i in items]),g,fail
    #return -np.dot(x,(((1+reachable_sets_len)*y).sum(axis=1) - 0.5*(y*((y[:,:,None]*intersections).sum(axis=2))).sum(axis=1)))

# Initialize problem
opt_prob = Optimization('Algorithm A: Real-valued Relaxation',objective,use_groups=True)
opt_prob.addVarGroup('x',len(x),'c',lower=0.0,upper=1.0,value=0.5)
opt_prob.addVarGroup('y',len(y.flatten()),'c',lower=0.0,upper=1.0,value=0.5)
opt_prob.addObj('f')
opt_prob.addCon('g0','e')
#opt_prob.addConGroup('g1',len(users),'i')
print(opt_prob)

# Instantiate Optimizer (SLSQP) & Solve Problem
slsqp = SLSQP()
slsqp.setOption('IPRINT',-1)
slsqp(opt_prob,sens_type='FD')
print(opt_prob.solution(0))