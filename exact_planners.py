import cvxpy as cp
import numpy as np
from utils import is_plan
from spectral import radius

class CvxMixer:
    def __init__(self, A):
        self.A = A
        self.W = np.ones((A,A))/A
    
    def uniform(self, C):
        W = np.ones((self.A,self.A)) / self.A
        K = np.sum(C, axis=0)
        for a in range(self.A):
            if K[a] > 0:
                W[a] = C[a] / K[a]
        return W

    def __call__(self, C, pos=False, sym=True):
        # C (A,A)
        # find optimal mix (fastest mixing markov chain on a graph, but not neccessarily positive)
        U = np.ones((self.A,self.A)) / self.A
        W = cp.Variable((self.A,self.A))
        s = cp.Variable(pos=True)

        problem = cp.Problem(cp.Minimize(s), [
            cp.sigma_max(W) <= 1,
            cp.sigma_max(W-U) <= s,
            cp.sum(W, axis=0) == 1,
            cp.multiply(W, 1 - C) == 0,
        ] + ([ 0 <= W ] if pos else []) + ([ W == W.T ] if sym else []))

        problem.solve()
        if 'optimal' in problem.status:
            if radius(W.value) > 1:
                self.W = self.uniform(C)
            else:
                self.W = W.value
            return W.value
        else:
            self.W = self.uniform(C)
        return self.W

class CvxPlanner:
    def __init__(self, A, M):
        # initialize cvx planner
        self.A = A
        self.M = M
        self.plan = cp.Variable((A,M))
        self.plans = np.zeros((A,A,M))
        self.cost = cp.Parameter((A,M))
    
        objective = cp.Minimize(cp.sum(cp.multiply(self.cost, self.plan)))
        constraints = [
            self.plan >= 0,
            cp.sum(self.plan, axis=0) <= 1,
            cp.sum(self.plan, axis=1) <= 1,
        ]
        self.problem = cp.Problem(objective, constraints)
    
    def __call__(self, cost):
        # solve cvx planner
        self.cost.value = cost
        self.problem.solve()
        assert self.problem.status == cp.OPTIMAL, 'not optimal'
        is_plan(self.plan.value)
        plan = self.plan.value
        return plan

class NaivePlanner:
    def __init__(self, A, M):
        self.plans = np.zeros((A,A,M))
        self.A = A
        self.M = M

    def __call__(self, L):
        plan = np.zeros((self.A,self.M))
        for a in range(self.A):
            l = L[a]
            m = np.argmin(l)
            if L[a, m] < 0:
                plan[a,m] = 1
        self.plans[:] = plan
        return self.plans