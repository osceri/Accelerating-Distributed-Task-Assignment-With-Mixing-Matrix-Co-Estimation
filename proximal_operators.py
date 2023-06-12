import cvxpy as cp
import numpy as np
from spectral import radius, eigstep, eigstep_RK4

class Prox_T:
    def __init__(self, A, M):
        # initialize proximal operator
        # Will L2-project onto space of viable plans
        self.A = A
        self.M = M
        self.plan = cp.Variable((A,M))
        self.plan_prev = cp.Parameter((A,M))
    
        objective = cp.Minimize(cp.sum_squares(self.plan - self.plan_prev))
        constraints = [
            self.plan >= 0,
            cp.sum(self.plan, axis=0) <= 1,
            cp.sum(self.plan, axis=1) <= 1,
        ]
        self.problem = cp.Problem(objective, constraints)

        self(np.zeros((A,M)))
    
    def __call__(self, plan):
        # project
        self.plan_prev.value = plan
        try:
            self.problem.solve()
            assert 'optimal'in self.problem.status
            return self.plan.value
        except:
            return plan

class Prox_W:
    def __init__(self, A, pos=False, sym=False):
        # initialize proximal operator
        # Will L2-project onto space of viable mixing matrices
        self.A = A

        # steup CVXPY problem
        self.W = cp.Variable((A,A))
        self.nA = cp.Parameter((A,A))
        self.W_prev = cp.Parameter((A,A))
    
        objective = cp.Minimize(cp.sum_squares(self.W - self.W_prev))
        constraints = [
            cp.sigma_max(self.W) <= 1,
            cp.sum(self.W, axis=0) == 1,
            cp.multiply(self.W, 1 - self.nA) == 0,
        ] + ([ self.W >= 0 ] if pos else []) + ([ self.W == self.W.T] if sym else [])
        self.problem = cp.Problem(objective, constraints)

        self(np.ones((A,A)), np.zeros((A,A)))

    def __call__(self, C, W):
        self.W_prev.value = W
        self.nA.value = C
        try:
            self.problem.solve()
            assert 'optimal' in self.problem.status
            return self.W.value
        except:
            return W