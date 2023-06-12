
import numpy as np
from proximal_operators import Prox_T, Prox_W
from enum import Enum
from spectral import radius, eigstep, eigstep_RK4
from exact_planners import CvxMixer, CvxPlanner
from utils import mix

rate = 1

class LAverager:
    def __init__(self, A, M, N=10):
        self.A = A
        self.M = M
        self.N = N

        self.L10 = np.zeros((A,A,M))
        self.L05 = np.zeros((A,A,M))

    def __call__(self, l, W):
        # l (a,a) -- true
        # W (a,a,a)
        alpha = 0.5

        for a in range(self.A):
            self.L05[a] = self.L10[a]
        
        for a in range(self.A):
            for _ in range(self.N):
                self.L10[a] = mix(W[a,a], self.L05)
                self.L10[a,a,:] = l[a]
        
        return self.L10

class CAverager:
    def __init__(self, A, N=10):
        self.A = A
        self.N = N

        self.C10 = np.zeros((A,A,A))
        self.C05 = np.zeros((A,A,A))

    def __call__(self, c, W):
        # c (a,a) -- true
        # W (a,a,a)
        alpha = 0.5

        for a in range(self.A):
            self.C05[a] = self.C10[a]
        
        for a in range(self.A):
            for _ in range(self.N):
                self.C10[a] = mix(W[a,a], self.C05)
                self.C10[a,a,:] = c[a]
                #self.C10[a,:,a] = c[a]

        return self.C10

class ExactMixer:
    def __init__(self, A):
        self.A = A

        self.cvx_mixer = CvxMixer(A)
        self.W10 = np.zeros((A,A,A))
        self.r = 0

    def __call__(self, C, W, alpha=0.5):
        # L (a,a,a)
        # W (a,a,a)
        for a in range(self.A):
            self.W10[a] = self.cvx_mixer(C[a])
        return self.W10

class IterEigMixer:
    def __init__(self, A):
        self.A = A

        self.prox_W = Prox_W(A)

        self.W10 = np.zeros((A,A,A))
        self.W05 = np.zeros((A,A,A))
        self.r = 0

    def __call__(self, C, W, alpha=0.5):
        # C (a,a,a)
        # W (a,a,a)

        for a in range(self.A):
            self.W05[a] = self.W10[a] - eigstep_RK4(alpha, self.W10[a])
        
        for a in range(self.A):
            self.W10[a] = self.prox_W(C[a], self.W05[a])
        
        return self.W10

class IterL2Mixer:
    def __init__(self, A):
        self.A = A

        self.prox_W = Prox_W(A)

        self.W10 = np.zeros((A,A,A))
        self.W05 = np.zeros((A,A,A))
        self.r = 0

    def __call__(self, C, W, alpha=0.5):
        # C (a,a,a)
        # W (a,a,a)
        U = np.ones((self.A, self.A)) / self.A

        for a in range(self.A):
            self.W05[a] = self.W10[a] - alpha * (U - self.W10[a])
        
        for a in range(self.A):
            self.W10[a] = self.prox_W(C[a], self.W05[a])
        
        return self.W10

class PgdEigMixer:
    def __init__(self, A):
        self.A = A

        self.prox_W = Prox_W(A)

        self.W10 = np.zeros((A,A,A))
        self.W05 = np.zeros((A,A,A))
        self.r = 0

    def __call__(self, C, W, alpha=0.5):
        # C (a,a,a)
        # W (a,a,a)

        for a in range(self.A):
            self.W05[a] = self.W10[a] - eigstep_RK4(alpha, self.W10[a])
        
        for a in range(self.A):
            self.W10[a] = self.prox_W(C[a], mix(W[a,a], self.W05))
        
        return self.W10

class PgdL2Mixer:
    def __init__(self, A):
        self.A = A

        self.prox_W = Prox_W(A)

        self.W10 = np.zeros((A,A,A))
        self.W05 = np.zeros((A,A,A))
        self.r = 0

    def __call__(self, C, W, alpha=0.5):
        # C (a,a,a)
        # W (a,a,a)

        U = np.ones((self.A, self.A)) / self.A

        for a in range(self.A):
            self.W05[a] = self.W10[a] - alpha * (U - self.W10[a])
        
        for a in range(self.A):
            self.W10[a] = self.prox_W(C[a], mix(W[a,a], self.W05))
        
        return self.W10

class ExactPlanner:
    def __init__(self, A, M):
        self.A = A
        self.M = M

        self.cvx_planner = CvxPlanner(A, M)
        self.T10 = np.zeros((A,A,M))
        self.r = 0

    def __call__(self, L, W, alpha=0.5):
        # L (a,a,M)
        # T (a,a,M)
        for a in range(self.A):
            self.T10[a] = self.cvx_planner(L[a])
        return self.T10

class IterPlanner:
    def __init__(self, A, M):
        self.A = A
        self.M = M

        self.prox_T = Prox_T(A, M)

        self.T10 = np.zeros((A,A,M))
        self.T05 = np.zeros((A,A,M))
        self.r = 0

    def __call__(self, L, W, alpha=0.5):
        # L (a,a,a)
        # W (a,a,a)

        for a in range(self.A):
            self.T05[a] = self.T10[a] - alpha * L[a]
        
        for a in range(self.A):
            self.T10[a] = self.prox_T(self.T05[a])
        
        return self.T10

class ExactAvgU:
    def __init__(self, A, M):
        self.A = A
        self.M = M

        self.L_averager = LAverager(A, M)
        self.C_averager = CAverager(A)
        self.mixer = ExactMixer(A)
        self.planner = ExactPlanner(A, M)

        self.L = np.zeros((A,A,M))
        self.T = np.zeros((A,A,M))
        self.C = np.zeros((A,A,A))
        self.W = np.zeros((A,A,A))

        self.r = 0
    
    def __call__(self, c, l):
        W = np.zeros((self.A, self.A, self.A))
        C = np.zeros((self.A, self.A, self.A))
        for a in range(self.A):
            W[a,a] = c[a] / np.maximum(np.sum(c[a], axis=0).reshape((-1,1)), 0)
            C[a,a] = c[a]

        self.L = self.L_averager(l, self.W)
        self.C = C

        if (self.r % rate) == 0:
            self.T = self.planner(self.L, self.W)
        self.r += 1

        self.W = W
        return self.T

class ExactAvgW:
    def __init__(self, A, M, theta=0.5):
        self.A = A
        self.M = M
        self.theta = theta

        self.L_averager = LAverager(A, M)
        self.C_averager = CAverager(A)
        self.mixer = ExactMixer(A)
        self.planner = ExactPlanner(A, M)

        self.L = np.zeros((A,A,M))
        self.T = np.zeros((A,A,M))
        self.C = np.zeros((A,A,A))
        self.W = np.zeros((A,A,A))

        self.r = 0
    
    def __call__(self, c, l):
        W = np.zeros((self.A, self.A, self.A))
        for a in range(self.A):
            W[a,a] = c[a] / np.maximum(np.sum(c[a], axis=0).reshape((-1,1)), 0)

        self.L = self.L_averager(l, self.W)
        self.C = self.C_averager(c, self.W)
        if (self.r % rate) == 0:
            self.T = self.planner(self.L, self.W)
            self.W = self.mixer((self.C > 1e-1).astype(np.float64), self.W)
        self.r += 1

        self.W = W if radius(W) < radius(self.W) else self.W
        return self.T

class NidsPlanner:
    def __init__(self, A, M):
        self.A = A
        self.M = M

        self.prox_T = Prox_T(A, M)

        self.T10 = np.zeros((A,A,M))
        self.T05 = np.zeros((A,A,M))
        self.w10 = np.zeros((A,A,M))
        self.w05 = np.zeros((A,A,M))
        self.r = 0

    def __call__(self, L, W, alpha=0.5):
        # L (a,a,a)
        # W (a,a,a)
        I = np.zeros((self.A, self.A, self.A))
        I[:] = np.eye(self.A)

        for a in range(self.A):
            self.T10[a] = self.T05[a]
            self.w10[a] = self.w05[a]

        for a in range(self.A):
            self.T05[a] = self.prox_T(mix(W[a,a], self.T10[a]) - alpha * L[a] - self.w10[a])
            self.w05[a] = self.w10[a] + 0.5 * mix((I-W)[a,a], self.T10)
        
        return self.T10

class PgdPlanner:
    def __init__(self, A, M):
        self.A = A
        self.M = M

        self.prox_T = Prox_T(A, M)

        self.T10 = np.zeros((A,A,M))
        self.T05 = np.zeros((A,A,M))
        self.r = 0

    def __call__(self, L, W, alpha=0.5):
        # L (a,a,a)
        # W (a,a,a)

        for a in range(self.A):
            self.T05[a] = self.T10[a]
        
        for a in range(self.A):
            self.T10[a] = self.prox_T(mix(W[a,a], self.T05) - alpha * L[a])
        
        return self.T10

class PgdU:
    def __init__(self, A, M):
        self.A = A
        self.M = M

        self.mixer = ExactMixer(A)
        self.planner = PgdPlanner(A, M)

        self.L = np.zeros((A,A,M))
        self.T = np.zeros((A,A,M))
        self.C = np.zeros((A,A,A))
        self.W = np.zeros((A,A,A))

        self.r = 0
    
    def __call__(self, c, l):
        L = np.zeros((self.A, self.A, self.M))
        W = np.zeros((self.A, self.A, self.A))
        C = np.zeros((self.A, self.A, self.A))
        for a in range(self.A):
            W[a,a] = c[a] / np.maximum(np.sum(c[a], axis=0).reshape((-1,1)), 0)
            C[a,a] = c[a]
            L[a,a] = l[a]

        self.L = L
        self.C = C

        self.T = self.planner(self.L, self.W)

        self.W = W
        return self.T

class PgdW:
    def __init__(self, A, M):
        self.A = A
        self.M = M

        self.mixer = PgdEigMixer(A)
        self.planner = PgdPlanner(A, M)

        self.L = np.zeros((A,A,M))
        self.T = np.zeros((A,A,M))
        self.C = np.zeros((A,A,A))
        self.W = np.zeros((A,A,A))

        self.r = 0
    
    def __call__(self, c, l):
        L = np.zeros((self.A, self.A, self.M))
        W = np.zeros((self.A, self.A, self.A))
        C = np.ones((self.A, self.A, self.A))
        for a in range(self.A):
            W[a,a] = c[a] / np.maximum(np.sum(c[a], axis=0).reshape((-1,1)), 0)
            C[a,a] = c[a]
            L[a,a] = l[a]

        self.L = L
        self.C = C

        self.T = self.planner(self.L, self.W)
        self.W = self.mixer((self.C > 1e-1).astype(np.float64), self.W)

        self.W = W if radius(W) < radius(self.W) else self.W
        return self.T