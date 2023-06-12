import numpy as np
import networkx as nx
import cvxpy as cp
from itertools import product, combinations
from matplotlib import pyplot as plt
from spectral import radius, eigstep, eigstep_RK4
from exact_planners import CvxMixer
from proximal_operators import Prox_W

class Graph:
    def __init__(self, A):
        # initialize graph
        self.nxg = nx.Graph()
        self.A = A
        self.C = np.zeros((A,A))    # 1 if connected, 0 else (not self-connected)
        self.W = np.ones((A,A))     # mixing matrix
    
    def is_connected(self):
        # verify that a graph is connected
        return np.isclose(np.sum(np.linalg.matrix_power(self.C + np.eye(self.A), self.A*self.A) > 0), self.A*self.A)
    
    def is_self_nonconnected(self):
        # verify that the diagonal is empty
        return (np.multiply(np.eye(self.A), self.C) < 1e-4).all()

    def is_valid_graph(self):
        # run all connectivity graph tests
        assert self.is_connected(), 'not connected'
        assert self.is_self_nonconnected(), 'not self-connected'
    
    def is_convergent(self):
        # some methods require all eigs to be within [-1,1]
        return np.abs(np.linalg.eigvalsh(self.W)).max() <= 1 + 1e-4
    
    def is_stochastic(self):
        # we need all rows to sum to 1
        return np.isclose(np.sum(self.W, axis=1), 1).all()
    
    def is_symmetric(self):
        # we need the matrix to be symmetric
        return np.isclose(self.W, self.W.T).all()

    def is_unambiguous(self):
        # we need the nullspace of the matrix minus identity to be the 1-vector (consensus)
        s, v, d = np.linalg.svd(np.eye(self.A) - self.W)
        k = np.isclose(v, 0)
        null_space = d[k,:]
        return np.array([ np.isclose(a, b) for a, b in combinations(null_space, 2) ]).all()
    
    def is_exactly_connected(self):
        # we don't want spaces that are not connected to be mixed
        nA = self.C + np.eye(self.A)
        return not (np.multiply(1 - nA, self.W) >= 1e-3).any()
    
    def is_valid_mixing_matrix(self):
        # run tests to see if matrix is a valid mixing matrix
        assert self.is_convergent(), 'not convergent'
        assert self.is_stochastic(), 'not stochastic'
        assert self.is_unambiguous(), 'ambiguous'
        #assert self.is_symmetric(), 'not symmetric'
        assert self.is_exactly_connected(), 'overly connected'
    
    def spectral_radius(self):
        # report eigenvalues
        return np.linalg.eigvalsh(self.W)

    def uniform_mix(self):
        # find TRUE uniform mix -- wont pass tests
        self.W = self.C / np.maximum(np.sum(self.C, axis=0).reshape((-1,1)), 1)

    def metropolis_mix(self, l = 0.01):
        # find metropolis mix
        self.W = np.zeros((self.A,self.A))

        for i, j in product(range(self.A), range(self.A)):
            if i != j and self.C[i,j] == 1:
                self.W[i,j] = 1 / (l + np.max([np.sum(self.C[i,:]), np.sum(self.C[:,j])]))

        for i in range(self.A):
            self.W[i,i] = 1 - np.sum(self.W[i])

        self.is_valid_mixing_matrix()
    
    def eig_mix(self, pos=False, sym=False):
        cvx_mixer = CvxMixer(self.A)
        self.W = cvx_mixer(self.C, pos=pos, sym=sym)
        self.is_valid_mixing_matrix()

    def L2_mix(self, sym=False):
        # find mix closest to uniform mix (maybe lowest spectral radius)
        U = np.ones((self.A,self.A)) / self.A
        W = cp.Variable((self.A,self.A))
        nA = self.C
        problem = cp.Problem(cp.Minimize(cp.sum_squares(W-U)), [
            cp.sigma_max(W) <= 1,
            cp.sum(W, axis=0) == 1,
            cp.multiply(W, 1 - nA) == 0
        ] + ([ W == W.T ] if sym else []))
        problem.solve()
        assert problem.status == cp.OPTIMAL, 'not optimal'
        self.W = W.value

        self.is_valid_mixing_matrix()

    def random_graph(self, p = 0.5, seed=42):
        # generate random graph
        self.nxg = nx.erdos_renyi_graph(self.A, p, seed=seed)
        while True:
            self.nxg = nx.erdos_renyi_graph(self.A, p)
            self.C = nx.adjacency_matrix(self.nxg).todense()
            if not self.is_connected():
                p += 0.05
            else:
                break
        
        self.is_valid_graph()
    
    def random_change(self, p=0.5, seed=42):
        if np.random.uniform(0, 1) < p:
            i = np.random.choice(range(self.A))
            j = i
            while j == i:
                j = np.random.choice(range(self.A))
            if self.C[i,j] == 1:
                self.nxg.remove_edge(i,j)
            else:
                self.nxg.add_edge(i,j)
            self.C = nx.adjacency_matrix(self.nxg).todense()
        
    def path_grap(self):
        # generate path graph
        self.nxg = nx.path_graph(self.A)
        self.C = nx.adjacency_matrix(self.nxg).todense()

        self.is_valid_graph()
    
    def draw(self):
        # you will need to call plt.show()
        nx.draw(self.nxg)