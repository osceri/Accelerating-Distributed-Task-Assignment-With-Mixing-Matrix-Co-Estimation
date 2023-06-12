import matplotlib.pyplot as plt
import numpy as np
from graphs import Graph
from enum import Enum
from itertools import combinations
import networkx as nx

class State(Enum):
    IDLE = 0
    PICK_UP = 1
    CHARGING = 2

class MissionState(Enum):
    AVAILABLE = 0
    PICKED_UP = 1
    DROPPED_OFF = 2

class ChargerState(Enum):
    AVAILABLE = 0
    IN_USE = 1

def move(bot, goal, speed, dt):
    disp = goal - bot
    dist = np.linalg.norm(disp)
    if dist < speed * dt:
        return disp, dist
    else:
        return speed * dt * disp / dist, speed * dt

class Simulator:
    def __init__(self, A, M, C, N, radius=3, speed=0.5, fuel_efficiency=0.1, dt=0.05):
        self.graph = Graph(A)
        self.A = A
        self.M = M
        self.C = C
        self.P = M + C
        self.N = N

        self.fuel_efficieny = fuel_efficiency
        self.dt = dt
        self.speed = speed
        self.radius = radius

        self.reset()
    
    def reset(self):
        self.plans = np.zeros((self.A,self.A,self.P))
        self.cost = np.zeros((self.A,self.P))

        self.bots = np.random.random((self.A,self.N))
        self.states = [ State.IDLE for a in range(self.A) ]
        self.missions = -np.ones((self.A,)).astype(np.int64)
        self.colors = np.random.random((self.A,3))

        self.charge = np.ones((self.A,))
        self.chargers = np.random.random((self.C,self.N))
        self.charger_status = [ ChargerState.AVAILABLE for c in range(self.C) ]

        self.pick_ups = np.random.random((self.M,self.N))
        self.drop_offs = np.random.random((self.M,self.N))
        self.mission_time = np.zeros((self.M,))
        self.mission_status = [ MissionState.AVAILABLE for m in range(self.M) ]

        self.update_graph(self.radius)

        self.time_elapsed = 0
        self.time_wasted = 0
        self.missions_completed = 0
    
    def update_graph(self, T):
        Y = np.argsort(np.linalg.norm(self.bots.reshape(1, -1, 2) - self.bots.reshape(-1, 1, 2), axis=2))
        G = np.outer(np.ones((self.A,)), np.arange(self.A)).astype(np.int64)
        for y, g, i in zip(G,Y,np.arange(self.A)):
            self.graph.C[i,g]=y
        self.graph.C = self.graph.C < T + 1
        self.graph.C = self.graph.C.astype(np.float64) - np.eye(self.A)

        self.graph.uniform_mix()

    def get_cost(self):
        bots_2_pick_ups = np.linalg.norm(self.bots.reshape((self.A,1,1,self.N)) - self.pick_ups.reshape((1,self.M,1,self.N)), axis=3)
        pick_ups_2_drop_offs = np.linalg.norm(self.pick_ups.reshape((1,self.M,1,self.N)) - self.drop_offs.reshape((1,self.M,1,self.N)), axis=3)
        drop_offs_2_chargers = np.linalg.norm(self.drop_offs.reshape((1,self.M,1,self.N)) - self.chargers.reshape((1,1,self.C,self.N)), axis=3)
        distance_matrix = bots_2_pick_ups + pick_ups_2_drop_offs + drop_offs_2_chargers
        mission_distance_matrix = np.min(distance_matrix, axis=2)

        L = -1 / (1 + mission_distance_matrix) - self.mission_time.reshape((1,self.M)) * 0.1

        # mission is infeasible if it requires more charge than available
        for a in range(self.A):
            for m in range(self.M):
                if mission_distance_matrix[a,m] * self.fuel_efficieny + 0.2 > self.charge[a]:
                    L[a, m] = 1
        # mission is infeasible if mission is already underway
        for k, status in enumerate(self.mission_status):
            if status != MissionState.AVAILABLE:
                L[:, k] = 1
            
        bots_2_chargers = np.linalg.norm(self.bots.reshape((self.A,1,self.N)) - self.chargers.reshape((1,self.C,self.N)), axis=2)
        distance_matrix = bots_2_chargers

        Y = -1 / (1 + distance_matrix)

        # charger is infeasible if any mission is feasible
        for a in range(self.A):
            if (L[a,:].any() < 0) or (self.charge[a] > 0.8):
                Y[a,:] = 1
        # charger is infeasible if it requires more charge than available
        for a in range(self.A):
            for c in range(self.C):
                if distance_matrix[a,c] * self.fuel_efficieny + 0.2 > self.charge[a]:
                    Y[a,c] = 1
        # charger is infeasible if it is already in use
        for k, status in enumerate(self.charger_status):
            if status != ChargerState.AVAILABLE:
                Y[:,k] = 1

        H = np.zeros((self.A, self.P))
        H[:,:self.M] = L
        H[:,self.M:] = Y

        return H
    
    def displace_bot(self, disp, a):
        D = np.linalg.norm((self.bots[a] + disp).reshape((1,self.N)) - self.bots, axis=1)

        for b, d in enumerate(D):
            if a == b:
                continue

            if d < 0.05:
                self.bots[b] -= disp

        self.bots[a] += disp
    
    def step(self):
        self.update_graph(self.radius)
        self.cost = self.get_cost()

        # for each bot
        for a in range(self.A):
            if self.charge[a] < 0:
                continue

            if self.states[a] == State.IDLE:
                # if all missions are infeasible, move towards charger
                if (self.plans[a,a] > 1e-4).any():
                    p = np.argmax(self.plans[a,a])
                    self.missions[a] = p
                    if p < self.M:
                        m = self.missions[a]
                        disp, dist = move(self.bots[a], self.pick_ups[m], self.speed, self.dt)
                        self.displace_bot(disp, a)
                        self.charge[a] -= dist * self.fuel_efficieny

                        if np.linalg.norm(self.bots[a] - self.pick_ups[m]) < 0.0001:
                            self.mission_status[m] = MissionState.PICKED_UP
                            self.states[a] = State.PICK_UP
                    if p >= self.M:
                        c = self.missions[a] - self.M
                        disp, dist = move(self.bots[a], self.chargers[c], self.speed, self.dt)
                        self.displace_bot(disp, a)
                        self.charge[a] -= dist * self.fuel_efficieny

                        if np.linalg.norm(self.bots[a] - self.chargers[c]) < 0.0001:
                            self.charger_status[c] = ChargerState.IN_USE
                            self.states[a] = State.CHARGING

            # perform pick-up. If successful, move towards drop-off
            elif self.states[a] == State.PICK_UP:
                m = self.missions[a]
                disp, dist = move(self.bots[a], self.drop_offs[m], self.speed, self.dt)
                self.displace_bot(disp, a)
                self.charge[a] -= dist * self.fuel_efficieny

                if np.linalg.norm(self.bots[a] - self.drop_offs[m]) < 0.0001:
                    self.mission_status[m] = MissionState.DROPPED_OFF
                    self.states[a] = State.IDLE
            
            # charge until full, then go idle
            elif self.states[a] == State.CHARGING:
                self.charge[a] += 0.1
                c = self.missions[a] - self.M

                if self.charge[a] > 1:
                    self.charge[a] = 1
                    self.charger_status[c] = ChargerState.AVAILABLE
                    self.states[a] = State.IDLE

        self.time_elapsed += self.dt

        # for each mission
        for m in range(self.M):
            # if mission is active, increment time
            if self.mission_status[m] != MissionState.DROPPED_OFF:
                self.mission_time[m] += self.dt
                self.time_wasted += np.sum(self.mission_time)
            if self.mission_status[m] == MissionState.DROPPED_OFF:
                self.missions_completed += 1
                self.pick_ups[m] = np.random.random((self.N,))
                self.drop_offs[m] = np.random.random((self.N,))
                self.mission_time[m] = 0
                self.mission_status[m] = MissionState.AVAILABLE