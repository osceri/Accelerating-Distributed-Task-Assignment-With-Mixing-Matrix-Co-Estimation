import numpy as np
from utils import is_plan
from proximal_operators import Prox_T

class cPgdPlanner:
    def __init__(self, A, M, N = 1000):
        self.A = A
        self.M = M

        self.alpha = 0.1
        self.prox = Prox_T(A, M)
        self.plan = np.zeros((A,M))
    
    def __call__(self, cost, alpha = 0.1):
        self.plan = self.prox(self.plan - alpha * cost)
        return self.plan
    
if __name__ == '__main__':
    from exact_planners import CvxPlanner
    from utils import nice, digital, Time

    time = Time()
    A = 50
    M = 50
    cost = -np.random.normal(2, 5, (A,M))

    exact_planner = CvxPlanner(A, M)
    with time:
        exact = exact_planner(cost)
    print(time)

    cpgd = cPgdPlanner(A, M)
    steps = 4
    with time:
        for _ in range(steps):
            plan = cpgd(cost)
    print(time)

    print(np.sum(exact))
    print(np.sum(plan))
    print(np.sum(exact != plan))


