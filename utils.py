from time import time
import numpy as np

class Time:
    def __enter__(self):
        self.t0 = time()
        return self
    def __exit__(self ,type, value, traceback):
        self.t1 = time()
    def __repr__(self) -> str:
        return f'{self.t1-self.t0:.4f}'
    @property
    def elapsed(self):
        return self.t1 - self.t0

def mix(W, X):
    return np.sum([w * x for w, x in zip(W.ravel(), X)], axis=0)

def digital(X):
    return (X > 0.5).astype(np.int64)

def nice(X):
    return np.round(X * 100) / 100

def is_plan(plan):
    assert (np.sum(plan, axis=0) <= 1 + 1e-3).all(), 'agent double assignment'
    assert (np.sum(plan, axis=1) <= 1 + 1e-3).all(), 'mission double assignment'
    assert (plan >= -1e-3).all(), 'not non-negative'