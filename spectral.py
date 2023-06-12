import numpy as np

def radius(W):
    w = W.shape[0]
    U = np.ones((w,w))/w
    E, V = np.linalg.eig(W - U)
    return np.abs(E).max()

def eigstep(alpha, W):
    w = W.shape[0]
    U = np.ones((w,w))/w
    E, V = np.linalg.eig(W - U)
    v = V[:, np.argmax(np.abs(E))]
    e = E[np.argmax(np.abs(E))]
    Q = e * np.outer(v, v)
    return alpha * Q

def eigstep_RK4(alpha, W):
    # wow !
    k1 = eigstep(1, W)
    k2 = eigstep(1, W + alpha * k1 / 2)
    k3 = eigstep(1, W + alpha * k2 / 2)
    k4 = eigstep(1, W + alpha * k3)
    return alpha * (k1 + 2 * k2 + 2 * k3 + k4) / 6
