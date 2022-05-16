import numpy as np 
from copy import deepcopy
import cv2

eps = 1e-6
R = np.array([[0, -1], [1, 0]]) # rotation matrix to calculate normal
class Block(object):
    def __init__(self, points):
        # points are input in order
        self.N = len(points)
        assert self.N > 0
        self.D = len(points[0])
        self.points = np.zeros((self.N + 1, self.D))
        self.points[:-1,:] = points
        self.points[-1,:] = points[0]
        self.As = self.points[:-1,:]
        self.Bs = self.points[1:,:]
        self.norms = np.matmul(R, (self.Bs - self.As).T).T 
        nlen = np.linalg.norm(self.norms, axis=-1, keepdims=True)
        nlen = np.repeat(nlen, 2, axis=-1)
        self.norms = self.norms / nlen # normalized norms

    def in_region(self, points):
        # input: M points, M x D ndarray
        M = points.shape[0]
        Ts = np.repeat(np.expand_dims(points, axis=1), self.N, axis=1)
        Ss = np.repeat(np.expand_dims(self.As, axis=0), M, axis=0)
        Ns = np.repeat(np.expand_dims(self.norms, axis=0), M, axis=0)
        prod = np.einsum('ijk,ijk->ij', Ts - Ss, Ns)
        ret = np.all(prod < 0.0, axis=-1)
        return ret
    
    def intersection(self, pA, pB):
        # input: 2 points, from pA to pB
        pA = np.repeat(np.expand_dims(pA, axis=0), self.N, axis=0)
        pB = np.repeat(np.expand_dims(pB, axis=0), self.N, axis=0) # N x 2
        X = pA - self.As
        Y = self.As - self.Bs 
        Z = pA - pB
        # parameter t=det(X,Y)/det(Z,Y)
        numerator_t = X[:,0]*Y[:,1] - X[:,1]*Y[:,0]
        numerator_u = -Z[:,0]*X[:,1] + Z[:,1]*X[:,0]
        denominator = Z[:,0]*Y[:,1] - Z[:,1]*Y[:,0]
        ts = numerator_t / (denominator + eps)
        us = numerator_u / (denominator + eps)
        return ts, us

    def first_contact(self, P, delta):
        # input: from P to P+delta
        ts, us = self.intersection(P, P+delta)
        ts = np.where(ts > 1., np.inf, ts)
        ts = np.where(ts < 0., np.inf, ts)
        ts = np.where(us > 1., np.inf, ts)
        ts = np.where(us < 0., np.inf, ts)
        min_t_idx = np.argmin(ts)
        t = ts[min_t_idx]
        norm = self.norms[min_t_idx]
        return t, norm

class BlockMap(object):
    def __init__(self, blocks):
        self.blocks = [Block(rec) for rec in blocks]

    def reflected_point(self, p, dp):
        # move from p to p+dp
        tr = [deepcopy(p)]
        dp = deepcopy(dp)
        while True:
            t = []
            norm = []
            for i in range(len(self.blocks)):
                bt, bnorm = self.blocks[i].first_contact(p, dp)
                t.append(bt)
                norm.append(bnorm)
            min_t_idx = np.argmin(t)
            min_t = t[min_t_idx]
            if min_t <= 1.0 and min_t >= 0.0:
                n = norm[min_t_idx]
                p += dp*min_t
                dp *= (1-min_t)
                dp -= 1.05*(dp@n)*n
                tr.append(deepcopy(p))
            else:
                break           
        return tr, deepcopy(dp)

    def render(self, img):
        scale_f = 512.0
        boundaries = 0.0
        for b in self.blocks:
            tl = ((b.As[0,:] + boundaries) * scale_f).astype(np.int16).tolist() # top left corner
            br = ((b.As[2,:] + boundaries) * scale_f).astype(np.int16).tolist() # bottom right corner
            cv2.rectangle(img, tuple(tl), tuple(br), (197, 197, 197), -1)
        return img