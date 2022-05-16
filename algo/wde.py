import torch
import numpy as np 
from .model import MLP
from torch.optim import Adam

class Method(object):
    def __init__(self, max_iter=0):
        self.max_iter = max_iter

    def estimate(self, As, Bs):
        raise NotImplementedError

    def train(self, As, Bs):
        raise NotImplementedError

class wgan(Method):
    def __init__(self, input_shape, vol, max_iter=2000):
        super(wgan, self).__init__(max_iter)
        self.input_shape = input_shape
        self.vol = vol
        self.clamp_max = 0.01
        self.losses = np.zeros((self.max_iter,))
        self.vals = np.zeros((self.max_iter,))

        self.device = torch.device("cuda")
        self.disc = MLP(input_shape, hidden_dim=64, num_outputs=1).to(device=self.device)
        self.disc_optim = Adam(self.disc.parameters(), lr=0.002)

    def update_parameters(self, As, Bs, shuffle=True):
        if shuffle:
            np.random.shuffle(As)
            np.random.shuffle(Bs)
        As = torch.FloatTensor(As).to(self.device)
        Bs = torch.FloatTensor(Bs).to(self.device)
        VAs = self.disc(As)
        VBs = self.disc(Bs)

        loss1 = VAs.mean()
        loss2 = -VBs.mean()
        self.disc_optim.zero_grad()
        loss1.backward()
        loss2.backward()
        self.disc_optim.step()
        for p in self.disc.parameters():
            p.data.clamp_(-self.clamp_max, self.clamp_max)

        return (loss1 + loss2).item()

    def estimate(self, As, Bs):
        As = torch.FloatTensor(As).to(self.device)
        Bs = torch.FloatTensor(Bs).to(self.device)
        VAs = self.disc(As)
        VBs = self.disc(Bs)
        rv = torch.abs(VAs.mean() - VBs.mean())
        return rv.squeeze().detach().cpu().numpy()
    
    def score(self, s, idx):
        s = torch.FloatTensor(s).to(self.device)
        score = self.disc(s.unsqueeze(0))
        if idx == 0:
            score = -score
        return score.squeeze().detach().cpu().numpy()

    def train(self, As, Bs):
        for i in range(self.max_iter):
            loss = self.update_parameters(As, Bs)
            self.losses[i] = loss       
            self.vals[i] = self.estimate(As, Bs)         

class bgrl(Method):
    def __init__(self, input_shape, vol, max_iter=500):
        super(bgrl, self).__init__(max_iter)
        self.input_shape = input_shape
        self.vol = vol
        self.gamma = 1.0 # control the effect of softmax
        self.losses = np.zeros((self.max_iter,))
        self.vals = np.zeros((self.max_iter,))

        self.device = torch.device("cuda")
        self.mu = MLP(input_shape, hidden_dim=64, num_outputs=1).to(device=self.device)    
        self.nu = MLP(input_shape, hidden_dim=64, num_outputs=1).to(device=self.device)    
        self.tf_optim = Adam(list(self.mu.parameters()) + list(self.nu.parameters()), lr=0.002)   

    def update_parameters(self, As, Bs, shuffle=True):
        if shuffle:
            np.random.shuffle(As)
            np.random.shuffle(Bs)        
        As = torch.FloatTensor(As).to(self.device)
        Bs = torch.FloatTensor(Bs).to(self.device)
        VAs = self.mu(As)
        VBs = self.nu(Bs)

        cost = torch.norm(As - Bs, p=2, dim=-1)
        damping = VAs.squeeze() - VBs.squeeze() - cost
        damping = self.gamma * torch.exp(damping / self.gamma)
        loss = -VAs.mean() + VBs.mean() + damping.mean()

        self.tf_optim.zero_grad()
        loss.backward()
        self.tf_optim.step()

        return loss.item()

    def estimate(self, As, Bs):
        As = torch.FloatTensor(As).to(self.device)
        Bs = torch.FloatTensor(Bs).to(self.device)
        VAs = self.mu(As)
        VBs = self.nu(Bs)
        rv = torch.abs(VAs.mean() - VBs.mean())
        return rv.squeeze().detach().cpu().numpy()

    def score(self, s, idx):
        s = torch.FloatTensor(s).to(self.device)
        if idx == 0:
            return self.mu(s.unsqueeze(0)).squeeze().detach().cpu().numpy()
        else:
            return -self.nu(s.unsqueeze(0)).squeeze().detach().cpu().numpy()

    def train(self, As, Bs):
        for i in range(self.max_iter):
            loss = self.update_parameters(As, Bs)
            self.losses[i] = loss       
            self.vals[i] = self.estimate(As, Bs)  

class pwil(Method):
    def __init__(self):
        super(pwil, self).__init__()

    def estimate(self, As, Bs):
        vol, _ = As.shape
        dist = np.zeros((vol,))
        avail = [True for _ in range(vol)]
        for i in range(vol):
            min_idx = 0
            min_dist = np.inf
            for j in range(vol):
                if avail[j]:
                    d = np.linalg.norm(As[i] - Bs[j], ord=2)
                    if d < min_dist:
                        min_dist = d
                        min_idx = j
            avail[min_idx] = False
            dist[i] = min_dist
        return np.mean(dist)

    def train(self, As, Bs):
        pass

class swd(Method):
    def __init__(self):
        super(swd, self).__init__()
        self.M = 100

    def estimate(self, As, Bs):
        _, d = As.shape
        wd = np.zeros((self.M,))
        cov = np.eye(d)
        mean = np.zeros(d)
        v = np.random.multivariate_normal(mean, cov, self.M)
        l = 1./np.linalg.norm(v, ord=2, axis=-1)
        v = v * l[:, None]
        for i in range(self.M):
            pA = np.matmul(As, v[i])
            pB = np.matmul(Bs, v[i])
            pA = np.sort(pA)
            pB = np.sort(pB)
            wd[i] = np.linalg.norm(pA - pB, ord=2)
        return np.mean(wd)

    def train(self, As, Bs):
        pass

class pwd(Method):
    def __init__(self):
        super(pwd, self).__init__()
        self.M = 100

    def estimate(self, As, Bs):
        _, d = As.shape
        wd = np.zeros((self.M,))
        cov = np.eye(d)
        mean = np.zeros(d)
        v = np.random.multivariate_normal(mean, cov, self.M)
        l = 1./np.linalg.norm(v, ord=2, axis=-1)
        v = v * l[:, None]
        for i in range(self.M):
            pA = np.matmul(As, v[i])
            pB = np.matmul(Bs, v[i])
            iA = np.argsort(pA)
            iB = np.argsort(pB)
            A = As[iA]
            B = Bs[iB]
            wd[i] = np.mean(np.linalg.norm(A - B, ord=2, axis=-1))
        return np.mean(wd)

    def train(self, As, Bs):
        pass

# Advanced pwds
class apwd(Method):
    def __init__(self):
        super(apwd, self).__init__()
        self.M = 100

    def estimate(self, As, Bs):
        _, d = As.shape
        wd = np.zeros((self.M,))
        cov = np.eye(d)
        mean = np.zeros(d)
        v = np.random.multivariate_normal(mean, cov, self.M)
        l = 1./np.linalg.norm(v, ord=2, axis=-1)
        v = v * l[:, None]
        for i in range(self.M):
            pA = np.matmul(As, v[i])
            pB = np.matmul(Bs, v[i])
            iA = np.argsort(pA)
            iB = np.argsort(pB)
            A = As[iA]
            B = Bs[iB]
            wd[i] = self.match(A, B)
        return np.mean(wd)

    def match(self, As, Bs):
        D = As.shape[0]
        T = Bs.shape[0]
        count_A = [T for _ in range(D)]
        count_B = [D for _ in range(T)]
        i = j = 0
        ret = 0.0
        while i<D:
            if count_A[i] == 0:
                i += 1
            elif count_B[j] == 0:
                j += 1
            else:
                delta = min(count_A[i], count_B[j])
                count_A[i] -= delta
                count_B[j] -= delta
                ret += np.linalg.norm(As[i] - Bs[j], ord=2, axis=-1)*delta
        return ret/(D*T)

    def train(self, As, Bs):
        pass

class opwd(Method):
    def __init__(self, input_shape, vol=0):
        super(opwd, self).__init__()
        self.vol = vol
        self.d = input_shape
        self.M = 100

    def estimate(self, As, Bs):
        _, d = As.shape
        wd = np.zeros((self.M,))
        cov = np.eye(self.d)
        mean = np.zeros(self.d)
        v = np.random.multivariate_normal(mean, cov, self.M)
        ovs = np.zeros((self.M, self.d))
        k = 0
        for t in range(self.M):
            vt = v[t]
            if k > 0:
                for i in range(k):
                    vt -= np.dot(vt, ovs[t-i-1])*ovs[t-i-1]
            lt = 1./np.linalg.norm(vt, ord=2, axis=-1)
            vt *= lt
            ovs[t] = vt
            k = (k+1) % self.d
        for i in range(self.M):
            pA = np.matmul(As, v[i])
            pB = np.matmul(Bs, v[i])
            iA = np.argsort(pA)
            iB = np.argsort(pB)
            A = As[iA]
            B = Bs[iB]
            wd[i] = np.mean(np.linalg.norm(A - B, ord=2, axis=-1))
        return np.mean(wd)

    def train(self, As, Bs):
        pass