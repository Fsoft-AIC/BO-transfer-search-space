import numpy as np
from functions.rover_functions import *
from functions.push_function import *
from functions.helper import ConstantOffsetFn, NormalizedInputFn


class Push:
    def __init__(self):
        self.f = PushReward()
        self.minimum = 0
        self.lb = self.f.xmin
        self.ub = self.f.xmax
        self.minimum = 0
        self.minimum_point = np.zeros(14)

    def __call__(self, x):
        return self.f(x)

class Rover:
    def __init__(self):
        start = np.zeros(2) + 0.05
        goal = np.array([0.95, 0.95])
        domain = create_small_domain(start, goal)
        n_points = domain.traj.npoints
        raw_x_range = np.repeat(domain.s_range, n_points, axis=1)
        f_max = 5.0
        f = ConstantOffsetFn(domain, f_max)
        self.f = NormalizedInputFn(f, raw_x_range)
        self.minimum = 0
        self.lb = -3 * np.ones(20)
        self.ub = 3 * np.ones(20)
        self.minimum = 0
        self.minimum_point = np.zeros(20)

    def __call__(self, x):
        return -self.f(x)

class Ackley_:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -2 * np.ones(dim)
        self.ub = 2 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.zeros(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        S = 0
        M = 0
        for i in range(n):
            S += x[i]**2
        S = -0.2*np.sqrt(S/n)
        for i in range(n):
            M += np.cos(2*np.pi*x[i])
        M = np.exp(M/n)
        val = -20 * np.exp(S) - M + np.exp(1) + 20
        return val - self.minimum
    
class Ackley:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.zeros(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        x_copy = x * 32
        S = 0
        M = 0
        for i in range(n):
            S += x_copy[i]**2
        S = -0.2*np.sqrt(S/n)
        for i in range(n):
            M += np.cos(2*np.pi*x_copy[i])
        M = np.exp(M/n)
        val = -20 * np.exp(S) - M + np.exp(1) + 20
        return val - self.minimum
    
class Eggholder:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = -959.6407/1000
        self.minimum_point = np.array([1,404.2319/512])
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        x_copy = x * 512
        S = -(x_copy[1]+47)*np.sin(np.sqrt(np.abs(x_copy[1]+x_copy[0]/2+47))) - x_copy[0]*np.sin(np.sqrt(np.abs(x_copy[0]-(x_copy[1]+47))))
        return S/1000 - self.minimum
    
class Rosenbrock:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.ones(dim)/2.048
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        x_copy = x * 2.048
        S = 0
        for i in range(n-1):
            S += 100*(x_copy[i+1] - x_copy[i]**2)**2 + (x_copy[i] - 1)**2
        return S - self.minimum
    
class Dixon:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.array([2**(-1+1/(2**(i-1))) for i in range(1, dim + 1)])/10
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        x_copy = x * 10
        x1 = x_copy[0]
        dim = len(x_copy)
        term1 = (x1-1)**2
        S = 0
        for i in range(1,n):
            xi = x_copy[i]
            xold = x_copy[i-1]
            new = (i+1) * (2*xi**2 - xold)**2
            S += new
        return S + term1 - self.minimum
    
class Stybtang:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = -39.16599 * dim
        self.minimum_point = dim * [-2.903534/5]
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        x_copy = x * 5
        S = 0
        for i in range(n):
            S += x_copy[i]**4 - 16 * x_copy[i]**2 + 5 * x_copy[i]
        return S/2 - self.minimum 

class Levy:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.ones(dim)/10
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        S = 0
        x_copy = 10 * x
        w = 1 + (x_copy -1)/4
        S = np.sin(np.pi * w[0])**2 + (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1]**2))
        for i in range(1,n - 1):
            S += (w[i]-1)**2 * (1 + 10*np.sin(np.pi * w[i]+1)**2)
        return S - self.minimum
    
class Hartmann6:
    def __init__(self):
        self.lb = -1 * np.ones(6)
        self.ub = 1 * np.ones(6)
        self.minimum = -3.32237
        self.minimum_point = np.array([0.20169,0.150011,0.476874,0.275332,0.311652,0.6573])
        self.alpha = [1.0,1.2,3.0,3.2]

        self.A = np.array([[10,3,17,3.5,1.7,8],
                  [0.05,10,17,0.1,8,14],
                  [3,3.5,1.7,10,17,8],
                  [17,8,0.05,10,0.1,14]])
    
        self.P = 10**(-4) * np.array([[1312,1696,5569,124,8283,5886],
                             [2329,4135,8307,3736,1004,9991],
                             [2348,1451,3522,2883,3047,6650],
                             [4047,8828,8732,5743,1091,381]])
        
    def __call__(self, x):
        assert len(x) == 6
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        S = 0
        for i in range(4):
            M = 0
            for j in range(6):
                M -= self.A[i][j] * (x[j] - self.P[i][j])**2
            S -= self.alpha[i] * np.exp(M)
        return S - self.minimum
    
class Branin:
    def __init__(self):
        self.lb = -5 * np.ones(2)
        self.ub = 5 * np.ones(2)
        self.minimum = 0.397887
        self.minimum_point = np.array([np.pi, 2.275])
        
    def __call__(self, x):
        assert len(x) == 2
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        # parameters
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        bra = a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s
        return bra - self.minimum
    
class Powell:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -2 * np.ones(dim)
        self.ub = 2 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.zeros(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        S = (x[0] + 10 * x[1])**2 + 5 * (x[2] - x[3])**2 + (x[1] - 2 * x[2])**4 + 10 * (x[0] - x[3])**4
        return S - self.minimum
    
class Hyper:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.zeros(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        x_copy = x * 65
        S = 0
        for i in range(n):
            for j in range(i):
                S += x_copy[j]**2
        return S - self.minimum
    
class Schwefel:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.ones(dim) * 420.9687/500
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        x_copy = x * 500
        S = 418.9829*self.dim
        for i in range(n):
            S -= x_copy[i]*np.sin(np.sqrt(np.abs(x_copy[i])))
        return S/1000 - self.minimum
    
class Perm:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.array([1/(i+1) for i in range(dim)])/dim
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        x_copy = x * self.dim
        S = 0
        for i in range(n):
            M = 0
            for j in range(n):
                M += (j+1)*(x_copy[j]**(i+1) - 1/(j+1)**(i+1))
            S += M**2
        return S/1e45 - self.minimum
    
class Alpine:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = -22.144
        self.minimum_point = [0.5834]*dim
        self.local_point = [4.816/5-1,4.816/5-1,7.917/5-1]
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        x_copy = (x + 1) * 5
        S = 1
        for i in range(n):
            if x_copy[i] < 0:
                return 100.0
            else:
                S *= np.sqrt(np.clip(x_copy[i],0,10))*np.sin(np.clip(x_copy[i],0,10))
        return S - self.minimum 

class Griewank:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.zeros(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        x_copy = x * 600
        S = 0
        M = 1
        for i in range(n):
            S += x_copy[i]**2 / 4000
            M *= np.cos(x_copy[i]/np.sqrt(i+1))
        
        return S-M+1 - self.minimum
    
class Rastrigin:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -1 * np.ones(dim)
        self.ub = 1 * np.ones(dim)
        self.minimum = 0
        self.minimum_point = np.zeros(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        #assert np.all(x <= self.ub) and np.all(x >= self.lb)
        n = len(x)
        x_copy = x * 5.12
        S = 0
        for i in range(n):
            S += x_copy[i]**2 - 10*np.cos(2*np.pi*x_copy[i])
        S += 10*self.dim
        
        return S - self.minimum