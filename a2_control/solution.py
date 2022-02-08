import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class LQR(MathematicalProgram):
    """
    Parameters
    K integer
    A in R^{n x n}
    B in R^{n x n}
    Q in R^{n x n} symmetric
    R in R^{n x n} symmetric
    yf in R^n

    Variables
    y[k] in R^n for k=1,...,K
    u[k] in R^n for k=0,...,K-1

    Optimization Problem:
    LQR with terminal state constraint

    min 1/2 * sum_{k=1}^{K}   y[k].T Q y[k] + 1/2 * sum_{k=1}^{K-1}      u [k].T R u [k]
    s.t.
    y[1] - Bu[0]  = 0
    y[k+1] - Ay[k] - Bu[k] = 0  ; k = 1,...,K-1
    y[K] - yf = 0

    Hint: Use the optimization variable:
    x = [ u[0], y[1], u[1],y[2] , ... , u[K-1], y[K] ]

    Use the following features:
    1 - a single feature of types OT.f
    2 - the features of types OT.eq that you need
    """

    def __init__(self, K, A, B, Q, R, yf):
        """
        Arguments
        -----
        K: integer
        A: np.array 2-D
        B: np.array 2-D
        Q: np.array 2-D
        R: np.array 2-D
        yf: np.array 1-D
        """
        # in case you want to initialize some class members or so...
        self.K = K
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.yf = yf
        self.n = len(yf)
        self.H = np.zeros((K*self.n, 2*K*self.n))
        
        
        for t in range(self.K):
            if t > 0:
                self.H[self.n*t : self.n*(t + 1), self.n * (2*t - 1) : self.n * 2*t] = - A
            
            self.H[self.n*t : self.n*(t + 1), self.n * 2*t : self.n * (2*t + 1)] = - B
            self.H[self.n*t : self.n*(t + 1), self.n * (2*t + 1) : self.n * (2*t + 2)] = np.eye(self.n)


    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J
        # y = ...
        # J = ...

        # y is a 1-D np.array of dimension m
        # J is a 2-D np.array of dimensions (m,n)
        # where m is the number of features and n is dimension of x
        # return  y  , J
        u = x.reshape((2 * self.K, self.n))[ : : 2]
        y = x.reshape((2 * self.K, self.n))[1 : : 2]
        
        sos = np.array([np.trace(.5 * u @ self.R @ u.T)+np.trace(.5 * y @ self.Q @ y.T)])

        Jsos = np.zeros((self.getDimension()))
    
        for t in range(self.K):
            Jsos[self.n * 2 * t : self.n * (2 * t + 1)] = self.R @ u[t]
            Jsos[self.n * (2 * t + 1) : self.n * (2 * t + 2)] = self.Q @ y[t]
            
        Jsos = np.reshape(Jsos, (1, -1))
        
        
        h_left = self.H @ x
        Jh_left = self.H
        
        h_right = y[self.K - 1,:] - self.yf
        Jh_right = np.zeros((self.n, 2 * self.K * self.n))
        Jh_right[:, -self.n:] = np.eye(self.n)

        phi = np.hstack((sos, h_left, h_right))        
        J = np.hstack((Jsos[np.newaxis, ...], Jh_left[np.newaxis, ...], Jh_right[np.newaxis, ...]))[0, :, :]
        return  phi, J


    def getFHessian(self, x):
        """
        """
        # return
        dim = self.getDimension()
        H = np.zeros((dim, dim))

        for t in range(self.K):
            H[self.n*2*t : self.n*(2*t + 1) , self.n*2*t : self.n*(2*t + 1)] = self.R
            H[self.n*(2*t+1) : self.n*(2*t+2) , self.n*(2*t+1) : self.n*(2*t+2)] = self.Q

        return H


    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return
        return 2 * self.n * self.K

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.zeros(self.getDimension())

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        # return
        return [OT.f] + [OT.eq] * ((self.K + 1) * self.n)
