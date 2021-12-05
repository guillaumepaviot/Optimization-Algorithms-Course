import math
import sys
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class AntennaPlacement(MathematicalProgram):
    """
    """

    def __init__(self, P, w):
        """
        Arguments
        ----
        P: list of 1-D np.arrays
        w: 1-D np.array
        """
        # in case you want to initialize some class members or so...
        self.P = np.array(P)
        self.w = w

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J
        y = 0
        J = np.zeros((2,))
        for i in range(self.getDimension()):
            y = y + self.w[i]*np.exp(-np.linalg.norm(x-self.P[i])**2)
            J = J + self.w[i]*np.exp(-np.linalg.norm(x-self.P[i])**2)*2*(x-self.P[i])
        return np.array([-y]) , np.array([J])



    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return 2

    def getFHessian(self, x):
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        # add code to compute the Hessian matrix
        H = np.zeros((2,2))
        for i in range(self.getDimension()):
            expo = np.exp(-np.linalg.norm(x-self.P[i])**2)
            H[0,0] = H[0,0]+self.w[i]*(2*expo - 2*x[0]*2*(x[0]-self.P[i][0])*expo +2*self.P[i][0]*2*(x[0]-self.P[i][0])*expo)
            H[1,1] = H[1,1]+self.w[i]*(2*expo - 2*x[1]*2*(x[1]-self.P[i][1])*expo +2*self.P[i][1]*2*(x[1]-self.P[i][1])*expo)
            H[1,0] = H[1,0]+self.w[i]*(-2*x[0]*2*(x[1]-self.P[i][1])*expo+2*self.P[i][0]*2*(x[1]-self.P[i][1])*expo)
            H[0,1] = H[0,1]+self.w[i]*(-2*x[1]*2*(x[0]-self.P[i][0])*expo+2*self.P[i][1]*2*(x[0]-self.P[i][0])*expo)
        return H


    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        x0 = np.mean(self.P, axis=0)
        return x0

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types

        """
        return [OT.f]
