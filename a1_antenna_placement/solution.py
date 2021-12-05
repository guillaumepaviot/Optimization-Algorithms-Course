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
        expo = np.exp(-np.linalg.norm(x-self.P, ord=2, axis=1)**2)
        y = -self.w @ expo.T
        J = -2 * self.w @ ((x-self.P).T @ expo.T).T
        print(J)
        return np.array([y]) , J.reshape(1, -1)


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
        expo = np.exp(-np.linalg.norm(x-self.P, ord=2, axis=1)**2)
        H = 2*np.sum(self.w.T@expo)*np.eye(self.getDimension()) - 4*np.sum(self.w.T@expo*(x-self.P)*(x-self.P).T)
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
