import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.mathematical_program import  MathematicalProgram
from optimization_algorithms.interface.objective_type import OT

class ConstrainedProblem(MathematicalProgram):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def evaluate(self, x) :
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """
        # add the main code here! E.g. define methods to compute value y and Jacobian J
        # and return as a tuple of arrays, namely of dim (1) and (1,n)
        y = np.array([np.sum(x), np.matmul(x.T, x) - 1, -x[0]])
        grad = np.array([[1, 1], [2*x[0], 2*x[1]], [-1, 0]])

        return y, grad

    def getDimension(self) : 
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem, e.g.
        return 2

    def getFHessian(self, x) : 
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        # add code to compute the Hessian matrix
        return np.ones(self.getDimension())

    def getInitializationSample(self) : 
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.array([0.5, 0.5]) 

    def getFeatureTypes(self):
        """
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.f, OT.ineq, OT.ineq]
    
    def report(self, verbose):
        """
        See Also
        ------
        MathematicalProgram.report
        """
        strOut = "Test problem"
        return strOut