import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.mathematical_program import  MathematicalProgram


class Rastrigin(MathematicalProgram):
    """
    """

    def __init__(self, c, a, dim, random):

        # in case you want to initialize some class members or so...
        # assert C.shape[0] == C.shape[1] # matrix must be quadratic
        self.c = c
        self.a = a
        self.dim = dim
        self.random = random
        self.C = np.diag([c**((i-1)/(dim-1)) for i in range(1,dim+1)])
        assert self.C.shape[0] == self.C.shape[1] # matrix must be quadratic


    def evaluate(self, x) :
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """
        # add the main code here! E.g. define methods to compute value y and Jacobian J
        # and return as a tuple of arrays, namely of dim (1) and (1,n)
        phi = np.array([[np.sin(self.a*x[0])], [np.sin(self.a*self.c*x[1])], [2*x[0]], [2*self.c*x[1]]])
        y = phi.T @ phi
        J = np.array([[self.a*np.cos(self.a*x[0]), 0], [0, self.a * self.c * np.cos(self.a*self.c*x[1])], [2, 0], [0, 2*self.c]])
        gradient = 2 * J.T @ phi
        return np.array([y[0][0]]), gradient

    def getDimension(self) : 
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem, e.g.
        return self.dim

    def getFHessian(self, x) : 
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        # add code to compute the Hessian matrix
        J = np.array([[self.a*np.cos(self.a*x[0]), 0], [0, self.a * self.c * np.cos(self.a*self.c*x[1])], [2, 0], [0, 2*self.c]])
        H = 2 * J.T @ J
        return H

    def getInitializationSample(self) : 
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        if self.random :
            return 2 * np.random.rand(self.getDimension()) - 1
        else:
            return np.ones(self.getDimension())    

    def report(self , verbose): 
        """
        See Also
        ------
        MathematicalProgram.report
        """
        return "Function phi^T * phi, phi being the Rastrigin function "