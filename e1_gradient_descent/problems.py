#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:50:40 2021

@author: aboumessouer
"""

import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.mathematical_program import  MathematicalProgram


class FSQ(MathematicalProgram):
    """
    """

    def __init__(self, c, dim):

        # in case you want to initialize some class members or so...
        # assert C.shape[0] == C.shape[1] # matrix must be quadratic
        self.c = c
        self.dim = dim
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
        y = x.T @ self.C @ x
        gradient = (self.C + self.C.T) @ x
        # J = gradient.T
        return np.array([y]), gradient.reshape(1,-1)

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
        H = self.C + self.C.T
        return H

    def getInitializationSample(self) : 
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.ones(self.getDimension())

    def report(self , verbose): 
        """
        See Also
        ------
        MathematicalProgram.report
        """
        return "Function x^T C x "


    
class FHOLE(MathematicalProgram):
    """
    """

    def __init__(self, c, a, dim):

        # in case you want to initialize some class members or so...
        self.a = a
        self.c = c
        self.dim = dim
        self.C = np.diag([c**((i-1)/(dim-1)) for i in range(1,dim+1)])
        assert self.C.shape[0] == self.C.shape[1] # matrix must be quadratic


    def evaluate(self, x) :
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """
        # add the main code here! E.g. define methods to compute value y and Jacobian J        

        
        xCx = x.T @ self.C @ x
        y = xCx / (self.a**2 + xCx)
        gradient = (2 * self.a**2 * self.C @ x) / (self.a**2 + xCx)**2
        # J = gradient.T
        # and return as a tuple of arrays, namely of dim (1) and (1,n)
        return np.array([y]), gradient.reshape(1,-1)

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
        # # add code to compute the Hessian matrix
        # t_0 = (self.a ** 2)
        # t_1 = (self.C).dot(x)
        # t_2 = (t_0 + (x).dot(t_1))
        # t_3 = ((2 * t_0) / (t_2 ** 2))
        # t_4 = ((4 * t_0) / (t_2 ** 3))
        # y = (t_3 * t_1)
        # H = ((t_3 * self.C) - ((t_4 * np.multiply.outer(t_1, t_1)) + (t_4 * np.multiply.outer(t_1, (x).dot(self.C)))))

        # add code to compute the Hessian matrix
        t_0 = self.a ** 2
        t_1 = self.C @ x
        t_2 = 2 * t_0
        t_3 = t_0 + x @ t_1
        t_4 = t_2 / t_3
        t_5 = t_2 / (t_3 ** 2)
        # y = t_4 * t_1
        H = ((t_4 * self.C) - ((t_5 * np.multiply.outer(t_1, t_1)) + (t_5 * np.multiply.outer(t_1, x @ self.C))))

        return H

    def getInitializationSample(self) : 
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.ones(self.getDimension())

    def report(self , verbose ): 
        """
        See Also
        ------
        MathematicalProgram.report
        """
        return "Function x^T C x "