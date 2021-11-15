import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.nlp_solver import  NLPSolver

class Solver0(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        
        # in case you want to initialize some class members or so...


    def solve(self) :
        """

        See Also:
        ----
        NLPSolver.solve

        """
        
        # write your code here

        # use the following to get an initialization:
        x = self.problem.getInitializationSample()
        fx = self.problem.evaluate(x)[0][0]
        print(f"x_init = {x} \nfx_init = {fx:.4f}")
        step_norm = np.sum(np.linalg.norm(np.inf - x, ord=2))
        
        # use the following to query the problem:
        # phi, J = self.problem.evaluate(x)
        # phi is a vector (1D np.array); use phi[0] to access the cost value (a float number). J is a Jacobian matrix (2D np.array). Use J[0] to access the gradient (1D np.array) of the cost value.

        # now code some loop that iteratively queries the problem and updates x til convergence....
        theta = 1e-4
        alpha = 0.1
        iteration = 0
        while step_norm >= theta:
            phi, J = self.problem.evaluate(x)
            fx = phi[0]
            gradient = J[0]
            step = -gradient * alpha
            x = x + step
            iteration += 1
            step_norm = np.sum(np.linalg.norm(step, ord=2))
            
            print(f"\niteration = {iteration}")
            print(f"x = {x}")
            print(f"fx = {fx:.4f}")
            print(f"gradient = {gradient}")  
            print(f"step = {step}")
            print(f"step_norm = {step_norm:.5f}")          

        # finally:
        return x 
