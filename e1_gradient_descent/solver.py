import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.nlp_solver import  NLPSolver

class FlexibleSolver(NLPSolver):

    def __init__(self, linesearch=True, backtracking=True, verbose=1, **kwargs):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        if backtracking and not linesearch:
            raise ValueError("linesearch is a prerequisite for backtracking.")
        self.linesearch = linesearch
        self.backtracking = backtracking
        self.verbose = verbose
        self.kwargs = kwargs
        
    def evaluate(self, x, add_to_trace=False):
        if add_to_trace:
            return self.problem.evaluate(x)[0][0]
        return self.problem.mathematical_program.evaluate(x)[0][0]
        

    def solve(self):
        # stepsize
        alpha = self.kwargs.get("alpha", 1)         
        # line search factor, generally in interval [0.01, 0.3]
        rho_ls = self.kwargs.get("rho_ls", 0.05)         
        # increase_factor
        rho_alpha_plus = self.kwargs.get("rho_alpha_plus", 1.2)        
        # line search decrease_factor
        rho_alpha_minus = self.kwargs.get("rho_alpha_minus", 0.5)        
        # tolerance  
        theta = self.kwargs.get("theta", 1e-3)
        # metric
        metric = self.kwargs.get("metric", np.identity(self.problem.getDimension()))
        self.dim = self.problem.getDimension()
        
        # initialization
        x = self.kwargs.get("x_init", self.problem.getInitializationSample())
        fx = self.problem.mathematical_program.evaluate(x)[0][0]
        step_norm = np.sum(np.linalg.norm(np.inf - x, ord=2))
        if self.dim <=3:
            print(f"x_init = {x} \nfx_init = {fx:.4f} \nstep_norm = {step_norm:.5f}")
        
        # optimization loop
        iteration = 0
        while step_norm >= theta:
            phi, J = self.problem.evaluate(x)
            fx = phi[0]
            gradient = J[0]            
            if self.linesearch:
                delta = -gradient / np.linalg.norm(gradient, ord=2)
                step = alpha * np.linalg.inv(metric) @ delta
                fx_new = self.problem.mathematical_program.evaluate(x + step)[0][0]
                
                if self.backtracking:
                    while fx_new > fx + rho_ls * gradient.T @ step:
                        alpha = rho_alpha_minus * alpha
                        step = alpha * np.linalg.inv(metric) @ delta
                        fx_new = self.problem.mathematical_program.evaluate(x + step)[0][0]
                else:
                    while fx_new > fx:
                        alpha = rho_alpha_minus * alpha
                        step = alpha * np.linalg.inv(metric) @ delta
                        fx_new = self.problem.mathematical_program.evaluate(x + step)[0][0]             
                # alpha = rho_alpha_plus * alpha
                alpha = 1            
            else:
                # plain gradient descent
                step = -gradient * alpha            
            x = x + step
            iteration += 1
            step_norm = np.sum(np.linalg.norm(step, ord=2))            
            if self.verbose == 2:
                self.print_progress(iteration, x, fx, gradient, step, step_norm)
        
        if self.verbose == 1:
            self.print_progress(iteration, x, fx, gradient, step, step_norm)            
        return x
    
    def print_progress(self, iteration, x, fx, gradient, step, step_norm):        
        if self.dim <= 3:
            print(f"\niteration = {iteration}")
            print(f"x = {x}")
            print(f"fx = {fx:.4f}")
            print(f"gradient = {gradient}")  
            print(f"step = {step}")
            print(f"step_norm = {step_norm:.5f}")            
        else:
            print(f"\niteration = {iteration}")
            print(f"fx = {fx:.4f}")
            print(f"step_norm = {step_norm:.5f}")
                
