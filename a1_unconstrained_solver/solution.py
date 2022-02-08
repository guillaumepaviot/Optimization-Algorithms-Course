import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT
import time


class SolverUnconstrained(NLPSolver):

    def __init__(self, verbose=0, **kwargs):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        # in case you want to initialize some class members or so...
        self.verbose = verbose
        self.kwargs = kwargs
    
    def setProblem(self, problem):
        super().setProblem(problem)
        self.dim = self.problem.getDimension()
        types = self.problem.getFeatureTypes()
        # get all features of type f
        self.index_f = [i for i, x in enumerate(types) if x == OT.f]
        assert len(self.index_f) <= 1 # at most, only one term of type OT.f
        # get all sum-of-square features
        self.index_r = [i for i, x in enumerate(types) if x == OT.sos]


    def calcDelta(self, fx, grad, H, lamda, approx=False):
        D = H + lamda * np.eye(H.shape[0])
        try:
            D_inv = np.linalg.inv(D)
            delta = -1*np.matmul(D_inv, grad.T)
        except:
            delta = np.divide(-grad, abs(grad + 1e-5))
            pass

        if approx:
            delta = np.matmul(delta,fx)
            grad = 2*np.matmul(grad.T, fx)
        if np.matmul(grad, delta)>0:
            delta = np.divide(-grad, abs(grad+1e-5))
        return delta

    def getHessian(self, x, grad):
        try:
            if len(self.index_r) > 0: NotImplementedError()
            H = self.problem.getFHessian(x)
            approx = False
        except NotImplementedError:
            H =  2*np.matmul(grad.T, grad)
            approx = True
        
        return H, approx


    def computeCost(self, fx):
        c = 0
        if len(self.index_f) > 0:
            c += fx[self.index_f][0]
        if len(self.index_r) > 0:
            c += fx[self.index_r].T @ fx[self.index_r]
        return c

    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        # write your code here
        # stepsize
        alpha = self.kwargs.get("alpha", 1)         
        # line search factor, generally in interval [0.01, 0.3]
        rho_ls = self.kwargs.get("rho_ls", 0.05)         
        # increase_factor
        rho_alpha_p = self.kwargs.get("rho_alpha_plus", 1.2)        
        # line search decrease_factor
        rho_alpha_m = self.kwargs.get("rho_alpha_minus", 0.5)        
        # tolerance  
        theta = self.kwargs.get("theta", 1e-3)
        # damping
        lamda = self.kwargs.get("lambda", 1e-3) 

        self.dim = self.problem.getDimension()

        start_time = time.time()

        iteration = 0
        x = self.kwargs.get("x_init", self.problem.getInitializationSample())
        approx = False
        fx, grad = self.problem.evaluate(x)
        iteration += 1
        fx_o = fx
        if self.dim <=3:
            print(f"Problem : {self.problem.report(verbose=True)} \n x_init = {x} \nfx_init = {fx} \n")
        H, approx = self.getHessian(x, grad)
        
        if grad.shape[0] == 1:
            grad = grad[0]
        delta = self.calcDelta(fx, grad, H, lamda, approx = approx)

        while np.linalg.norm(alpha*delta, np.inf) > theta:
            x_new = x + alpha*delta
            fx_n, grad_n = self.problem.evaluate(x_new)
            iteration += 1
            print(f"x new {x_new}")
            print(f"delta {delta}")
            print(f"grad {grad_n}")
            H_n, approx = self.getHessian(x_new, grad_n)
            print(f"H {H_n}")
           
            if grad_n.shape[0] == 1:
                grad_o = grad_n[0]
                grad_n = grad_n[0]
            else:
                grad_o = grad_n
            c_new = self.computeCost(fx_n)
            c = self.computeCost(fx_o)
            fx_o = fx_n
            if approx:
                grad_o = 2*np.matmul(grad_o.T, fx_o)

            while (c_new) > (c + rho_ls * np.matmul(grad_o.T,(alpha * delta))):
                alpha *= rho_alpha_m
                delta = self.calcDelta(fx_n, grad_n, H_n, lamda, approx=approx)
                x_new_t = x_new + alpha * delta
                fx_o = fx_n
                fx_n, grad_n = self.problem.evaluate(x_new_t)
                iteration += 1
                H_n, approx = self.getHessian(x_new_t, grad_n)
                
                if grad_n.shape[0] == 1:
                    grad_n = grad_n[0]
                c_new = self.computeCost(fx_n)
                c = self.computeCost(fx_o)
            try:
                x_new = x_new_t
            except:
                pass
            x = x_new
            alpha = min(rho_alpha_p*alpha, 1)
            delta = self.calcDelta(fx_n, grad_n, H_n, lamda, approx = approx )

        # finally:
        print(f"Required evaluations: {iteration} \n Time needed : {time.time() - start_time :.3f}s \n")
        return x

