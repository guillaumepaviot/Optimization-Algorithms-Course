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
        self.time_limit = 600
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


    def calcDelta(self, phi, J, H, lamda, approx=False):
        if approx: D = H + lamda * np.eye(H.shape[0])
        else:  D = H
        try:
            D_inv = np.linalg.inv(D)
            delta = -1*np.matmul(D_inv, J.T)
        except:
            delta = np.divide(-J, abs(J + 1e-5))
            pass

        if approx:
            delta = np.matmul(delta,phi)
            J = 2*np.matmul(J.T, phi)
        if np.matmul(J, delta)>0:
            delta = np.divide(-J, abs(J+1e-5))
        return delta

    def getHessian(self, x, J):
        try:
            if len(self.index_r) > 0: NotImplementedError()
            H = self.problem.getFHessian(x)
            approx = False
        except NotImplementedError:
            H =  2*np.matmul(J.T, J)
            approx = True
        
        return H, approx


    def computeCost(self, phi):
        c = 0
        if len(self.index_f) > 0:
            c += phi[self.index_f][0]
        if len(self.index_r) > 0:
            c += phi[self.index_r].T @ phi[self.index_r]
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
        lamda = self.kwargs.get("lambda", 1e-3) # added
        # metric
        metric = self.kwargs.get("metric", np.identity(self.problem.getDimension()))

        self.dim = self.problem.getDimension()

        start = time.time()

        iteration = 0
        x = self.kwargs.get("x_init", self.problem.getInitializationSample())
        approx = False
        phi, J = self.problem.evaluate(x)
        iteration += 1
        phi_o = phi
        H, approx = self.getHessian(x, J)
        
        if J.shape[0] == 1:
            J = J[0]
        delta = self.calcDelta(phi, J, H, lamda, approx = approx)

        while np.linalg.norm(alpha*delta, np.inf) > theta:
            if iteration > 1000 or time.time()-start > self.time_limit : break
            x_new = x + alpha*delta
            phi_n, J_n = self.problem.evaluate(x_new)
            iteration += 1
            H_n, approx = self.getHessian(x_new, J_n)
           
            if J_n.shape[0] == 1:
                J_o = J_n[0]
                J_n = J_n[0]
            else:
                J_o = J_n
            c_new = self.computeCost(phi_n)
            c = self.computeCost(phi_o)
            phi_o = phi_n
            if approx:
                J_o = 2*np.matmul(J_o.T, phi_o)

            while (c_new) > (c + rho_ls * np.matmul(J_o.T,(alpha * delta))):
                if iteration>1000 or time.time()-start > self.time_limit: break
                alpha *= rho_alpha_m
                delta = self.calcDelta(phi_n, J_n, H_n, lamda, approx=approx)
                x_new_t = x_new + alpha * delta
                phi_o = phi_n
                phi_n, J_n = self.problem.evaluate(x_new_t)
                iteration += 1
                H_n, approx = self.getHessian(x_new_t, J_n)
                
                if J_n.shape[0] == 1:
                    J_n = J_n[0]
                c_new = self.computeCost(phi_n)
                c = self.computeCost(phi_o)
            try:
                x_new = x_new_t
            except:
                pass
            x = x_new
            alpha = min(rho_alpha_p*alpha, 1)
            delta = self.calcDelta(phi_n, J_n, H_n, lamda, approx = approx )

        if iteration > 1000:
            print('Solution not found. Max. iterations reached.')
        if time.time()-start > self.time_limit:
            print('Solution not found. Max. time limit reached.')
        # finally:
        print('Required evaluations:', iteration)
        return x

