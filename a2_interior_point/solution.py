import numpy as np
import sys, time

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverInteriorPoint(NLPSolver):

    def __init__(self, verbose=False, **kwargs):
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
        assert len(self.index_f) <= 1  # at most, only one term of type OT.f
        # get all sum-of-square features
        self.index_r = [i for i, x in enumerate(types) if x == OT.sos]
        # get all inequalities features
        self.index_g = [i for i, x in enumerate(types) if x == OT.ineq]

    def getVars(self, x, mu):
        phi, J = self.problem.evaluate(x)

        c = 0
        H = np.zeros([self.dim, self.dim])
        grad = np.zeros(self.dim)

        if len(self.index_f) > 0:
            c += phi[self.index_f][0]
            grad += J[self.index_f][0]
            H += self.problem.getFHessian(x)

        if len(self.index_r) > 0:
            c += phi[self.index_r].T @ phi[self.index_r]
            grad += 2 * J[self.index_r].T @ phi[self.index_r]
            H += 2 * J[self.index_r].T @ J[self.index_r]

        if len(self.index_g) > 0:
            for i in self.index_g:
                if phi[i] > 0:
                    c = np.inf
                    grad = np.zeros(self.dim)
                    break
                c -= mu * np.log(-phi[i])
                grad -= mu * J[i] / phi[i]
                H += mu * (1/phi[i]**2) * np.outer(J[i], J[i])

        # Check H is positive definite
        if np.any(np.linalg.eigvals(H) < 0):
            H += (abs(min(np.linalg.eigvals(H)))+.02) * np.eye(self.dim)
        return c, grad, H


    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        
        # write your code here
        # mu 
        mu = 1
        # decrease stepsize for mu
        rho_mu = 0.5
        # tolerance  
        mu_theta = self.kwargs.get("mu_theta", 1e-6)
        # stepsize
        alpha = self.kwargs.get("alpha", 1)         
        # line search factor, generally in interval [0.01, 0.3]
        rho_ls = self.kwargs.get("rho_ls", 0.1)         
        # increase_factor
        rho_alpha_p = self.kwargs.get("rho_alpha_plus", 1.2)        
        # line search decrease_factor
        rho_alpha_m = self.kwargs.get("rho_alpha_minus", 0.5)        
        # tolerance  
        theta = self.kwargs.get("theta", 1e-3)
        # damping
        lamda = self.kwargs.get("lambda", 1e-3) 
        
        x = self.kwargs.get("x_init", self.problem.getInitializationSample())

        self.dim = self.problem.getDimension()

        start_time = time.time()

        mu_iter = 1
        newton_iter = 1
        backtracking_iter = 0
        count = 1

        fx, grad, H = self.getVars(x, mu)
        
        if self.verbose : print(f"Problem : {self.problem.report(verbose=True)} \n x_init = {x} \nfx_init = {fx} \n")
        

        while mu_iter <= 1000:
            xt = x.copy()
            
            while newton_iter <= 1000:
                fx, grad, H = self.getVars(x, mu)
                count += 1
                delta = np.linalg.solve(H + lamda*np.eye(self.dim), -grad)
                if grad.T @ delta > 0:
                    delta = delta/np.linalg.norm(delta,2)
                
                fx_n, grad_n, H_n = self.getVars(x + alpha *delta, mu)
                count += 1

                backtrack = 0
                while fx_n > (fx + rho_ls * np.matmul(grad.T,(alpha * delta))) and backtrack < 100:
                    alpha = rho_alpha_m * alpha
                    fx_n, grad_n, H_n = self.getVars(x + alpha *delta, mu)
                    count += 1
                    backtrack += 1
                
                backtracking_iter += backtrack
                
                x += alpha * delta            
                alpha *= rho_alpha_p
                
                if np.linalg.norm(alpha * delta) < theta : break
                newton_iter += 1
                
            if np.linalg.norm(xt-x) < mu_theta : break
                
            mu_iter += 1
            mu *= rho_mu

        if self.verbose :
            print(f"Solution : {x}")
            print(f"Calls to program : {count}")
            print(f"Iterations over mu : {mu_iter}")
            print(f"Newton steps : {newton_iter}")
            print(f"Backtracking iterations : {backtracking_iter}")
            print(f"Time elapsed : {time.time()-start_time}s")
            print("\n")
        return x
