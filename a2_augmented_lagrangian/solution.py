import numpy as np
import sys, time

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverAugmentedLagrangian(NLPSolver):

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
        self.index_g = np.array([i for i, x in enumerate(types) if x == OT.ineq], dtype=np.int64)
        # get all equalities features
        self.index_h = [i for i, x in enumerate(types) if x == OT.eq]

        self.lambda_lagrangian  = np.zeros(len(self.index_g))
        self.kappa_lagrangian  = np.zeros(len(self.index_h))


    def getVars(self, x, mu, nu):
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
            mask = self.index_g[np.logical_or(phi[self.index_g] >= 0, self.lambda_lagrangian > 0)]
            c += mu*phi[mask].T @ phi[mask] + self.lambda_lagrangian.T @ phi[self.index_g]
            grad += 2*mu*J[mask].T @ phi[mask] + self.lambda_lagrangian.T @ J[self.index_g]
            H += 2*mu*J[mask].T @ J[mask]

        if len(self.index_h) > 0:
            c += nu*phi[self.index_h].T @ phi[self.index_h] + self.kappa_lagrangian.T @ phi[self.index_h]
            grad +=  2*nu*J[self.index_h].T @ phi[self.index_h] + self.kappa_lagrangian.T @ J[self.index_h]
            H += 2*nu*J[self.index_h].T @ J[self.index_h]


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
        mu =  self.kwargs.get("mu", 1)
        # increase stepsize for mu
        rho_mu = self.kwargs.get("rho_mu", 1.2)
        # nu 
        nu =  self.kwargs.get("nu", 1)
        # increase stepsize for nu
        rho_nu = self.kwargs.get("rho_nu", 1.2)
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
        # tolerance on constraains
        epsilon = self.kwargs.get("epsilon", 1e-3)
        # damping
        lamda = self.kwargs.get("lambda", 1e-3) 
        
        x = self.kwargs.get("x_init", self.problem.getInitializationSample())

        self.dim = self.problem.getDimension()


        start_time = time.time()

        mu_iter = 0
        newton_iter = 0
        backtracking_iter = 0
        count = 1

        fx, grad, H = self.getVars(x, mu, nu)
        try :
            problem = self.problem.report(verbose=True)
        except NotImplementedError:
            problem = "Mathematical Programm"

        if self.verbose : print(f"Problem : {problem} \n x_init = {x} \nfx_init = {fx} \n")
        

        while mu_iter <= 1000:
            xt = x.copy()
            
            while newton_iter <= 1000:
                fx, grad, H = self.getVars(x, mu, nu)
                count += 1
                delta = np.linalg.solve(H + lamda*np.eye(self.dim), -grad)
                if grad.T @ delta > 0:
                    delta = delta/np.linalg.norm(delta,2)
                
                fx_n, grad_n, H_n = self.getVars(x + alpha *delta, mu, nu)
                count += 1

                backtrack = 0
                while fx_n > (fx + rho_ls * np.matmul(grad,(alpha * delta))) and backtrack < 100:
                    alpha = rho_alpha_m * alpha
                    fx_n, grad_n, H_n = self.getVars(x + alpha *delta, mu, nu)
                    count += 1
                    backtrack += 1
                
                backtracking_iter += backtrack

                x += alpha * delta            
                alpha *= rho_alpha_p
                
                if np.linalg.norm(alpha * delta) < theta : break
                newton_iter += 1


                count += 1
                phi, J = self.problem.evaluate(x)
                self.lambda_lagrangian = np.maximum(self.lambda_lagrangian + 2*mu*phi[self.index_g], np.zeros(len(self.lambda_lagrangian)))
                self.kappa_lagrangian += 2*nu*phi[self.index_h]

                
            if np.linalg.norm(xt-x) < theta and np.all(phi[self.index_g] < epsilon) and np.all(abs(phi[self.index_h]) < epsilon) : break
                
            nu *=rho_nu
            mu *= rho_mu
            mu_iter += 1
            
        if self.verbose :
            print(f"Solution : {x}")
            print(f"Calls to program : {count}")
            print(f"Iterations over mu : {mu_iter}")
            print(f"Newton steps : {newton_iter}")
            print(f"Backtracking iterations : {backtracking_iter}")
            print(f"Time elapsed : {time.time()-start_time}s")
            print("\n")
        return x

