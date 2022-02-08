import numpy as np
import sys, time

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverInteriorPoint(NLPSolver):

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
        assert len(self.index_f) <= 1  # at most, only one term of type OT.f
        # get all sum-of-square features
        self.index_r = [i for i, x in enumerate(types) if x == OT.sos]
        # get all inequalities features
        self.index_g = [i for i, x in enumerate(types) if x == OT.ineq]

    def getVars(self, x, mu):
        phi, J = self.problem.evaluate(x)

        c = 0
        grad = np.zeros(self.dim)

        if len(self.index_f) > 0:
            c += phi[self.index_f][0]
            grad += J[self.index_f][0]

        if len(self.index_r) > 0:
            c += phi[self.index_r].T @ phi[self.index_r]
            grad += 2 * J[self.index_r].T @ phi[self.index_r]

        if len(self.index_g) > 0:
            for i in self.index_g:
                if phi[i] > 0:
                    c = np.inf
                    grad = np.zeros(self.dim)
                    break
                c -= mu * np.log(-phi[i])
                grad -= mu * J[i] / phi[i]

        return c, grad


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
        mu_theta = self.kwargs.get("theta", 1e-6)
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
        gradient_iter = 1
        count = 1

        fx, grad = self.getVars(x, mu)
        print(f"Problem : {self.problem.report(verbose=True)} \n x_init = {x} \nfx_init = {fx} \n")
        

        while mu_iter <= 1000:
            xt = x.copy()
            mu *= rho_mu
            
            while gradient_iter <= 1000:
                fx, grad = self.getVars(x, mu)
                count += 1
                delta = -fx/grad
                if np.linalg.norm(delta) > 0:
                    delta = delta/np.linalg.norm(delta,2)
                
                fx_n, grad_n = self.getVars(x + alpha *delta, mu)
                count += 1

                while fx_n > (fx + rho_ls * np.matmul(grad.T,(alpha * delta))):
                    alpha = rho_alpha_m * alpha
                    fx_n, grad_n = self.getVars(x + alpha *delta, mu)
                    count += 1
                
                x += alpha * delta            
                alpha *= rho_alpha_p
                
                if np.linalg.norm(alpha * delta) < theta:
                    break
                gradient_iter += 1
                
            if np.linalg.norm(xt-x) < mu_theta:
                print(f"Solution : {x}")
                print(f"Calls to program : {count}")
                print(f"Iterations over mu : {mu_iter}")
                print(f"Backtracking iterations : {gradient_iter}")
                print(f"Time elapsed : {time.time()-start_time}s")
                print("\n")
                break
                
            mu_iter += 1
        return x



class SolverSquaredPenalty(NLPSolver):

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
        assert len(self.index_f) <= 1  # at most, only one term of type OT.f
        # get all sum-of-square features
        self.index_r = [i for i, x in enumerate(types) if x == OT.sos]
        # get all inequalities features
        self.index_g = [i for i, x in enumerate(types) if x == OT.ineq]

    def getVars(self, x, mu):
        phi, J = self.problem.evaluate(x)

        c = 0
        grad = np.zeros(self.dim)

        if len(self.index_f) > 0:
            c += phi[self.index_f][0]
            grad += J[self.index_f][0]

        if len(self.index_r) > 0:
            c += phi[self.index_r].T @ phi[self.index_r]
            grad += 2 * J[self.index_r].T @ phi[self.index_r]

        if len(self.index_g) > 0:
            c_g, grad_g = 0, 0
            for i in self.index_g:
                if phi[i] > 0:
                    c_g += phi[i]**2
                    grad_g += 2*phi[i]*J[i]
            c = c + c_g * mu
            grad = grad + grad_g * mu

        return c, grad


    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        
        # write your code here
        # mu 
        mu = 1
        # increase stepsize for mu
        rho_mu = 2
        # tolerance  
        epsilon = self.kwargs.get("epsilon", 1e-6)
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

        outer_iter = 1
        gradient_iter = 1
        count = 1

        fx, grad = self.getVars(x, mu)
        print(f"Problem : {self.problem.report(verbose=True)} \n x_init = {x} \nfx_init = {fx} \n")
        

        while outer_iter <= 1000:
            xt = x.copy()
            
            while gradient_iter <= 1000:
                fx, grad = self.getVars(x, mu)
                count += 1
                delta = -fx/grad
                if np.linalg.norm(delta) > 0:
                    delta = delta/np.linalg.norm(delta,2)
                
                fx_n, grad_n = self.getVars(x + alpha *delta, mu)
                count += 1

                while fx_n > (fx + rho_ls * np.matmul(grad.T,(alpha * delta))):
                    alpha = rho_alpha_m * alpha
                    fx_n, grad_n = self.getVars(x + alpha *delta, mu)
                    count += 1
                
                x = x + alpha * delta            
                alpha *= rho_alpha_p
                
                if np.linalg.norm(alpha * delta) < theta:
                    break
                gradient_iter += 1
                
            phi, J = self.problem.evaluate(x)
            if np.linalg.norm(xt-x) < theta and np.all(phi[self.index_g] < epsilon):
                print(f"Solution : {x}")
                print(f"Calls to program : {count}")
                print(f"Iterations : {outer_iter}")
                print(f"Backtracking iterations : {gradient_iter}")
                print(f"Time elapsed : {time.time()-start_time}s")
                print("\n")
                break
                
            mu *= rho_mu
            outer_iter += 1
        return x

