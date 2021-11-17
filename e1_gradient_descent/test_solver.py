import numpy as np
import sys
sys.path.append("..")
# import the test classes
from optimization_algorithms.interface.mathematical_program_traced import  MathematicalProgramTraced
from optimization_algorithms.mathematical_programs.quadratic_identity_2 import QuadraticIdentity2
from e1_gradient_descent.problems import FSQ, FHOLE
# from solution import Solver0
from e1_gradient_descent.solver import FlexibleSolver
# from e1_gradient_descent.solver_oop import FlexibleSolver
from plotting.plotter import plotFunc


class TestSolver:
    """
    test on problem A
    """
    
    def __init__(self, problem, solver):
        self.problem = problem
        self.solver = solver

    def test_convergence(self, plot=True):
        """
        check that student solver converges
        """
        assert self.problem and self.solver
        
        self.solver.setProblem(self.problem)
        output =  self.solver.solve()
        last_trace = self.problem.trace_x[-1]
        # check that we have made some progress toward the optimum
        dim = self.problem.mathematical_program.getDimension()
        assert np.linalg.norm(np.zeros(dim) - last_trace) < 0.9
        
        if plot and dim <=2:
            self.plot_func()
            
    def plot_func(self):
        
        def func(x):
            """Closure to only return function value."""
            return self.problem.mathematical_program.evaluate(x)[0][0]
            
        trace_x = np.array(self.problem.trace_x)
        trace_phi = np.array(self.problem.trace_phi)
        plotFunc(func, bounds_lo=[-2,-2], bounds_up=[2,2], trace_xy = trace_x, trace_z = trace_phi)
        
    def plot_trace(self):
        trace_phi = np.array(self.problem.trace_phi)
        # fig = plt.figure(figsize=(10,7))
        # plt.plot(trace_phi)
        pass


if __name__ == "__main__":
    
    # 1. Choose Problem ------------------------------------------------------
    
    # problem: QuadraticI dentity ---
    # problem = MathematicalProgramTraced(QuadraticIdentity2())
       
    # problem: FSQ ---
    c = 10
    problem = MathematicalProgramTraced(FSQ(c, dim=2))
    
    # problem: FHOLE ---
    # c = 10
    # a = 0.1
    # problem = MathematicalProgramTraced(FHOLE(c,a, dim=2))

    # 2. Choose Solver -------------------------------------------------------
    
    # plain gradient descent ---
    # solver = FlexibleSolver(linesearch=False, backtracking=False, alpha=0.01)
       
    # gradient descent + linesearch ---
    # solver = FlexibleSolver(linesearch=True, backtracking=False, alpha=1)
    # solver = FlexibleSolver(linesearch=True, backtracking=False, alpha=1, metric=problem.mathematical_program.C)
       
    # gradient descent + linesearch + backtracking ---
    solver = FlexibleSolver(linesearch=True, backtracking=True, alpha=1)
    # solver = FlexibleSolver(linesearch=True, backtracking=True, alpha=1, metric=problem.mathematical_program.C)
       
    # 3. Optimize ------------------------------------------------------------
    
    TestSolver(problem, solver).test_convergence()
    # unittest.main()


