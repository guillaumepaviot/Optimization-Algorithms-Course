import numpy as np
import sys
sys.path.append("..")
# import the test classes
from problem import ConstrainedProblem
from optimization_algorithms.interface.mathematical_program_traced import  MathematicalProgramTraced
from plotting.plotter import plotFunc
from solver import SolverInteriorPoint as SIP
from solver import SolverSquaredPenalty as SQP


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
        print(np.linalg.norm(np.zeros(dim) - last_trace) < 0.9)
        
        self.plot_func()
            
    def plot_func(self):
        
        def func(x):
            """Closure to only return function value."""
            return self.problem.mathematical_program.evaluate(x)[0][0]
            
        trace_x = np.array(self.problem.trace_x)
        trace_phi = np.array(self.problem.trace_phi)
        plotFunc(func, bounds_lo=[-2,-2], bounds_up=[2,2], trace_xy = trace_x, trace_z = trace_phi)
        

if __name__ == "__main__":
    
    # 1. Choose Problem ------------------------------------------------------
    
    problem = MathematicalProgramTraced(ConstrainedProblem(), max_evaluate=10000000000000000)

    # 2. Choose Solver -------------------------------------------------------
    
    #solver = SIP()
    solver = SQP()
       
    # 3. Optimize ------------------------------------------------------------
    
    TestSolver(problem, solver).test_convergence()
    # unittest.main()


