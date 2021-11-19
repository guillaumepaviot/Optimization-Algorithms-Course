import numpy as np
import sys
sys.path.append("..")
# import the test classes
from optimization_algorithms.interface.mathematical_program_traced import  MathematicalProgramTraced
from optimization_algorithms.mathematical_programs.quadratic_identity_2 import QuadraticIdentity2
from e1_gradient_descent.problems import FSQ, FHOLE
from e3_approximate_newton_method.problems import Rastrigin
from e3_approximate_newton_method.solver import FlexibleSolver


class CompareSolver:
    """
    test on problem A
    """
    
    def __init__(self, problem, solver):
        self.problem = problem
        self.solver = solver
        self.iterations = []

    def mean_convergence(self, samples):
        """
        check that student solver converges
        """
        assert self.problem and self.solver

        for _ in range(samples):
            self.solver.setProblem(self.problem)
            output, iteration =  self.solver.solve()
            last_trace = self.problem.trace_x[-1]
            # check that we have made some progress toward the optimum
            dim = self.problem.mathematical_program.getDimension()

            self.iterations.append(iteration)
        print(sum(self.iterations)/len(self.iterations))
        
        

if __name__ == "__main__":
    
    # 1. Choose Problem ------------------------------------------------------
    
    # problem: Quadratic Identity ---
    # problem = MathematicalProgramTraced(QuadraticIdentity2())
       
    # problem: FSQ ---
    #c = 10
    #problem = MathematicalProgramTraced(FSQ(c, dim=2))
    
    # problem: FHOLE ---
    # c = 10
    # a = 0.1
    # problem = MathematicalProgramTraced(FHOLE(c,a, dim=2))
    
    # problem: Rastrigin ---
    c = 3
    a = 4
    problem = MathematicalProgramTraced(Rastrigin(c,a, dim=2, random=True))

    # 2. Choose Solver -------------------------------------------------------
    
    # plain gradient descent ---
    solver = FlexibleSolver(method="gd", alpha=0.01)
       
    # gradient descent + linesearch ---
    # solver = FlexibleSolver(method="ls")
    # solver = FlexibleSolver(method="ls", metric=problem.mathematical_program.C)
       
    # gradient descent + linesearch + backtracking ---
    # solver = FlexibleSolver(method="lsbt")
    # solver = FlexibleSolver(method="lsbt", metric=problem.mathematical_program.C)
    
    # newton method ---
    # solver = FlexibleSolver(method="newton")
       
    # 3. Compare ------------------------------------------------------------
    
    CompareSolver(problem, solver).mean_convergence(samples=500)
    # unittest.main()


