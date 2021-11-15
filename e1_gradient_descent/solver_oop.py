#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:50:40 2021

@author: aboumessouer
"""

import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.nlp_solver import  NLPSolver

class FlexibleSolver(NLPSolver):

    def __init__(self, linesearch=True, backtracking=True, **kwargs):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        if backtracking and not linesearch:
            raise ValueError("linesearch is a prerequisite for backtracking.")
        self.linesearch = linesearch
        self.backtracking = backtracking
        
        # stepsize
        self._alpha = kwargs.get("alpha", 1)         
        # line search factor, generally in interval [0.01, 0.3]
        self._rho_ls = kwargs.get("rho_ls", 0.05)         
        # increase_factor
        self._rho_alpha_plus = kwargs.get("rho_alpha_plus", 1.2)        
        # line search decrease_factor
        self._rho_alpha_minus = kwargs.get("rho_alpha_minus", 0.5)        
        # tolerance  
        self._theta = kwargs.get("theta", 1e-3)
        
        self.x = None
        self.iteration = 0
    
    @property
    def gradient(self):
        _, J = self.problem.evaluate(self.x)
        return J[0]
    
    def fx(self, x):
        phi, _ = self.problem.evaluate(x)
        return phi[0]
    
    @property
    def delta(self):
        assert self.linesearch
        return -self.gradient / np.linalg.norm(self.gradient, ord=2)
    
    @property
    def step(self):        
        if self.linesearch:
            return self._alpha * self.delta 
        return -self.gradient * self._alpha
    
    @property
    def step_norm(self):
        if self.iteration == 0:
            return np.inf
        return np.sum(np.linalg.norm(self.step, ord=2))
    
    def wolfe_cond(self):
        return self.fx(self.x) + self._rho_ls * self.gradient.T @ self.step

    def solve(self):        
        
        # initialization
        # x = self.problem.getInitializationSample()
        self.x = self.problem.getInitializationSample()
        
        # fx = self.problem.evaluate(x)[0][0]
        print(f"x_init = {self.x} \nfx_init = {self.fx(self.x):.4f}")
        # step_norm = np.sum(np.linalg.norm(np.inf - x, ord=2))
        
        # iteration = 0
        while self.step_norm >= self._theta:
            # phi, J = self.problem.evaluate(x)
            # fx = phi[0]
            # gradient = J[0]            
            if self.linesearch:
                # delta = -gradient / np.linalg.norm(gradient, ord=2)
                # fx_next = self.problem.evaluate(x + self._alpha * delta)[0][0]
                
                if self.backtracking:
                    while self.fx(self.x + self.step) > self.wolfe_cond():
                        self._alpha = self._rho_alpha_minus * self._alpha
                        # fx_next = self.problem.evaluate(x + self._alpha * delta)[0][0]
                else:
                    while self.fx(self.x + self.step) > self.fx(self.x):
                        self._alpha = self._rho_alpha_minus * self._alpha
                        # fx_next = self.problem.evaluate(x + alpha * delta)[0][0]                
                print(f"alpha = {self._alpha}")
                
                # step = self._alpha * delta         
                # alpha = self._rho_alpha_plus * self._alpha
                self._alpha = 1
            
            else:
                # plain gradient descent
                # step = -gradient * self._alpha
                pass
            
            self.x = self.x + self.step
            self.iteration += 1
            # step_norm = np.sum(np.linalg.norm(step, ord=2))
            
            print(f"\niteration = {self.iteration}")
            print(f"x = {self.x}")
            print(f"fx = {self.fx(self.x):.4f}")
            print(f"gradient = {self.gradient}")  
            print(f"step = {self.step}")
            print(f"step_norm = {self.step_norm:.5f}")
            
        return self.x

