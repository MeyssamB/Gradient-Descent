# -*- coding: utf-8 -*-
"""
Created on Thr Nov 28 2019

@author: bagheri

using Gradient Descent for finding minimum of f(x,y)
"""


import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self,problem,derivative):
        self.problem=problem
        self.derivative=derivative
        
    def findOpt(self,min_val,max_val,max_iters,init_state,alpha,error,maximize=True):
        fitness_curve =[]
        curr_f=self.problem(init_state)
        fitness_curve.append(curr_f)
        prev_f=1
        curr_state=init_state
        itr=0
        if maximize:
            while itr<max_iters and np.abs(prev_f-curr_f)>error:
                prev_f=curr_f
                prev_state=curr_state
                d=self.derivative(prev_state)
                for v in range(0,len(prev_state)):
                    curr_state[v]=prev_state[v]+d[v]*alpha
                    if curr_state[v]>max_val or curr_state[v]<min_val:
                        break
                curr_f=self.problem(curr_state)
                itr=itr+1
        if not(maximize):
            while itr<max_iters and np.abs(prev_f-curr_f)>error:
                prev_f=curr_f
                prev_state=curr_state
                d=self.derivative(prev_state)
                for v in range(0,len(prev_state)):
                    curr_state[v]=prev_state[v]-d[v]*alpha
                curr_f=self.problem(curr_state)
                fitness_curve.append(curr_f)
                itr=itr+1
        return curr_state,fitness_curve  
            
def myProblem(state):
    return ((state[0]-3)*(state[0]-3))+((state[1]-2)*(state[1]-2))

def myProblemDerivative(state):
    return 2*(state[0]-3),2*(state[1]-2)


def main():
    gd=GradientDescent(myProblem,myProblemDerivative)
    best_state,fitness_curve=gd.findOpt(min_val=0,max_val=6,max_iters=200,init_state=[5.00,5.00],alpha=0.1,error=0.01,maximize=False)
    print("best state= ", best_state)
    plt.plot(fitness_curve)
    plt.ylabel('Fitness')
    plt.xlabel('Iteration')
    plt.show()
    
main()