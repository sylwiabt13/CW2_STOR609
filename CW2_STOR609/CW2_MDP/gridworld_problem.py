#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:02:22 2026

@author: bathetay
"""

"""
A grid world is an idealization of a robot in an environment. 
At each time, the robot is at some location and can move to neighboring locations, 
collecting rewards and punishments. Suppose that the actions are stochastic, 
so that there is a probability distribution over the resulting states given the action and the state.
"""

S = ['TL', 'TR', 'BL', 'BR']  # STATE SPACE
A = ['R', 'L', 'D', 'U']     # ACTION SPACE


P = { #TRANSITION PROBABILITIES
    ('TL', 'R'): [0, 0.9, 0.1, 0],
    ('TL', 'D'): [0, 0.1, 0.9, 0],
    ('TR', 'L'): [0.9, 0, 0, 0.1],
    ('TR', 'D'): [0.2, 0, 0, 0.8],
    ('BL', 'R'): [0.1, 0, 0, 0.9],
    ('BL', 'U'): [0.8, 0, 0, 0.2]
} #State BR is terminal, so there is no associated probability vector

RC = { #CONDITIONAL REWARDS: conditional reward is 0 when there is no given reward, the probability 
      ('TL', 'R'): [0,-1,-2,0],
      ('TL', 'D'): [0, -1, -2, 0],
      ('TR', 'L'): [-1.5, 0, 0, 10],
      ('TR', 'D'): [-1, 0, 0, 15],
      ('BL', 'R'): [-2.5, 0, 0, 20],
      ('BL', 'U'): [-0.5, 0, 0, 5]
}

R = {}

# Calculate dot products for each state-action pair
for state_action in P.keys():
    prob_vector = P[state_action]
    reward_vector = RC[state_action]  # Default to [0,0,0,0] if no reward
    dot_product = sum(p*r for p,r in zip(prob_vector,reward_vector)) 
    R[state_action] = dot_product
    

from VIpackage import value_iteration

results = value_iteration(S, A, P, R, 0.9, 1000)
print(results) 

