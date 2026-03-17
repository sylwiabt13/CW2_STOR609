#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:17:36 2026

@author: bathetay
"""

"""
Suppose Sam wanted to make an informed decision about whether to party or relax over the weekend. 
Sam prefers to party, but is worried about getting sick. 
Such a problem can be modeled as an MDP with two states, healthy and sick, and two actions, relax and party. 

This problem can be characterised as follows
"""

S = [0,1] #Healthy or Sick
A = [0,1] #Relax or Party
P={(0,0):[0.95,0.05],(0,1):[0.7,0.3],(1,0):[0.5,0.5],(1,1):[0.1,0.9]} #State transition probabilities
R={(0,0):7,(0,1):10,(1,0):0,(1,1):2} #Reward probabilities

from value_iteration import value_iteration

results = value_iteration(S, A, P, R, 0.99, 100000)
print(results) #ALWAYS RELAX

results = value_iteration(S, A, P, R, 0.5, 100000)
print(results) #PARTY IF HEALTHY

results = value_iteration(S, A, P, R, 0.1, 100000)
print(results) #ALWAYS PARTY

"""
RESULTS will demonstrate that the chosen policy varies with the choice of gamma the discount factor
"""

#Unit test proposal as its a simple calculation

results = value_iteration(S, A, P, R, 1, 2)
print(results) #ALWAYS PARTY

