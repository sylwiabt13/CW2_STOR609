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

S = ['Healthy','Sick'] #Healthy or Sick
A = ['Relax','Party'] #Relax or Party
P={('Healthy','Relax'):[0.95,0.05],('Healthy','Party'):[0.7,0.3],('Sick','Relax'):[0.5,0.5],('Sick','Party'):[0.1,0.9]} #State transition probabilities
R={('Healthy','Relax'):7,('Healthy','Party'):10,('Sick','Relax'):0,('Sick','Party'):2} #Reward probabilities

from VIpackage import value_iteration

results = value_iteration(S, A, P, R, 0.99, 1000)
print(results) #ALWAYS RELAX

results = value_iteration(S, A, P, R, 0.5, 1000)
print(results) #PARTY IF HEALTHY

results = value_iteration(S, A, P, R, 0.1, 1000)
print(results) #ALWAYS PARTY

"""
RESULTS will demonstrate that the chosen policy varies with the choice of gamma the discount factor
"""

#Unit test proposal as its a simple calculation

results = value_iteration(S, A, P, R, 1, 2)
print(results) #ALWAYS PARTY

