#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:02:22 2026

@author: bathetay
"""

S = ['TL', 'TR', 'BL', 'BR']  # STATE SPACE
A = ['R', 'L', 'D', 'U']     # ACTION SPACE
P = { #TRANSITION PROBABILITIES
    ('TL', 'R'): [0.9, 0.1, 0.0, 0.0],
    ('TL', 'D'): [0.1, 0.0, 0.9, 0.0],
    ('TR', 'L'): [0.9, 0.0, 0.1, 0.0],
    ('TR', 'D'): [0.2, 0.0, 0.0, 0.8],
    ('BL', 'R'): [0.1, 0.0, 0.9, 0.0],
    ('BL', 'U'): [0.8, 0.0, 0.2, 0.0],
}
R = { #REWARD FUNCTION
    ('TL', 'R'): -1,
    ('TL', 'D'): -2,
    ('TR', 'L'): -1.5,
    ('TR', 'D'): 15,
    ('BL', 'R'): 20,
    ('BL', 'U'): -0.5,
}

from value_iteration import value_iteration

results = value_iteration(S, A, P, R, 0.99, 100000)
print(results) 

#ERROR need to investigate