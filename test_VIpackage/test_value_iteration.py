#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:29:30 2026

@author: bathetay
"""

from VIpackage import value_iteration

S = ['Healthy','Sick'] #Healthy or Sick
A = ['Relax','Party'] #Relax or Party
P={('Healthy','Relax'):[0.95,0.05],('Healthy','Party'):[0.7,0.3],('Sick','Relax'):[0.5,0.5],('Sick','Party'):[0.1,0.9]} #State transition probabilities
R={('Healthy','Relax'):7,('Healthy','Party'):10,('Sick','Relax'):0,('Sick','Party'):2} #Reward probabilities


def test_vi_basic():
    policy, values = value_iteration(S, A, P, R, 1, 2)
    assert policy == {'Healthy': 'Party', 'Sick': 'Relax'}
    assert values == {'Healthy': 17.6, 'Sick': 6.0}

def test_vi_gamma():
    policy, values = value_iteration(S, A, P, R, 0.9, 2)
    assert policy == {'Healthy': 'Party', 'Sick': 'Relax'}
    assert values == {'Healthy': 10.0+7.6*0.9, 'Sick': 6.0*0.9}
    

    