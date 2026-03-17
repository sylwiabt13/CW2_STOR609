#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:58:56 2026

@author: bathetay
"""


def value_iteration(state_space: list[str], action_space: list[str], 
                    transition_func: dict[tuple[str,str],list[float]], 
                    reward_func: dict[tuple[str,str],float], 
                    gamma: float,
                    termination: int,
                    epsilon: float = 1e-6,
                    init_value_func: dict = None 
                    )->tuple[dict[str,str], dict[str,float]]:
    """
    Parameters
    ----------
    state_space : list[str]
        STATE SPACE, a list of strings, each corresponding to a unique state.
    action_space : list[str]
        ACTION SPACE, a list of strings, each corresponding to a unique action
    transition_func : dict[tuple[str,str],list[float]]
        Dictionary of tuples as keys, corresponding to state action pairs,
        with the values being a list of probabilities over the state space. 
        That is given s,a what is the probability we move to s' .
    reward_func : dict[tuple[str,str],float]
        Dictionary of state action pairs as input, with corresponding output as the reward.
    gamma : float
        Discount factor between 0 and 1
    termination : int
        Maximal number of iterations we want to perform
    epsilon : float, optional
        For convergence checking. The default is 1e-6.
    init_value_func: dict, optional
        Optional initial value function values, the default choice is 0 for all states.

    Returns
    -------
    (policy,value_func) = (tuple[dict[str,str], dict[str,float]])
    
        #policy: outputted policy should be a dictionary with keys as states corresponding to actions, both of which are strings

        #value_func: outputted value function with keys corresponding to state space and values corresponding to expected reward of out policy

    """
    
    
    #Value function initialisation
    if init_value_func == None:
        value_func = {s: 0 for s in state_space} #If not provided with value function initialisation, try 
    else:
        value_func = init_value_func.copy()
    new_value_func = value_func.copy()
    

    def Q_func(s: str,a: str)->float:
        """
        Calculate the Q-function for state-action pair (s, a)
        """
        return reward_func[(s,a)] + gamma*sum(p * value_func[s_prime] for p, s_prime in zip(transition_func[(s, a)], state_space))
    
    #Update value function to next iteration of value function
    k = 0 
    while True:
        for s in state_space:
            #maximise the following expression, calculating the New Value Function Dictionary
            new_value_func[s] = max(Q_func(s,a) for a in action_space)
        
        #Convergence check using epsilon threshold
        if all(abs(new_value_func[s] - value_func[s]) < epsilon for s in state_space):
            break
        
        #Discard Old Value Function Dictionary
        value_func = new_value_func.copy()
        k += 1
        if k >= termination:
            break
        
        
    #Find policy corresponding to optimised value function
    policy = {s: max(action_space, key=lambda a: Q_func(s, a)) for s in state_space}
    
    return policy, value_func