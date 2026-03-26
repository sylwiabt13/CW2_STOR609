# CW2_STOR609 Value Iteration for Markov Decision Processes

An implementation of the Value Iteration algorithm for Markov Decision Processes, for the purposes of coursework for STOR609. 

## Package installation/Unit testing

The package can be installed by running the following:

```
python -m pip install 'git+https://github.com/sylwiabt13/CW2_STOR609.git'
```

One can verify the package is functioning correctly with `pytest`.

## Explanation of Markov Decision Processes and the Value Iteration Algorithm

In general a Markov Decision Process will have the following characteristics:

In general, a Markov Decision Process will have the following characteristics:

* A state space $\mathcal{S}$.
* An action space $\mathcal{A}$.
* A function $P: \mathcal{S} \times \mathcal{A} \longrightarrow \mathcal{S}$ characterising the transition probabilities given a state-action pair.
* A function $\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \longrightarrow \mathbb{R}$ characterising the reward for entering a state given the previous state-action pair.

For the Value Iteration algorithm you only strictly need the expected reward given a state action pair $R: \mathcal{S} \times \mathcal{A} \longrightarrow \mathbb{R}$ but this can be determined easily by observing:

$$
R(s,a)= \sum_{\tilde{s}} \mathcal{R}(s,a,\tilde{s})P(\tilde{s} \mid s,a)
$$

The value iteration algorithm, seeks to identify a policy $\pi: \mathcal{S} \longrightarrow \mathcal{A}$, that will maximise some notion of a discounted future cumulative reward. We characterise this using the following two functions:

* The Value function $V^{\pi}: \mathcal{S} \longrightarrow \mathbb{R}$, characterises the expected discounted future cumulative reward given a policy $\pi$, and given we are in a state $s\in\mathcal{S}$.
* The $Q$ function $Q^{\pi}: \mathcal{S} \times \mathcal{A} \longrightarrow \mathbb{R}$, characterises this expected reward given a policy $\pi$, and given the state-action pair $(s,a)\in\mathcal{S}\times\mathcal{A}$.

These are defined recursively in terms of one another:

$$
Q^{\pi}(s,a) = R(s,a) + \gamma \sum\_{\tilde{s}}  P(\tilde{s}\mid s,a) V^{\pi}(\tilde{s})
$$

$$
V^{\pi}(s) = Q^{\pi}(s,\pi(s))
$$

Value iteration allows us to start with an initial value function value, and iterate through these equations until they converge. The optimal policy can then be calculated by identifying for a given state, which action maximises the final iteration of the $Q$ function.

## Pseudocode for Value Iteration Algorithm

```
DEFINE VALUE_ITERATION FUNCTION
	"""
	    Parameters/Inputs
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

	IF init_value_func IS UNDEFINED
	
		SET IT TO A DEFAULT VALUE OF 0 FOR ALL STATES
		
	DEFINE Q_func
	
		CALCULATE Q-FUNCTION GIVEN REWARD FUNCTION, VALUE FUNCTION, TRANSITION PROBABILITIES
		
	k = 0
	
	WHILE TRUE:
	
		FOR EVERY STATE s:
		
			CALCULATE Q_Values USING Q_func FOR ALL VALID ACTIONS
			
			NEW VALUE FUNCTION DICTIONARY AT STATE S = max(Q_Values)
			
		ENDFOR
			
		IF NEW VALUE FUNCTION WITHIN epsilon OF OLD VALUE FUNCTION BREAK
		
		DISCARD OLD VALUE FUNCTION DICTIONARY AND STORE NEW VALUE FUNCTION
		
		k += 1
		
		IF k >= termination BREAK 
	
	ENDWHILE
	
	CALCULATE OPTIMAL POLICY
	
	RETURN OPTIMAL POLICY AND VALUE FUNCTION
		
```

The pseudocode here is a more verbose version of the Pseudocode provided in Figure 9.16 


## References

Poole, D.L. and Mackworth, A.K. (2023) Artificial Intelligence: Foundations of Computational Agents. 3rd edn. Cambridge: Cambridge University Press. 


