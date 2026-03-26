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

The value iteration algorithm, seeks to identify a policy $\pi: \mathcal{S} \longrightarrow \mathcal{A}$, that will maximise some notion of a discounted future cumulative reward.

We let $V^{\pi}_k (s)$ denote the discounted future cumulative reward we get under policy $\pi$ initialised in a state $s \in \mathcal{S}$, then we have that:

$$
V^{\pi}_k (s) = \sum\_{t=k}^\infty \gamma^{t-k}r\_k
$$

## Pseudocode for Value Iteration Algorithm

## References

Poole, D.L. and Mackworth, A.K. (2023) Artificial Intelligence: Foundations of Computational Agents. 3rd edn. Cambridge: Cambridge University Press. 


