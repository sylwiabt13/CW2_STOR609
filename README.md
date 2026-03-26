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

* A state space $\mathcal S$.
* An action space $\mathcal A$.
* A function $P: \mathcal S \times \mathcal A \longrightarrow \mathcal S $ characterising the transition probabilities given a state-action pair.
* A function $R: $ S \times \mathcal A \longrightarrow \mathcal S $

## Pseudocode for Value Iteration Algorithm

## References

Poole, D.L. and Mackworth, A.K. (2023) Artificial Intelligence: Foundations of Computational Agents. 3rd edn. Cambridge: Cambridge University Press. 


