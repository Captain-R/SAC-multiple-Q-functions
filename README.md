# SAC-multiple-Q-functions

This repository contains the code for the both versions of the SAC algorithm. It also has the feature to use n Q-functions simultaneously to tackle overestimation bias.

This algorithm works perfectly fine on the Pendulum-v0 and the MountainCarContinuous-v0 environments in openAI Gym. To use for other environments changes must be made in some places in the code.

The code is run using a main function that initialises the agents. Atleast two agents, one with update able temperature and one without are initialised. The k variable conrols how many additional agents are required, with them having k+2 Q functions.

The u variable controls whether the temperature is update able or not with u = 1 implying that the agent will update temperature.

Moreover, tthe target Q network updates can be done in either of two ways, batch updates and Polyak averaging. This is ocntrolled using the beta and the tau parameters. Bets controls the number of update steps after which the target Q network will be updated. For Polyak averaging set beta to 1. Tau controls the fraction of update to target networks in the Polyak averaging case. Set it to 1 in case of batch updates.

Maxeps is the number of episodes for whihc the algorithm will run. Maxsteps is the upper cap on the number of steps per episode and batchsize is the size of samples that the agent will use to make updates in one step.

buf parameter is the number of actions that will be sampled at random from the environment at the start of each episode. This is done since SAC is off policy algorithm. Moreover, reward calculations are only done if the action was sampled by the agent.

All other hyperparameters are set up as recommended by Haarnoja et al. in their paper introducing SAC and its variant with update able temperature
