# Report

This report describes my implementation to solve the tennis project.  

## Learning Algorithm

The implementation here is a Multi-Agent Deep Deterministic Policy Gradient Algorithm. This implementation is chosen as the previous project was sloved by DDPG, therefore, continueing from there the MADDPG is used. The advantages of using this algorithm is that `all agents can share the same replay buffer` and `the critic is able to learn from the totality of the states and actions taken by each actor`. 

## Tennis

DDPG is a type of actor-critic methods which are applicable to continuous state space action. For more information about DDPG please read this [article](https://arxiv.org/pdf/1509.02971.pdf). DDPG trains simultaneously two networks: An actor (selects the optimal (deterministic) policy based on the current state) and a critic (approximates the value function of the state-action pair).

### Actor

The actor neural network consist of two hidden layer with ReLu function as activation function. The input layer has 550 neurons. The first hidden layer has 300 neurons and the second hidden layer has 130 neurons. The output layer has 4 neurons.

Batch normalization is performed after the first hidden layer and a tanh function is applied at the output layer so that the result can be in between [-1, 1].

The actor also add noise to its action as a Ornstein-Ulenbeck process with mu(0), theta(1) and sigma(0.15). Note that noise decays with a factor of 0.95 after each episode.  
