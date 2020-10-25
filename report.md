# Report

This report describes my implementation to solve the tennis project.  

## Learning Algorithm

The implementation here is a Multi-Agent Deep Deterministic Policy Gradient Algorithm. This implementation is chosen as the previous project was sloved by DDPG, therefore, continueing from there the MADDPG is used. The advantages of using this algorithm is that `all agents can share the same replay buffer` and `the critic is able to learn from the totality of the states and actions taken by each actor`. 

## Tennis

DDPG is a type of actor-critic methods which are applicable to continuous state space action. For more information about DDPG please read this [article](https://arxiv.org/pdf/1509.02971.pdf). DDPG trains simultaneously two networks: An actor (selects the optimal (deterministic) policy based on the current state) and a critic (approximates the value function of the state-action pair).

### Actor