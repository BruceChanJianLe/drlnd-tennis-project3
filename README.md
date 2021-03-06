# Deep Reinforcement Learning - Tennis - Project 3

This repository stores the solution to the udacity deep reinforcement learning nanodegree third project which is the tennis project.

## Project Overview

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

**Aim**  
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

**State Space**  
The state space is 24 which are variables corresponding to the position and velocity of the ball and racket.

**Action Space**  
The action space is 2 continuous actions, corresponding to movement toward (or away from) the net, and jumping.

## Getting Started

Please make sure `python3.6` is installed in your conda environment. And run the install script.

```bash
source ./install.sh
```

## Running

Please run `Tennis.ipynb` jupyter notebook. You will need to be able to connect to the jupyter notebook server before you can run it.

## Local Setup

Download unity program for tennis here or use the one provided in the repository.

   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
