#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:46:50 2023

@author: augmdc
"""

import numpy as np

WALL_VALUE = np.inf
GOAL_REWARD = 112

def generate_Q_table(n, m, actions, rewards, gamma, alpha, iterations, epsilon):
    # Initialize Q-values to zeros (Q-table)
    q_values = np.zeros((n, m, 4))
    
    # Perform Q-learning with epsilon-greedy exploration
    for it in range(iterations):
        for i in range(n):
            for j in range(m):
                for action_index, action in enumerate(actions):
                    ni, nj = i + action[0], j + action[1]

                    # Ensure the agent stays within the grid
                    ni = max(0, min(n - 1, ni))
                    nj = max(0, min(m - 1, nj))

                    # If current state is terminal state, penalties, or rewards, 
                    # update the Q-values based on their respective rewards
                    
                    # TO-DO: identify goal state and each penalty state
                    
                    if (i, j) == (0, 1): # Goal State
                        q_values[i, j, action_index] = 10
                        continue
                    elif (i, j) == (0, 2): # Penalty States
                        q_values[i, j, action_index] = -10
                        continue
                    
                    elif False:         # Reward States
                        q_values[i, j, action_index] = 10
                        continue

                    # Explore with probability epsilon or exploit with probability 1 - epsilon
                    if np.random.rand() < epsilon:
                        action_index = np.random.randint(0, len(actions))
                        action = actions[action_index]

                    # Q-learning update rule:
                    # 1. Calculate the sample using the reward and the maximum Q-value of the next state
                    sample = rewards[i, j] + gamma * np.max(q_values[ni, nj])

                    # 2. Update Q-value using the Q-learning update rule (weighted average)
                    q_values[i, j, action_index] = (1 - alpha) * q_values[i, j, action_index] + alpha * sample

                    # Clip the Q-value to a maximum of 10 and a minimum of -10???
                    if i == 0 and j == 1:
                        q_values[i, j, action_index] = min(q_values[i, j, action_index], 10.0)
                    elif i == 0 and j == 2:
                        q_values[i, j, action_index] = max(q_values[i, j, action_index], -10.0)
    return q_values

    def generate_optimal_policy():
        return 1
    
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Discount factor gamma determines the agent's consideration for future rewards.
gamma = 1.0

n, m = 4, 3

# Rewards grid setup
rewards = np.array([
    [0, 10, -10],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

# Learning rate alpha for Q-learning
alpha = 0.5

# Number of iterations
iterations = 10

# Exploration rate (epsilon) for epsilon-greedy straetegy
epsilon = 0.1

q_values = generate_Q_table(n , m, actions, rewards, gamma, alpha, iterations, epsilon)

# Display final Q-values and corresponding actions for all states and paths
print(f"Final Q-values after {iterations} iteration(s)")
for i in range(n):
    for j in range(m):
        for action_index, action in enumerate(actions):
            print(f"State ({i},{j}): {action} Q={q_values[i, j, action_index]:.2f}")
        print("-----------------")

