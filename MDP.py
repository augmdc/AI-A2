#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:36:20 2023

@author: augmdc
"""

import numpy as np

# Equivalent of R(s, a ,s')
# Computes the reward of moving from one state to another
# state is a tuple indicating x, y coordinates
# Edge cases: if next state is either a wall or the border, reward function returns the living reward
def reward_function(world, state, action, steps):
    n = world.size
    if world[state[0] + action[0], state[1] + action[1]] != -np.inf and (state[0] + action[0] != n-1 or state[0] + action[0] != -1) and (state[1] + action[1] != n-1 or state[1] + action[1] != -1):
        if world[state[0] + action[0], state[1] + action[1]] == 10: # If reward, apply discounting
            return world[state[0] + action[0], state[1] + action[1]] - 1 * (0.9 ** steps)
        else:
            return world[state[0] + action[0], state[1] + action[1]] - 1
    else:
        return -1 # Only the living reward returned if it is a wall or the edge of the map

# Equivalent of T(s, a, s')
# Computes the probability of moving from state s to state s' when performing an action
# Returns a list that maps an action to it's probability
def transition_probability(all_actions, noise_prob, action_performed):
    probs = []
    for possible_action in all_actions: # Four possible outcomes
        if possible_action == action_performed: # This is the probability that the intended action is executed successfully
            transition_prob = 1 - noise_prob # if noise_prob is 0.2, there's an 80% chance the intended action will occur.
        else:
            transition_prob = noise_prob/3 #  there's an equal chance it could be any of the other three actions.
        probs.append({"action": possible_action, "transition_prob": transition_prob})
    return probs

# Equivalent of V*(s)
# Returns a grid with each cell representing values
def value_iteration(n, m, iterations, actions, rewards, values, noise_prob, gamma, living_reward):
    values = np.zeros((n, m))
    
    # NOTE: ITERATIONS IS JUST FOR TESTING. SHOULD ONLY STOP WHEN CHANGES TO V ARE <0.001
    for it in range(iterations):
        # Temporary matrix to hold the updated values during this iteration.
        new_values = np.zeros((n, m))
        # For each cell
        for i in range(n):
            for j in range(m):
                # If the current cell is a terminal state, penalty, reward or wall
                if rewards[i, j] == 10 or rewards[i, j] == -10: #or rewards[i, j] == 100 or rewards[i, j] == -np.inf:
                    new_values[i, j] = rewards[i, j]
                    continue
                    
                # Calculate the maximum expected value for the current state by considering all possible actions.
                cell_values = []
                for action in actions:
                    expected_value = 0
                    probs = transition_probability(actions, noise_prob, action)
                    
                    for prob in probs:
                        # For each action-action pair, calculate the Q*(s, a) values
                        ni, nj = i + prob["action"][0], j + prob["action"][1]
                        
                        if 0 <= ni < n and 0 <= nj < m:
                            expected_value += prob["transition_prob"] * gamma * (living_reward + values[ni, nj])
                        else: # If the action results in hitting the wall, bounce back
                            expected_value += prob["transition_prob"] * gamma * (living_reward + values[i, j])
                        
                    cell_values.append(expected_value)
                    
                # The Bellman equation: the value of a cell is the maximum expected reward of all possible actions.
                new_values[i, j] = max(cell_values)
                
        # Update the value function with the new computed values.
        values = new_values

        # Display the value function for this iteration.
        print(f"Iteration {it + 1}")
        for row in values:
            print(" ".join(f"{x:7.2f}" for x in row))
        print("-----------------------------")
        
def final_policy():
    return 1

# Grid size
n, m = 4, 3

# Rewards grid setup
# This grid provides immediate rewards for each cell. The agent receives +10 for reaching one cell, -10 for another, and 0 elsewhere.
rewards = np.array([
    [0, 10, -10],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])

# Living reward (or penalty) is a cost for each move the agent makes.
# This encourages the agent to find a goal state as quickly as possible.
living_reward = 0

# Initialize the value function to zeros. This matrix holds the expected cumulative rewards
# for each state under the current policy.
values = np.zeros((n, m))

# Discount factor gamma determines the agent's consideration for future rewards.
# A value close to 1 makes the agent consider future rewards as important as immediate ones.
gamma = 0.9

# Noise probability: With a certain probability, the agent doesn't move in the intended direction.
# Instead, it may move in some other random direction.
noise_prob = 0.2

# Possible actions that the agent can take: move left, up, right, or down.
actions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

iterations = 100

print(transition_probability(actions, 0.2, (0 ,1)))

#for prob in probs:
    #print(prob["action"][0])

probs = value_iteration(4, 3, iterations, actions, rewards, values, noise_prob, gamma, living_reward)
