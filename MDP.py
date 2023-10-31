#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:36:20 2023

@author: augmdc
"""

import numpy as np

WALL_VALUE = np.inf
GOAL_REWARD = 112

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

def compute_difference(arr1, arr2):
    if len(arr1) != len(arr2) or len(arr1[0]) != len(arr2[0]):
        raise ValueError("Input arrays must have the same dimensions.")

    n = len(arr1)
    m = len(arr1[0])

    total_difference = 0

    for i in range(n):
        for j in range(m):
            total_difference += abs(arr1[i][j] - arr2[i][j])

    average_difference = total_difference / (n * m)
    
    return average_difference

# Equivalent of V*(s)
# Returns a grid with each cell representing values
def value_iteration(n, m, actions, rewards, values, noise_prob, living_reward):
    values = np.zeros((n, m))
    
    # NEXT SECTION CAN BE DELETED IN FUTURE
    # Experiment: put in rewards and penalties into values array
    """
    for i in range(n):
        for j in range(m):
            if rewards[i, j] == -10 or rewards[i, j] == 10:
                values[i, j] = rewards[i, j]
    """
    # Experiment: put in rewards and penalties into values array        
    
    # While the change between iterations is less than 0.0001, run the value iteration process
    difference= np.inf
    it = 0
    while difference > 0.000001:
        # Temporary matrix to hold the updated values during this iteration.
        old_values = values
        
        new_values = np.zeros((n, m))
        # For each cell
        for i in range(n):
            for j in range(m):
                # If the current cell is a terminal state, penalty, reward or wall, keep it
                if rewards[i, j] == GOAL_REWARD or rewards[i, j] == -10 or rewards[i, j] == 10:
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
                        
                        # Reward function is already implemented here
                        if 0 <= ni < n and 0 <= nj < m and values[ni, nj] !=  WALL_VALUE:
                            expected_value += prob["transition_prob"] * 0.9 * (living_reward + values[ni, nj])
                            
                        else: # If the action results in hitting the wall/edge of the map, bounce back
                            expected_value += prob["transition_prob"] * 0.9 * (living_reward + values[i, j])
                        
                    cell_values.append(expected_value)
                    
                # The Bellman equation: the value of a cell is the maximum expected reward of all possible actions.
                new_values[i, j] = max(cell_values)
                
        # Update the value function with the new computed values.
        values = new_values
        
        # Compare the old_values to the iterated ones
        difference = compute_difference(old_values, values)
        
        it += 1
        # Display the value function for this iteration.
        print(f"Iteration {it + 1}")
        for row in values:
            print(" ".join(f"{x:7.2f}" for x in row))
        print("-----------------------------")
    return values
        
def final_policy(n, m, rewards, values, actions):
    # Compute the policy based on the final state values
    final_policy = np.zeros((n, m), dtype=object)  # Change dtype to object to allow strings and numbers

    for i in range(n):
        for j in range(m):
            # If terminal state
            if rewards[i][j] == GOAL_REWARD: #rewards[i][j] == 10 or rewards[i][j] == -10
                final_policy[i][j] = "*"
                continue

            # For each action, compute the resulting state and its value
            action_values = []
            for action in actions:
                ni, nj = i + action[0], j + action[1]
                if 0 <= ni < n and 0 <= nj < m and values[ni, nj] != np.inf:
                    action_values.append(values[ni][nj])
                else:  # boundary case, stay in the same state
                    action_values.append(values[i][j])

            # Choose the action that leads to the highest state value
            best_action_idx = np.argmax(action_values)
            final_policy[i][j] = best_action_idx

    # Convert numeric policy to direction symbols, leaving "*" as is
    directions_map = {0: "←", 1: "↑", 2: "→", 3: "↓", "*": "*"}
    symbolic_policy = np.vectorize(directions_map.get)(final_policy)

    # Print the final symbolic policy
    return symbolic_policy

"""
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
living_reward = -1

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

final_values = value_iteration(4, 3, actions, rewards, values, noise_prob, living_reward)
print(final_policy(n, m, rewards, final_values))
"""


