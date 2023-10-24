#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:56:06 2023

@author: augmdc
"""

# Mandatory imports
import numpy as np

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

# Possible actions that the agent can take: move left, up, right, or down.
actions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

# Discount factor gamma determines the agent's consideration for future rewards.
# A value close to 1 makes the agent consider future rewards as important as immediate ones.
gamma = 0.9

# Noise probability: With a certain probability, the agent doesn't move in the intended direction.
# Instead, it may move in some other random direction.
noise_prob = 0.2

# Value Iteration process
iterations = 100
for it in range(iterations):
    # Temporary matrix to hold the updated values during this iteration.
    new_values = np.zeros((n, m))

    # For each cell
    for i in range(n):
        for j in range(m):
            # If the current cell is a terminal (absorbing) state (goal or penalty state),
            # its value remains constant.
            if rewards[i][j] == 10 or rewards[i][j] == -10:
                new_values[i][j] = rewards[i][j]
                continue

            # Calculate the maximum expected value for the current state by considering all possible actions.
            cell_values = []
            for action in actions: # For each intended action
                expected_value = 0

                for possible_action in actions: # Four possible outcomes
                    if possible_action == action: # This is the probability that the intended action is executed successfully
                      transition_prob = 1 - noise_prob # if noise_prob is 0.2, there's an 80% chance the intended action will occur.
                    else:
                       transition_prob = noise_prob/3 #  there's an equal chance it could be any of the other three actions.
                    # computing the next state (ni, nj) given the current state (i, j) and a chosen action.
                    ni, nj = i + possible_action[0], j + possible_action[1]

                    # If taking the action keeps the agent inside the grid, use the value of the destination cell.
                    # Otherwise, the agent "bounces" back, so use the current cell's value.
                    # V_(k+1) for each possible_action: Q*(s) values
                    if 0 <= ni < n and 0 <= nj < m:
                        expected_value += transition_prob * gamma * (living_reward + values[ni][nj])
                    else: # If the action results in hitting the wall, bounce back.
                        expected_value += transition_prob * gamma * (living_reward + values[i][j])

                cell_values.append(expected_value)
                # OR Combine living reward and expected value.
                # cell_values.append(living_reward + expected_value)

            # The Bellman equation: the value of a cell is the maximum expected reward of all possible actions.
            new_values[i][j] = max(cell_values)

    # Update the value function with the new computed values.
    values = new_values

    # Display the value function for this iteration.
    print(f"Iteration {it + 1}")
    for row in values:
        print(" ".join(f"{x:7.2f}" for x in row))
    print("-----------------------------")

# Compute the policy based on the final state values
final_policy = np.zeros((n, m), dtype=object)  # Change dtype to object to allow strings and numbers

for i in range(n):
    for j in range(m):
        if rewards[i][j] == 10 or rewards[i][j] == -10:
            final_policy[i][j] = "*"
            continue

        # For each action, compute the resulting state and its value
        action_values = []
        for action in actions:
            ni, nj = i + action[0], j + action[1]
            if 0 <= ni < n and 0 <= nj < m:
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
print(symbolic_policy)