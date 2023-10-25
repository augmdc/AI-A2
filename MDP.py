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
# If reward (+10), apply discount
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
# Computes the probability of moving from state s to state s' when performing action
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

# Returns a grid with each cell with it's values
def value_iteration(world, iterations, rewards, actions, values):
    n = world.size
    for it in range(iterations):
        # Temporary matrix to hold the updated values during this iteration.
        new_values = np.zeros((n, n))
        # For each cell
        for i in range(n):
            for j in range(n):
                # If the current cell is a terminal (absorbing) state (goal or penalty state),
                # its value remains constant.
                print("hello world")
        
def final_policy():
    return 1