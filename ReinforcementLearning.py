#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:46:50 2023

@author: augmdc
"""

import numpy as np

WALL_VALUE = np.inf
GOAL_REWARD = 112

# Moves robot multiple times for an "episode"
def Q_learning(world, rewards, episodes, actions, gamma, epsilon, alpha, living_reward, step_var):
    q_values = np.zeros((10, 10, 4))
    
    # Every iteration is an episode
    # Decrease epsilon after every episode
    min_epsilon = 0.1
    decay_rate = 0.001

    for episode in range(episodes):
        steps = 0
        i, j = world.robot_pos # Capture initial position
        while steps < step_var: 
            random_value = np.random.rand()
            #Exploration
            if random_value < epsilon: 
                action_index = np.random.randint(0, len(actions))
                print("Explore")
           #Exploitation
            else:
                print("Exploit")
                action_index = np.argmax(q_values[i, j])
                
            actual_action = actions[action_index]
            
            pi, pj = i, j # Past position
            i, j = i + actual_action[0],  j + actual_action[1]# Current position
            print(i, j)
            
            # Ensure the agent stays within the grid
            i = max(0, min(10 - 1, i))
            j = max(0, min(10 - 1, j))
                
            # Q-learning update rule:
            # 1. Calculate the sample using the reward and the maximum Q-value of the next state
            sample = rewards[pi , pj] + gamma * np.max(q_values[i, j]) + living_reward

            # 2. Update Q-value using the Q-learning update rule (weighted average)
            q_values[pi, pj, action_index] = (1 - alpha) * q_values[pi, pj, action_index] + alpha * sample
            
            # If hits goal, restart episode
            if rewards[i, j] == GOAL_REWARD:
                print("Hit Goal")
                break
            
            # If tried to move into a wall (boundary or actual), restart the episode
            if pi == i and pj == j:
                print("Hit Wall/Boundary")
                break
            
            print(f"Episode {episode}, step {steps}")
            #print(q_values)
            steps += 1
          
        if min_epsilon < epsilon:
            epsilon -= decay_rate
            
    return q_values

def final_policy(n, m, rewards, q_values, actions):
    final_policy = np.zeros((n, m), dtype=object)
    for i in range(n):
        for j in range(m):
            # If terminal state
            if rewards[i, j] == GOAL_REWARD: #rewards[i][j] == 10 or rewards[i][j] == -10
                final_policy[i][j] = "*"
                continue
            
            # Choose best among q_values for each square (4 per [n, m] pair)
            best_action_idx = np.argmax(q_values[i, j])
            final_policy[i][j] = best_action_idx
    
    # Convert numeric policy to direction symbols, leaving "*" as is
    directions_map = {0: "←", 1: "↑", 2: "→", 3: "↓", "*": "*"}
    symbolic_policy = np.vectorize(directions_map.get)(final_policy)
    return symbolic_policy
    

