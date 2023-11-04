#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Changelog (04/11/23 3:15pm):
- Epsilon is now scaled down proportionally to percentCellsExplored in ReinforcementLearning.py
- Vars for epsilon now initialized in ReinforcementLearning.py
- "decay_rate" removed from ReinforcementLearning.py; to be replaced with epsilon scaling subroutine
- Removed calls and imports for QLearning.py in GridWorld.py, replaced with ReinforcementLearning.py
- "steps_var" renamed to "step_var" in GridWorld.py for consistency with naming in ReinforcementLearning.py
- Initialized "living_reward" in GridWorld.py as -1, since ReinforcementLearning.py expects it as a param
- Minor comment modifications, to align with python coding standards; few spelling corrections
- Changed WALL_VALUE to -np.inf in ReinforcementLearning.py and GridWorld.py, from np.inf and -1000 respectively
"""

import numpy as np

WALL_VALUE = -np.inf  # Negative infinity
GOAL_REWARD = 6

# Moves robot multiple times for an "episode"
def Q_learning(world, rewards, episodes, gamma, min_epsilon, max_epsilon, alpha, living_reward, step_var):
    q_values = np.zeros((10, 10, 4))
    print(rewards)
    
    # Initialize vars related to epsilon scaling
    # Epsilon is scaled down proportionally to percent_cells_explored
    # This ensures that exploration and exploitation are balanced based on current world knowledge
    epsilon = max_epsilon  # Start by favouring exploration over exploitation
    epsilon_range = max_epsilon - min_epsilon
    num_cells_explored = 0  # Count of novel cells visited
    percent_cells_explored = 0
    unvisited_cells = []  # Track coodinates of all cells that are not walls, and have not been explored in any episode
    
    # Populate unvisited_cells list
    for i in range(world.size):
        for j in range(world.size):
            if rewards[i][j] != WALL_VALUE:
                unvisited_cells.append([i,j])
    print(unvisited_cells)
    total_cells = len(unvisited_cells)  # Number of cells in world that are not walls

    # Every iteration is an episode
    for episode in range(episodes):
        steps = 0
        i, j = world.robot_pos # Capture initial position
        actions = set()
        while steps < step_var:
            random_value = np.random.rand()  # Generates a float where 0 < random_value < 1
            # Exploration
            if random_value < epsilon: 
                action_index = np.random.randint(0, len(actions))
           # Exploitation
            else:
                action_index = np.argmax(q_values[i, j])
                
            actual_action = actions[action_index]
            
            new_i, new_j = i + actual_action[0], j + actual_action[1] # New position
            #print(new_i, new_j)
            
            # Ensure the agent stays within the grid
            new_i = max(0, min(10 - 1, new_i))
            new_j = max(0, min(10 - 1, new_j))
            
            # If tries to move into a wall, do not update Q-values
            if rewards[new_i, new_j] != WALL_VALUE:
                
                # Takes into account goal, penalty and reward states
                # 1. Calculate the sample using the reward and the maximum Q-value of the next state
                sample = rewards[i , j] + gamma * np.max(q_values[new_i, new_j]) + living_reward

                # 2. Update Q-value using the Q-learning update rule (weighted average)
                q_values[i, j, action_index] = (1 - alpha) * q_values[i, j, action_index] + alpha * sample
                
                # Update current position
                i, j = new_i, new_j
                print('pos = ' + str(i) + ',' + str(j))
                # Remove current cell from unvisited_cells list
                if [i,j] in unvisited_cells:
                    unvisited_cells.remove([i,j])
                
                # If hits goal, restart episode
                if rewards[new_i, new_j] == GOAL_REWARD:
                    break
            
            #print(f"Episode {episode}, step {steps}")
            steps += 1
         
        # Update vars related to epsilon scaling
        num_cells_explored = total_cells - len(unvisited_cells)
        percent_cells_explored = num_cells_explored / total_cells
        
        # Scale epsilon down proportionally to percent_cells_explored
        epsilon = max_epsilon - (epsilon_range * percent_cells_explored)
            
    return q_values

def final_policy(n, m, rewards, q_values, actions):
    final_policy = np.zeros((n, m), dtype=object)
    for i in range(n):
        for j in range(m):
            # If terminal state
            if rewards[i, j] == GOAL_REWARD: # rewards[i][j] == 10 or rewards[i][j] == -10
                final_policy[i][j] = "*"
                continue
            
            # Choose best among q_values for each square (4 per [n, m] pair)
            best_action_idx = np.argmax(q_values[i, j])
            final_policy[i][j] = best_action_idx
    
    # Convert numeric policy to direction symbols, leaving "*" as is
    directions_map = {0: "←", 1: "↑", 2: "→", 3: "↓", "*": "*"}
    symbolic_policy = np.vectorize(directions_map.get)(final_policy)
    return symbolic_policy
    

