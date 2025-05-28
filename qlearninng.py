import numpy as np
import random

# actions = [(0,0), (0,1), (1,0), (1,1)] 

class Grid: 
    def __init__(self, occupancy_grid_data, width, height, resolution, origin, start_state):
        self.grid = np.reshape(occupancy_grid_data, (height, width))

        self.resolution = resolution  # m/cell
        self.height = self.grid.shape[0]
        self.width  = self.grid.shape[1]
        self.origin = origin

        self.start_state = start_state  # (row, col)
        self.state = start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        return self.grid[state] == 1 or self.grid[state] == -1
    
    def get_next_state(self, state, action):
        # return (state[0] + action[0], state[1] + action[1])
        next_state = list(state)
        if action == 0:  # Move up
            next_state[0] = max(0, state[0] - 1)
        elif action == 1:  # Move right
            next_state[1] = min(3, state[1] + 1)
        elif action == 2:  # Move down
            next_state[0] = min(3, state[0] + 1)
        elif action == 3:  # Move left
            next_state[1] = max(0, state[1] - 1)
        return tuple(next_state)
    
    def compute_reward(self, prev_state, action):  # TODO: becs
        # Obstacle or goal
        # Manhattan distance
        pass 
    
    def step(self, action):
        next_state = self.get_next_state(self.state, action)
        reward = self.compute_reward(self.state, action) # self.grid[next_state] (from medium article)
        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def m_to_grid(self, x, y):
        return x / self.resolution, y / self.resolution 

    def grid_to_m(self, row, col):
        return col * self.resolution, row * self.resolution
    
    def expand_grid(self, row, col):  # TODO: scottie
        pass 
    

class QLearningAgent:
    def __init__(self, discount_rate=0.9, learning_rate=0.1, exploration_rate=0.1):
        self.Q_table = np.zeros((4, 4, 4))  # x,y,q-value 

        self.learning_rate = learning_rate
        self.discount_factor = discount_rate
        self.exploration_rate = exploration_rate 

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3) # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])  # Best Q-value for next state
        current_q = self.q_table[state][action]
        # Q-learning formula
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
    
    def train(self, num_episodes):
        pass 
# Q-learning class:
    # TODO __init__(Q table, discount_rate, learning_rate) 
    # TODO Choose action(self, curr_state)
    # TODO Update value(self, curr_state, action, next_state, reward)


# hyperparameters to tune
discount_factor = 0.4
learning_rate = 0.1 
# g = Grid(np.zeros(100), 10, 10, 0.1, (0,0))
q = QLearningAgent()