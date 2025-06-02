import numpy as np
import random
import rclpy
from mapping import GridMapper
# from rclpy.node import Node
import threading

class QLearningAgent:
    def __init__(self, discount_rate=0.9, learning_rate=0.1, exploration_rate=0.1, initial_size=1000, res=0.05):
        grid_size = int(initial_size * res)
        self.q_table = np.zeros((grid_size, grid_size, 4))  # x,y,q-value -- x,y in m
        self.origin_x = round(initial_size / 2.0 * res)
        self.origin_y = round(initial_size / 2.0 * res)

        self.learning_rate = learning_rate
        self.discount_factor = discount_rate
        self.exploration_rate = exploration_rate 
        self.res = res 

        print(f"Q-table initialized with shape: {self.q_table.shape} and origin: {self.origin_x}, {self.origin_y}\n")

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0, 3) # Explore random action 
        else:
            action = np.argmax(self.q_table[state])  # Exploit best action from qtable
        
        movement = "UP" if action == 0 else "RIGHT" if action == 1 else "DOWN" if action == 2 else "LEFT"
        print(f"      Chose action: {action} ({movement}) for state: {state}")
        return action

    def update_q_value(self, state, action, reward, next_state):
        print(f"      Updating Q-value for state: {state}, action: {action}, reward: {reward}, next_state: {next_state}")
        max_future_q = np.max(self.q_table[next_state])  # Best Q-value for next state
        current_q = self.q_table[state][action]
        # Q-learning formula
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
    
    def expand_qtable(self):
        current_size = self.q_table.shape[0]
        expansion_size = 10

        # create new qtable with expanded size 
        new_q_table = np.zeros((current_size+expansion_size, current_size+expansion_size, 4))
        
        # offset origin by half of expansion size 
        self.origin_x = round(self.origin_x + expansion_size/2)
        self.origin_y = round(self.origin_y + expansion_size/2)
        
        # copy old qtable to new qtable 
        new_q_table[expansion_size/2:self.q_table.shape[0], expansion_size/2:self.q_table.shape[1], :] = self.q_table
        self.q_table = new_q_table

        print(f"Q-table expanded to shape: {self.q_table.shape} and origin: {self.origin_x}, {self.origin_y}")
    
    def in_bounds(self, state):
        return 0 <= state[0] < self.q_table.shape[0] and 0 <= state[1] < self.q_table.shape[1]

    def train(self, num_episodes, grid, reward_type ="obstacle"):
        print(f"Training for {num_episodes} episodes...")
        for _ in range(num_episodes):
            state = grid.reset()
            state_offset = (state[0] + self.origin_x, state[1] + self.origin_y)  # offset by origin 
            done = False
            print(f" --- State (px): {state} --- ")
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = grid.step(action, reward_type)
                
                next_state_offset = (next_state[0] + self.origin_x, next_state[1] + self.origin_y)  # offset by origin 
                if not self.in_bounds(next_state_offset):
                    self.expand_qtable()
                
                self.update_q_value(state_offset, action, reward, next_state_offset)
                state = next_state
                print(f" --- State (px): {state} --- ")

def main(args=None):
    rclpy.init(args=args)
        
    initial_size = 500 
    res = 0.05

    # odom px
    startx = 0 
    starty = 0 
    goal = (initial_size-1, initial_size-1)

    gm_node = GridMapper(goal=goal, pos_x=startx, pos_y=starty, initial_size=initial_size, res=res)
    q = QLearningAgent(initial_size=initial_size, res=res)
 
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(gm_node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # hyperparameters TODO: tune
    # discount_factor = 0.4
    # learning_rate = 0.1 
    # exploration_rate = 0.1

    q.train(100, gm_node)

    # try:
    #     rclpy.spin(gm_node)
    # except KeyboardInterrupt:
    #     pass
    # gm_node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()

