import numpy as np
import random
import rclpy
from mapping import GridMapper

class QLearningAgent:
    def __init__(self, discount_rate=0.9, learning_rate=0.1, exploration_rate=0.1):
        self.q_table = np.zeros((4,4,4))  # x,y,q-value 

        self.learning_rate = learning_rate
        self.discount_factor = discount_rate
        self.exploration_rate = exploration_rate 

        print(f"Q-table initialized with shape: {self.q_table.shape}")

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0, 3) # Explore random action 
        else:
            action = np.argmax(self.q_table[state])  # Exploit best action from qtable
        
        movement = "UP" if action == 0 else "RIGHT" if action == 1 else "DONW" if action == 2 else "LEFT"
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
    
    def train(self, num_episodes, grid, reward_type ="obstacle"):
        print(f"Training for {num_episodes} episodes...")
        for _ in range(num_episodes):
            state = grid.reset()
            done = False
            print(f"   Starting episode with state: {state}")
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = grid.step(action, reward_type)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                print(f"   Updated state: {state}")

def main(args=None):
    rclpy.init(args=args)
    gm_node = GridMapper()

    # hyperparameters TODO: tune
    # discount_factor = 0.4
    # learning_rate = 0.1 
    # exploration_rate = 0.1

    q = QLearningAgent()

    q.train(100, gm_node)

    try:
        rclpy.spin(gm_node)
    except KeyboardInterrupt:
        pass
    gm_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

