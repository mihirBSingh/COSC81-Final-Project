import numpy as np
import random
import rclpy
from mapping import GridMapper
# from rclpy.node import Node
import threading


def print_grid_window(map_data, center, window_size=5):
    cx, cy = center
    half = window_size // 2

    print(f"\n--- Grid window around {center} ---")
    for y in range(cy - half, cy + half + 1):
        row = ""
        for x in range(cx - half, cx + half + 1):
            if 0 <= x < map_data.shape[0] and 0 <= y < map_data.shape[1]:
                val = map_data[y, x]
                if (x, y) == center:
                    row += f"[{val:3}] "  # Highlight center
                else:
                    row += f" {val:3}  "
            else:
                row += "  X   "
        print(row)
    print("--- End Grid ---\n")

def print_qtable_window(q_table, state):
    """
    Print the Q-values for all actions at a single state.
    Args:
        q_table: 3D numpy array (x, y, actions)
        state: tuple of (x, y) coordinates
    """
    action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
    
    print(f"\n=== Q-values at state {state} ===")
    for action, name in enumerate(action_names):
        print(f"{name:5}: {q_table[state][action]:6.2f}")
    
    print("=== End Q-values ===\n")



class QLearningAgent:
    def __init__(self, discount_rate=0.9, learning_rate=0.1, exploration_rate=0.5, initial_size=1000, res=0.05):
        grid_size = int(initial_size * res)
        self.q_table = np.zeros((grid_size, grid_size, 4))  # x,y,q-value -- x,y in m
        self.origin_x = round(initial_size / 2.0 * res)
        self.origin_y = round(initial_size / 2.0 * res)

        self.learning_rate = learning_rate
        self.discount_factor = discount_rate
        self.exploration_rate = exploration_rate 
        self.res = res 

        print(f"Q-table initialized with shape: {self.q_table.shape} and origin: {self.origin_x}, {self.origin_y}\n")

    def choose_action(self, state, grid):
        tried = set()
        max_actions = 4
        actions = list(range(max_actions))

        while len(tried) < max_actions:
            if random.uniform(0, 1) < self.exploration_rate:
                print("   exploring")
                action = random.choice(list(set(actions) - tried))
            else:
                print("   exploiting")
                q_vals = self.q_table[state]
                for a in np.argsort(-q_vals):
                    if a not in tried:
                        action = a
                        break
            tried.add(action)

            # Compute the next state in world coordinates and then grid indices
            state_world = grid.get_next_state(state, action)  # in meters
            next_state = grid.world_to_grid(state_world[0], state_world[1])  # in grid coords (x, y)

            # Debug info: print the grid window around considered next state
            # print(f"Trying action {action} -> next state (grid coords): {next_state}")
            # print_grid_window(grid.map, next_state, window_size=5)

            # Check that next_state is within map bounds
            if not (0 <= next_state[0] < grid.width and 0 <= next_state[1] < grid.height):
                # print(f"      Next state {next_state} out of bounds, trying next action...")
                continue

            # Check occupancy grid cell value at next_state
            cell_val = grid.map[next_state[1], next_state[0]]

            # Only allow action if cell is free (0). Block if obstacle (100) or unknown (-1)
            if cell_val == 0:
                move_names = ["UP", "RIGHT", "DOWN", "LEFT"]
                print(f"      Chose action: {action} ({move_names[action]}) for state: {state}")
                return action
            else:
                print(f"      Blocked by obstacle or unknown cell (val={cell_val}) at {next_state}, trying next action...")

        # If all actions are blocked, default to action 0 (UP)
        print("      All directions blocked. Returning action 'None'.")
        return None

    def update_q_value(self, state, action, reward, next_state):
        print(f"      Updating Q-value for state: {state}, action: {action}, reward: {reward}, next_state: {next_state}")
        max_future_q = np.max(self.q_table[next_state])  # Best Q-value for next state
        current_q = self.q_table[state][action]
        # Q-learning formula
        td_update = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        print(f"      Q-value update: {td_update}")
        self.q_table[state][action] = td_update  # NOTE: state is x,y AND qtable is x,y,action (not y,x,action)

        print_qtable_window(self.q_table, state)
    
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

    def train(self, num_episodes, grid, reward_type ="manhattan"):
        print(f"Training for {num_episodes} episodes...")
        state = (0,0)
        for _ in range(num_episodes):
            print(f" --- State (px): {state} --- ")
            state_offset = (state[0] + self.origin_x, state[1] + self.origin_y)  # offset by origin 
            done = False

            while not done:
                action = self.choose_action(state, grid)

                next_state, reward, done = grid.step(action, reward_type)
                
                next_state_offset = (next_state[0] + self.origin_x, next_state[1] + self.origin_y)  # offset by origin 
                if not self.in_bounds(next_state_offset):
                    self.expand_qtable()
                
                self.update_q_value(state_offset, action, reward, next_state_offset)

                state = next_state
                state_offset = next_state_offset
                print(f" --- State (px): {state} --- ")
            state = grid.reset()

        print(f"Training complete.\n")
        print(f"Q-table shape: {self.q_table.shape}")
        print(f"Q-table: {self.q_table}")

def main(args=None):
    rclpy.init(args=args)
        
    initial_size = 1000 
    res = 0.05

    # odom px
    startx = 0 
    starty = 0 
    goal = (-3,1)  # m

    gm_node = GridMapper(goal=goal, pos_x=startx, pos_y=starty, initial_size=initial_size, res=res)
    q = QLearningAgent(initial_size=initial_size, res=res)
 
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(gm_node)
    executor.add_node(gm_node.planner)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # hyperparameters TODO: tune
    # discount_factor = 0.4
    # learning_rate = 0.1 
    # exploration_rate = 0.1

    num_episodes = 10
    q.train(num_episodes, gm_node)

    # try:
    #     rclpy.spin(gm_node)
    # except KeyboardInterrupt:
    #     pass
    # gm_node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()

