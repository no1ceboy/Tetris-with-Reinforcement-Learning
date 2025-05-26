import sys
import time
from dqn_agent import DQNAgent
from tetris import Tetris
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python run_model.py <model_file.keras>")
    sys.exit(1)

model_file = sys.argv[1]

env = Tetris()
# When loading, DQNAgent needs input_shape. Other architectural params are defaults for new model if load fails.
# Epsilon settings for inference mode (no exploration)
agent = DQNAgent(env.get_state_size(), modelFile=model_file, epsilon=0, epsilon_min=0) 

current_board_matrix = env.reset()
done = False
total_score = 0
total_steps = 0

RENDER_DELAY_INFERENCE = 0.05 # Adjust for comfortable viewing speed

try:
    while not done:
        env.render() # Render current state
        time.sleep(RENDER_DELAY_INFERENCE)

        next_possible_moves = env.get_next_states()
        
        if not next_possible_moves:
            print("No possible moves. Game Over.")
            break
            
        # Agent selects the best action (no exploration as epsilon=0)
        best_action = agent.best_state(next_possible_moves)
        
        if best_action is None:
            print("Agent could not decide on a best action. Game Over.")
            break
            
        # Perform the action
        # env.play returns new_board_state_matrix, reward, done_flag
        new_board_matrix, reward, done = env.play(best_action[0], best_action[1], render=False) # Rendering is handled above
        
        current_board_matrix = new_board_matrix
        total_score += reward # Accumulate reward if you want, or use env.get_game_score()
        total_steps += 1

        if done:
            print("Game Over!")
            env.render() # Render final state
            time.sleep(1)


except KeyboardInterrupt:
    print("\nGame interrupted by user.")
finally:
    print(f"Final Score (from env): {env.get_game_score()}")
    print(f"Total Steps: {total_steps}")
    if hasattr(env, 'cv2'):
        env.cv2.destroyAllWindows()