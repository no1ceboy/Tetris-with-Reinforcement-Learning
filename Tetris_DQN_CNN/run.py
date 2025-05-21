from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard # Using CustomTensorBoard for logging
from tqdm import tqdm # For progress bars
import numpy as np
from collections import deque # For efficient fixed-size queues (replay memory, scores window)
import tensorflow as tf # For GPU configuration
from typing import Optional, List, Tuple, Union, Dict

# --- HYPERPARAMETERS AND CONFIGURATION ---
EPISODES: int = 10000 # Total number of episodes to train for
MAX_STEPS_PER_EPISODE: Optional[int] = None # Max steps per episode, None for no limit
EPSILON_STOP_EPISODE_RATIO: float = 0.9 # Epsilon stops decaying after this fraction of total episodes
MEMORY_SIZE: int = 50000        # Max size of the replay memory buffer
DISCOUNT_FACTOR: float = 0.99   # Gamma, discount factor for future rewards
BATCH_SIZE: int = 64            # Number of experiences sampled from memory for each training step
EPOCHS_PER_TRAIN_STEP: int = 1  # Number of epochs for model.fit() during one training call

RENDER_EVERY_N_EPISODES: int = 100 # Render the game every N episodes (0 to disable)
RENDER_DELAY_TRAINING: float = 0.001 # Delay between frames when rendering during training
LOG_EVERY_N_EPISODES: int = 50    # Frequency to log metrics to console and TensorBoard
REPLAY_START_SIZE_FRACTION: float = 0.05 # Start training when memory is this fraction full
TRAIN_EVERY_N_EPISODES: int = 1   # Train the agent every N episodes (1 means train after each episode)
SAVE_BEST_MODEL: bool = True      # Whether to save the modelότε a new best average score is achieved

# Neural Network Architecture Configuration
CONV_LAYER_CONFIGS: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = [
    (32, (4, 4), (1, 1)), (64, (3, 3), (1, 1)), # (filters, kernel_size, strides)
]
POOLING_CONFIGS: List[Union[bool, Tuple[int, int]]] = [True, True] # True for (2,2) pool, or specific tuple
DENSE_LAYER_NEURONS: List[int] = [256] # Number of neurons in dense layers
DENSE_LAYER_ACTIVATIONS: List[str] = ['relu', 'linear'] # Activations for dense layers (last one for Q-output)

# Agent Specific Hyperparameters
LEARNING_RATE_DQN: float = 0.0001 # Learning rate for the DQN's optimizer
TARGET_NETWORK_UPDATE_FREQUENCY: int = 200 # Update target network every N training steps


def configure_gpu_memory_growth() -> None:
    """Configures TensorFlow to use GPU memory growth to avoid overallocating memory."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: # Iterate over all available GPUs
                # Set memory growth to True to allocate memory on-demand
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # This print statement confirms GPU configuration
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e: # Handle potential errors during configuration
            # This print statement indicates an error during GPU setup
            print(f"Error setting memory growth: {e}")

def train_dqn_agent() -> None:
    """
    Main function to initialize and train the DQN agent on the Tetris environment.
    """
    configure_gpu_memory_growth() # Attempt to configure GPU memory

    env = Tetris() # Initialize the Tetris game environment
    # Calculate absolute episode number for stopping epsilon decay
    epsilon_stop_episode_abs = int(EPISODES * EPSILON_STOP_EPISODE_RATIO)
    # Calculate absolute number of experiences needed before training starts
    replay_start_size_abs = int(MEMORY_SIZE * REPLAY_START_SIZE_FRACTION)

    # Initialize the DQNAgent with specified parameters
    agent = DQNAgent(
        input_shape=env.get_state_size(), mem_size=MEMORY_SIZE, discount=DISCOUNT_FACTOR,
        epsilon_stop_episode=epsilon_stop_episode_abs, conv_layers=CONV_LAYER_CONFIGS,
        pooling_layers=POOLING_CONFIGS, dense_neurons=DENSE_LAYER_NEURONS,
        dense_activations=DENSE_LAYER_ACTIVATIONS, optimizer_lr=LEARNING_RATE_DQN,
        replay_start_size=replay_start_size_abs, target_update_freq=TARGET_NETWORK_UPDATE_FREQUENCY
    )

    # --- Setup Logging ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") # Create a unique timestamp
    # Create a descriptive log directory name to differentiate runs
    log_dir_name = (f'tetris-DQN-core_logs-{timestamp}') # Changed name for core logs
    log_dir_path = f'logs/{log_dir_name}' # Full path to the log directory

    logger: Optional[CustomTensorBoard] = None # Initialize logger as None
    try:
        logger = CustomTensorBoard(log_dir=log_dir_path) # Attempt to create TensorBoard logger
    except Exception as e: # Handle potential errors during logger initialization
        # This print statement indicates an error with TensorBoard setup
        print(f"Could not initialize CustomTensorBoard: {e}. Logging will be basic (console only).")

    # --- Training Loop Initialization ---
    # Deque to store scores of recent episodes for calculating moving average and max
    scores_window: deque = deque(maxlen=LOG_EVERY_N_EPISODES)
    best_avg_score: float = -np.inf # Initialize best average score to a very low number

    # Main training loop over episodes
    for episode in tqdm(range(1, EPISODES + 1), unit="episode"):
        current_board_matrix: Optional[np.ndarray] = env.reset() # Reset environment for new episode
        done: bool = False            # Flag to indicate if the episode is finished
        steps_this_episode: int = 0   # Counter for steps in the current episode
        current_loss: Optional[float] = None # Store loss from training in this episode, reset each episode

        # Determine if this episode should be rendered
        render_this_episode = (RENDER_EVERY_N_EPISODES > 0 and episode % RENDER_EVERY_N_EPISODES == 0)

        # Inner loop for steps within an episode
        while not done and (MAX_STEPS_PER_EPISODE is None or steps_this_episode < MAX_STEPS_PER_EPISODE):
            if current_board_matrix is None: # Safety check, should not happen
                done = True; break

            # Get all possible next states (and corresponding actions) from the environment
            next_possible_moves: Dict[Tuple[int, int], np.ndarray] = env.get_next_states()
            if not next_possible_moves: # If no moves possible, end episode
                done = True; break

            # Agent chooses an action based on current state and possible moves
            chosen_action: Optional[Tuple[int, int]] = agent.best_state(next_possible_moves)
            if chosen_action is None: # If agent cannot decide (should be rare if moves exist)
                done = True; break

            # The board state *resulting* from the chosen_action (after piece lands, lines clear)
            # This is the S' that will be stored in memory for the (S, A, R, S') tuple.
            resulting_board_matrix_for_memory: np.ndarray = next_possible_moves[chosen_action]

            # Agent performs the chosen action in the environment
            # `new_board_matrix` is the state *after* the action and *after* a new piece has spawned.
            # This `new_board_matrix` becomes the `current_board_matrix` for the *next* step.
            new_board_matrix, reward_from_play, done = env.play(
                chosen_action[0], chosen_action[1], # Action components (column, rotation)
                render=render_this_episode,
                render_delay=RENDER_DELAY_TRAINING if render_this_episode else None
            )

            # Add the experience to the agent's replay memory
            agent.add_to_memory(current_board_matrix, resulting_board_matrix_for_memory, reward_from_play, done)
            
            current_board_matrix = new_board_matrix # Update current state for the next iteration
            steps_this_episode += 1
        
        # --- End of Episode ---
        episode_score: int = env.get_game_score() # Get final score for the episode
        scores_window.append(episode_score)      # Add to scores window for moving average calculation
        
        # Optionally, get other per-episode stats if needed for more detailed logging
        # episode_lines_cleared: int = env.get_lines_cleared_this_episode()
        # episode_total_reward: float = env.get_reward_accumulated_this_episode()

        # Train the agent if it's time and enough experiences are in memory
        if episode % TRAIN_EVERY_N_EPISODES == 0 and len(agent.memory) >= agent.replay_start_size:
            # Only assign current_loss if train() actually ran and returned a value
            train_loss_value = agent.train(batch_size=BATCH_SIZE, epochs=EPOCHS_PER_TRAIN_STEP)
            if train_loss_value is not None:
                current_loss = train_loss_value

        # Log metrics to console and TensorBoard at specified frequency
        if logger and episode % LOG_EVERY_N_EPISODES == 0:
            # Calculate average score from the scores_window
            avg_score_val = mean(scores_window) if scores_window else -np.inf # Default if window is empty
            
            # Console output (simplified)
            # This print statement shows progress on the console
            print(f"\nEp: {episode}, Avg Score (Win): {avg_score_val:.2f}, "
                  # f"Lines (Ep): {env.get_lines_cleared_this_episode()}, " # Optional: show lines cleared
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Loss: {current_loss if current_loss is not None else 'N/A'}")
            
            # Data to log to TensorBoard (core metrics)
            log_data = {
                'avg_score_window': avg_score_val,       # Moving average score
                'training_loss': current_loss,          # Training loss (can be None if not trained)
                'epsilon': agent.epsilon,               # Current epsilon value
                # Add other metrics here if desired for TensorBoard:
                'episode_lines_cleared': float(env.get_lines_cleared_this_episode()), # Lines cleared in this specific episode
                # 'episode_score': float(episode_score), # Raw score of this specific episode
                # 'max_score_window': float(max(scores_window)) if scores_window else -np.inf,
            }
            # Log to TensorBoard, filtering out None values to prevent errors
            logger.log(episode, **{k: v for k, v in log_data.items() if v is not None})

            # Save the model if it's the best performing one so far (based on avg_score_val)
            if SAVE_BEST_MODEL and avg_score_val > best_avg_score and scores_window:
                # This print statement indicates a new best model is being saved
                print(f"New best average score: {avg_score_val:.2f} (was {best_avg_score:.2f}). Saving model...")
                best_avg_score = avg_score_val
                agent.save_model("best_tetris_model_core_log.keras") # Model filename

    # --- End of Training ---
    # This print statement indicates training completion
    print("Training finished.")
    agent.save_model("final_tetris_model_core_log.keras") # Save the final model
    if logger: # Close the logger if it was initialized
        logger.close()
    if hasattr(env, 'cv2') and env.cv2: # Clean up OpenCV windows if used
        env.cv2.destroyAllWindows()

if __name__ == "__main__":
    train_dqn_agent() # Run the training process