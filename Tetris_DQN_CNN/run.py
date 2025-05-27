from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from logs import CustomTensorBoard
from tqdm import tqdm
import numpy as np
from collections import deque
import tensorflow as tf
from typing import Optional, List, Tuple, Union, Dict

EPISODES: int = 10000
MAX_STEPS_PER_EPISODE: Optional[int] = 1500  # Giới hạn số bước/ep để replay buffer đa dạng hơn
EPSILON_STOP_EPISODE_RATIO: float = 0.2
MEMORY_SIZE: int = 10000
DISCOUNT_FACTOR: float = 0.99
BATCH_SIZE: int = 64
EPOCHS_PER_TRAIN_STEP: int = 1

RENDER_EVERY_N_EPISODES: int = 500
RENDER_DELAY_TRAINING: float = 0.001
LOG_EVERY_N_EPISODES: int = 50
REPLAY_START_SIZE_FRACTION: float = 0.1
TRAIN_EVERY_N_EPISODES: int = 1
SAVE_BEST_MODEL: bool = True

CONV_LAYER_CONFIGS: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = [
    (64, (4, 4), (1, 1)), (128, (3, 3), (1, 1)), (128, (3, 3), (1, 1))
]
POOLING_CONFIGS: List[Union[bool, Tuple[int, int]]] = [True, True, True]
DENSE_LAYER_NEURONS: List[int] = [512, 256]
DENSE_LAYER_ACTIVATIONS: List[str] = ['relu', 'relu', 'linear']

LEARNING_RATE_DQN: float = 0.00005
TARGET_NETWORK_UPDATE_FREQUENCY: int = 500

def configure_gpu_memory_growth() -> None:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

def train_dqn_agent() -> None:
    configure_gpu_memory_growth()
    env = Tetris()
    epsilon_stop_episode_abs = int(EPISODES * EPSILON_STOP_EPISODE_RATIO)
    replay_start_size_abs = int(MEMORY_SIZE * REPLAY_START_SIZE_FRACTION)
    agent = DQNAgent(
        input_shape=env.get_state_size(),
        mem_size=MEMORY_SIZE,
        discount=DISCOUNT_FACTOR,
        epsilon_stop_episode=epsilon_stop_episode_abs,
        conv_layers=CONV_LAYER_CONFIGS,
        pooling_layers=POOLING_CONFIGS,
        dense_neurons=DENSE_LAYER_NEURONS,
        dense_activations=DENSE_LAYER_ACTIVATIONS,
        optimizer_lr=LEARNING_RATE_DQN,
        replay_start_size=replay_start_size_abs,
        target_update_freq=TARGET_NETWORK_UPDATE_FREQUENCY
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = (f'tetris-DQN-core_logs-{timestamp}')
    log_dir_path = f'logs/{log_dir_name}'

    logger: Optional[CustomTensorBoard] = None
    try:
        logger = CustomTensorBoard(log_dir=log_dir_path)
    except Exception as e:
        print(f"Could not initialize CustomTensorBoard: {e}. Logging will be basic (console only).")

    scores_window: deque = deque(maxlen=LOG_EVERY_N_EPISODES)
    best_avg_score: float = -np.inf

    for episode in tqdm(range(1, EPISODES + 1), unit="episode"):
        current_board_matrix: Optional[np.ndarray] = env.reset()
        done: bool = False
        steps_this_episode: int = 0
        current_loss: Optional[float] = None
        render_this_episode = (RENDER_EVERY_N_EPISODES > 0 and episode % RENDER_EVERY_N_EPISODES == 0)

        while not done and (MAX_STEPS_PER_EPISODE is None or steps_this_episode < MAX_STEPS_PER_EPISODE):
            if current_board_matrix is None:
                done = True
                break
            next_possible_moves: Dict[Tuple[int, int], np.ndarray] = env.get_next_states()
            if not next_possible_moves:
                done = True
                break
            chosen_action: Optional[Tuple[int, int]] = agent.best_state(next_possible_moves)
            if chosen_action is None:
                done = True
                break
            resulting_board_matrix_for_memory: np.ndarray = next_possible_moves[chosen_action]
            new_board_matrix, reward_from_play, done = env.play(
                chosen_action[0], chosen_action[1],
                render=render_this_episode,
                render_delay=RENDER_DELAY_TRAINING if render_this_episode else None
            )
            agent.add_to_memory(current_board_matrix, resulting_board_matrix_for_memory, reward_from_play, done)
            current_board_matrix = new_board_matrix
            steps_this_episode += 1

        episode_score: int = env.get_game_score()
        scores_window.append(episode_score)

        if episode % TRAIN_EVERY_N_EPISODES == 0 and len(agent.memory) >= agent.replay_start_size:
            train_loss_value = agent.train(batch_size=BATCH_SIZE, epochs=EPOCHS_PER_TRAIN_STEP)
            if train_loss_value is not None:
                current_loss = train_loss_value

        if logger and episode % LOG_EVERY_N_EPISODES == 0:
            avg_score_val = mean(scores_window) if scores_window else -np.inf
            print(f"\nEp: {episode}, Avg Score (Win): {avg_score_val:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Loss: {current_loss if current_loss is not None else 'N/A'}")
            log_data = {
                'avg_score': avg_score_val,
                'training_loss': current_loss,
                'epsilon': agent.epsilon,
                'episode_lines_cleared': float(env.get_lines_cleared_this_episode()),
            }
            logger.log(episode, **{k: v for k, v in log_data.items() if v is not None})

            if SAVE_BEST_MODEL and avg_score_val > best_avg_score and scores_window:
                print(f"New best average score: {avg_score_val:.2f} (was {best_avg_score:.2f}). Saving model...")
                best_avg_score = avg_score_val
                agent.save_model("best_tetris_cnn.keras")

    print("Training finished.")
    agent.save_model("final_tetris_cnn.keras")
    if logger:
        logger.close()
    if hasattr(env, 'cv2') and env.cv2:
        env.cv2.destroyAllWindows()

if __name__ == "__main__":
    train_dqn_agent()