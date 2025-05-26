from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.losses import Huber
from collections import deque
import numpy as np
import random
from typing import Tuple, List, Union, Optional, Dict, Any

class DQNAgent:
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 mem_size: int = 20000,
                 discount: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_stop_episode: int = 1500,
                 conv_layers: Optional[List[Tuple[int, Tuple[int, int], Tuple[int, int]]]] = None,
                 pooling_layers: Optional[List[Union[bool, Tuple[int, int]]]] = None,
                 dense_neurons: Optional[List[int]] = None,
                 dense_activations: Optional[List[str]] = None,
                 loss: Any = Huber(),
                 optimizer_lr: float = 0.0003,
                 replay_start_size: Optional[int] = None,
                 modelFile: Optional[str] = None,
                 target_update_freq: int = 100):
        # (validations unchanged)
        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError("input_shape must be a tuple of length 3 (height, width, channels)")
        self.input_shape: Tuple[int, int, int] = input_shape

        self.conv_layers_params: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = \
            conv_layers if conv_layers is not None else [(64, (4,4), (1,1)), (128, (3,3), (1,1)), (128, (3,3), (1,1))]
        self.pooling_layers_config: List[Union[bool, Tuple[int, int]]] = \
            pooling_layers if pooling_layers is not None else [True, True, True]
        self.dense_neurons_list: List[int] = dense_neurons if dense_neurons is not None else [512, 256]
        self.dense_activations_list: List[str] = \
            dense_activations if dense_activations is not None else ['relu', 'relu', 'linear']

        if len(self.dense_activations_list) != len(self.dense_neurons_list) + 1:
            raise ValueError(
                f"dense_activations (len {len(self.dense_activations_list)}) and "
                f"dense_neurons (len {len(self.dense_neurons_list)}) do not match. "
                "Expected len(dense_activations) == len(dense_neurons) + 1."
            )
        if any(n <= 0 for n in self.dense_neurons_list):
            raise ValueError("dense_neurons must contain positive integers")
        if not (0 <= discount <= 1):
            raise ValueError("discount must be between 0 and 1")
        if not (0 <= epsilon <= 1):
            raise ValueError("epsilon must be between 0 and 1")
        if not (0 <= epsilon_min <= 1) or epsilon_min > epsilon:
            raise ValueError("epsilon_min must be between 0 and 1 and <= epsilon")
        if epsilon_stop_episode < 0:
            raise ValueError("epsilon_stop_episode must be non-negative")
        if mem_size <= 0:
            raise ValueError("mem_size must be > 0")

        self.mem_size: int = mem_size
        self.memory: deque = deque(maxlen=mem_size)
        self.discount: float = discount

        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        if epsilon_stop_episode > 0:
            self.epsilon_decay: float = (self.epsilon - self.epsilon_min) / epsilon_stop_episode
        else:
            self.epsilon = self.epsilon_min
            self.epsilon_decay = 0.0

        self.loss: str = loss
        self.optimizer_lr: float = optimizer_lr

        if replay_start_size is None:
            replay_start_size = mem_size // 12
        if replay_start_size > mem_size:
            raise ValueError(f"replay_start_size ({replay_start_size}) must be <= mem_size ({mem_size})")
        if replay_start_size <= 0 and mem_size > 0:
            print("Warning: replay_start_size is 0 or less. Training may start immediately with few samples.")
        self.replay_start_size: int = replay_start_size

        self.model: Sequential = self._build_model()
        self.target_model: Sequential = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.target_update_freq: int = target_update_freq
        self.train_step_counter: int = 0

        if modelFile:
            self._load_model_from_file(modelFile)

    def _load_model_from_file(self, model_file_path: str) -> None:
        try:
            loaded_model = load_model(model_file_path)
            if loaded_model.input_shape[1:] != self.input_shape:
                print(
                    f"Warning: Loaded model input shape {loaded_model.input_shape[1:]} "
                    f"does not match environment's state shape {self.input_shape}."
                )
            self.model = loaded_model
            self.target_model.set_weights(self.model.get_weights())
            print(f"Model loaded successfully from {model_file_path}")
        except Exception as e:
            print(f"Error loading model from {model_file_path}: {e}")
            print("Proceeding with a new, untrained model.")

    def _build_model(self) -> Sequential:
        model = Sequential()
        for i, (filters, kernel_size, strides) in enumerate(self.conv_layers_params):
            if i == 0:
                model.add(Conv2D(filters, kernel_size, strides=strides, activation='relu',
                                 input_shape=self.input_shape, padding='same'))
            else:
                model.add(Conv2D(filters, kernel_size, strides=strides, activation='relu', padding='same'))
            if i < len(self.pooling_layers_config) and self.pooling_layers_config[i]:
                pool_param = self.pooling_layers_config[i]
                pool_size = (2, 2) if pool_param is True else pool_param
                model.add(MaxPooling2D(pool_size=pool_size))
            if i % 2 == 1:
                model.add(Dropout(0.15))
        model.add(Flatten())
        for i, neurons in enumerate(self.dense_neurons_list):
            model.add(Dense(neurons, activation=self.dense_activations_list[i]))
            if i == 0:
                model.add(Dropout(0.25))
        model.add(Dense(1, activation=self.dense_activations_list[-1]))
        custom_optimizer = Adam(learning_rate=self.optimizer_lr)
        model.compile(loss=self.loss, optimizer=custom_optimizer)
        return model

    def _update_target_model(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def add_to_memory(self, current_state: np.ndarray, next_state: np.ndarray,
                      reward: float, done: bool) -> None:
        self.memory.append((current_state, next_state, reward, done))

    def predict_value(self, state_matrix: Optional[np.ndarray]) -> np.ndarray:
        if state_matrix is None:
            return np.array([0.0], dtype=np.float32)
        state_np = state_matrix.reshape(1, *self.input_shape)
        return self.model.predict(state_np, verbose=0)[0]

    def best_state(self, next_possible_moves: Dict[Tuple[int, int], np.ndarray]) -> Optional[Tuple[int, int]]:
        if not next_possible_moves:
            return None
        if random.random() <= self.epsilon:
            return random.choice(list(next_possible_moves.keys()))
        else:
            actions = list(next_possible_moves.keys())
            board_matrices = list(next_possible_moves.values())
            all_states_np = np.array(board_matrices).reshape(-1, *self.input_shape)
            q_values = self.model.predict(all_states_np, verbose=0)
            best_action_idx = np.argmax(q_values[:, 0])
            return actions[best_action_idx]

    def train(self, batch_size: int = 32, epochs: int = 1) -> Optional[float]:
        if len(self.memory) < self.replay_start_size or len(self.memory) < batch_size:
            return None
        minibatch = random.sample(self.memory, batch_size)
        valid_batch = [(s, ns, r, d) for s, ns, r, d in minibatch if s is not None and ns is not None]
        if not valid_batch:
            return None
        current_states_matrices = np.array([item[0] for item in valid_batch]).reshape(-1, *self.input_shape)
        next_states_matrices = np.array([item[1] for item in valid_batch]).reshape(-1, *self.input_shape)
        rewards = np.array([item[2] for item in valid_batch], dtype=np.float32)
        dones = np.array([item[3] for item in valid_batch], dtype=np.bool_)
        q_next_target_model = self.target_model.predict(next_states_matrices, verbose=0)[:, 0]
        target_q_values = rewards + self.discount * q_next_target_model * (~dones)
        history = self.model.fit(current_states_matrices, target_q_values.reshape(-1, 1),
                                 batch_size=batch_size, epochs=epochs, verbose=0)
        if self.epsilon_decay > 0 and self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self._update_target_model()
        if history and 'loss' in history.history:
            return history.history['loss'][-1]
        return None

    def save_model(self, filepath: str) -> None:
        try:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")