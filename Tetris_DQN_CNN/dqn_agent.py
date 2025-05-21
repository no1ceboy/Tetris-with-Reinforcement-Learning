from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten # Dropout can be added if desired
from keras.optimizers import Adam # Or from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
from typing import Tuple, List, Union, Optional, Dict, Any # Data types for type hinting

class DQNAgent:
    """
    Deep Q-Network Agent for reinforcement learning.
    This agent learns to play games by interacting with an environment.
    """
    def __init__(self,
                 input_shape: Tuple[int, int, int],  # Input shape of the state (e.g., (height, width, channels))
                 mem_size: int = 20000,              # Maximum size of the replay memory
                 discount: float = 0.95,             # Discount factor (gamma) for future rewards
                 epsilon: float = 1.0,               # Initial exploration rate (epsilon-greedy)
                 epsilon_min: float = 0.01,          # Minimum exploration rate
                 epsilon_stop_episode: int = 1500,   # Episode after which epsilon stops decaying
                 # Neural network layer configurations (with default values if not provided)
                 conv_layers: Optional[List[Tuple[int, Tuple[int, int], Tuple[int, int]]]] = None,
                 pooling_layers: Optional[List[Union[bool, Tuple[int, int]]]] = None,
                 dense_neurons: Optional[List[int]] = None,
                 dense_activations: Optional[List[str]] = None,
                 loss: str = 'mse',                       # Loss function (typically Mean Squared Error for Q-learning)
                 optimizer_lr: float = 0.00025,           # Learning rate for the Adam optimizer
                 replay_start_size: Optional[int] = None, # Minimum experiences in memory before training starts
                 modelFile: Optional[str] = None,         # Path to a pre-trained model file (if any)
                 target_update_freq: int = 200):          # Frequency (in training steps) to update the target network
        """
        Initializes the DQNAgent.

        Args:
            input_shape: Shape of the input state (height, width, channels).
            mem_size: Maximum size of the replay memory.
            discount: Discount factor (gamma) for future rewards.
            epsilon: Initial exploration rate.
            epsilon_min: Minimum exploration rate.
            epsilon_stop_episode: Episode at which epsilon stops decaying.
            conv_layers: Configuration for convolutional layers.
                         Example: [(64, (4,4), (1,1)), (128, (3,3), (1,1))] (filters, kernel_size, strides)
                         Default: [(64, (4,4), (1,1)), (128, (3,3), (1,1))]
            pooling_layers: Configuration for pooling layers after conv layers.
                            True for (2,2) pool, specific tuple for other sizes, False for no pool.
                            Default: [True, True]
            dense_neurons: List of neuron counts for dense layers. Example: [512]
                           Default: [512]
            dense_activations: List of activation functions for dense layers.
                               Example: ['relu', 'linear'] (last one for output Q-value)
                               Default: ['relu', 'linear']
            loss: Loss function for the Keras model.
            optimizer_lr: Learning rate for the Adam optimizer.
            replay_start_size: Minimum experiences in memory before training starts.
                               Default: mem_size // 20.
            modelFile: Path to a pre-trained Keras model file.
            target_update_freq: Frequency (in training steps) to update the target network.
        """

        # --- Validate input parameters ---
        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError("input_shape must be a tuple of length 3 (height, width, channels)")
        self.input_shape: Tuple[int, int, int] = input_shape

        # Assign default layer configurations if not provided
        self.conv_layers_params: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = \
            conv_layers if conv_layers is not None else [(64, (4,4), (1,1)), (128, (3,3), (1,1))]
        self.pooling_layers_config: List[Union[bool, Tuple[int, int]]] = \
            pooling_layers if pooling_layers is not None else [True, True]
        self.dense_neurons_list: List[int] = dense_neurons if dense_neurons is not None else [512]
        self.dense_activations_list: List[str] = \
            dense_activations if dense_activations is not None else ['relu', 'linear']

        # Validate consistency between dense layers and their activations
        if len(self.dense_activations_list) != len(self.dense_neurons_list) + 1:
            raise ValueError(
                f"dense_activations (len {len(self.dense_activations_list)}) and "
                f"dense_neurons (len {len(self.dense_neurons_list)}) do not match. "
                "Expected len(dense_activations) == len(dense_neurons) + 1."
            )
        # Validate that neuron counts are positive
        if any(n <= 0 for n in self.dense_neurons_list):
            raise ValueError("dense_neurons must contain positive integers")

        # Validate reinforcement learning hyperparameters
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

        # --- Initialize agent attributes ---
        self.mem_size: int = mem_size
        self.memory: deque = deque(maxlen=mem_size) # Replay memory using deque for automatic old experience removal
        self.discount: float = discount             # Gamma, discount factor for future rewards

        self.epsilon: float = epsilon               # Current epsilon for epsilon-greedy strategy
        self.epsilon_min: float = epsilon_min       # Minimum epsilon value
        if epsilon_stop_episode > 0: # Calculate epsilon decay per episode
            self.epsilon_decay: float = (self.epsilon - self.epsilon_min) / epsilon_stop_episode
        else: # If no decay, epsilon stays at its minimum value
            self.epsilon = self.epsilon_min
            self.epsilon_decay = 0.0

        self.loss: str = loss # Loss function used for training the network
        self.optimizer_lr: float = optimizer_lr # Learning rate of the optimizer

        # Determine the replay buffer size required before starting training
        if replay_start_size is None: # If not provided, default to 1/20 of memory size
            replay_start_size = mem_size // 20
        if replay_start_size > mem_size: # Replay start size cannot be larger than memory size
            raise ValueError(f"replay_start_size ({replay_start_size}) must be <= mem_size ({mem_size})")
        # Warning if replay_start_size is too small, potentially leading to training on less diverse data
        if replay_start_size <= 0 and mem_size > 0:
            # This print statement will appear on the console if the object is created with replay_start_size <= 0
            print("Warning: replay_start_size is 0 or less. Training may start immediately with few samples.")
        self.replay_start_size: int = replay_start_size

        # Build the main model (Q-network) and the target model (Target Q-network)
        self.model: Sequential = self._build_model()
        self.target_model: Sequential = self._build_model()
        # Initialize the target_model's weights to be the same as the main model's for initial stability
        self.target_model.set_weights(self.model.get_weights())

        self.target_update_freq: int = target_update_freq # Frequency to update target model weights
        self.train_step_counter: int = 0 # Counter for the number of times train() has been called, used for target model updates

        # Load model from file if a path is provided
        if modelFile:
            self._load_model_from_file(modelFile)

    def _load_model_from_file(self, model_file_path: str) -> None:
        """Loads a model from a file and updates both the main model and target_model."""
        try:
            loaded_model = load_model(model_file_path)
            # Check if the loaded model's input_shape matches the current configuration
            if loaded_model.input_shape[1:] != self.input_shape:
                # This print statement will appear on the console if there's a mismatch
                print(
                    f"Warning: Loaded model input shape {loaded_model.input_shape[1:]} "
                    f"does not match environment's state shape {self.input_shape}."
                )
            self.model = loaded_model
            self.target_model.set_weights(self.model.get_weights()) # Synchronize target model with the loaded model
            # This print statement will appear on the console upon successful loading
            print(f"Model loaded successfully from {model_file_path}")
        except Exception as e: # Handle errors during model loading
            # This print statement will appear on the console if an error occurs
            print(f"Error loading model from {model_file_path}: {e}")
            print("Proceeding with a new, untrained model.")

    def _build_model(self) -> Sequential:
        """Builds and compiles the Keras CNN model."""
        model = Sequential() # Create a sequential model

        # Add Convolutional (CNN) layers to extract features from the state (game board image)
        for i, (filters, kernel_size, strides) in enumerate(self.conv_layers_params):
            if i == 0: # The first conv layer needs input_shape specified
                model.add(Conv2D(filters, kernel_size, strides=strides, activation='relu',
                                 input_shape=self.input_shape, padding='same'))
            else: # Subsequent conv layers do not need input_shape
                model.add(Conv2D(filters, kernel_size, strides=strides, activation='relu', padding='same'))

            # Add MaxPooling layer after each Conv layer if configured
            if i < len(self.pooling_layers_config) and self.pooling_layers_config[i]:
                pool_param = self.pooling_layers_config[i]
                pool_size = (2, 2) if pool_param is True else pool_param # Default to (2,2) if True
                model.add(MaxPooling2D(pool_size=pool_size))
                # model.add(Dropout(0.2)) # Optional: add Dropout to reduce overfitting

        model.add(Flatten()) # Flatten the output from conv/pool layers to feed into fully-connected (Dense) layers

        # Add Dense (fully-connected) layers
        for i, neurons in enumerate(self.dense_neurons_list):
            model.add(Dense(neurons, activation=self.dense_activations_list[i]))
            # model.add(Dropout(0.3)) # Optional: add Dropout

        # Output layer: 1 neuron with a linear activation function (no activation or linear activation)
        # to predict the Q-value of the input state.
        # This model predicts Q(S), the value of being in a state S,
        # not Q(S,a) for all actions 'a' simultaneously.
        model.add(Dense(1, activation=self.dense_activations_list[-1]))

        # Compile the model: specify optimizer and loss function
        custom_optimizer = Adam(learning_rate=self.optimizer_lr) # Use Adam optimizer
        model.compile(loss=self.loss, optimizer=custom_optimizer) # Loss function is Mean Squared Error (mse)
        return model

    def _update_target_model(self) -> None:
        """Copies weights from the main model (self.model) to the target model (self.target_model)."""
        # This helps stabilize learning by keeping the target Q-values fixed for a period.
        self.target_model.set_weights(self.model.get_weights())

    def add_to_memory(self, current_state: np.ndarray, next_state: np.ndarray,
                      reward: float, done: bool) -> None:
        """Adds an experience to the replay memory."""
        # An experience consists of (current state, next state after action, reward received, flag indicating if episode ended)
        self.memory.append((current_state, next_state, reward, done))

    def predict_value(self, state_matrix: Optional[np.ndarray]) -> np.ndarray:
        """Predicts the Q-value for a given state (state_matrix) using the main model (self.model)."""
        if state_matrix is None: # Handle the rare case of None input state (should be handled by caller)
            return np.array([0.0], dtype=np.float32)
        # Reshape the state to (1, height, width, channels) for model input
        state_np = state_matrix.reshape(1, *self.input_shape)
        # Return the Q-value (as a 1-element array)
        return self.model.predict(state_np, verbose=0)[0]

    def best_state(self, next_possible_moves: Dict[Tuple[int, int], np.ndarray]) -> Optional[Tuple[int, int]]:
        """
        Selects the best action (state here refers to (column, rotation)) based on the epsilon-greedy strategy.
        `next_possible_moves` is a dictionary: {action: resulting_board_state_matrix}.
        """
        if not next_possible_moves: # If no possible moves
            return None

        if random.random() <= self.epsilon: # Exploration phase
            return random.choice(list(next_possible_moves.keys())) # Choose a random action
        else: # Exploitation phase
            actions = list(next_possible_moves.keys())
            # The values in `next_possible_moves` are the board state matrices (results of actions)
            board_matrices = list(next_possible_moves.values())

            # Ensure all matrices have the correct shape
            all_states_np = np.array(board_matrices).reshape(-1, *self.input_shape)

            # Predict Q-values for all possible resulting states using the main model
            q_values = self.model.predict(all_states_np, verbose=0) # Shape: (num_actions, 1)

            # Choose the action that leads to the state with the highest Q-value
            best_action_idx = np.argmax(q_values[:, 0])
            return actions[best_action_idx]

    def train(self, batch_size: int = 32, epochs: int = 1) -> Optional[float]:
        """
        Trains the main model using a batch of experiences from the replay memory.
        Returns the training loss for the batch, or None if not trained.
        """
        # Only train if there's enough experience in memory
        if len(self.memory) < self.replay_start_size or len(self.memory) < batch_size:
            return None

        # Sample a random minibatch from memory (Experience Replay)
        minibatch = random.sample(self.memory, batch_size)
        
        # Filter out experiences with None states (should be rare if add_to_memory is used correctly)
        valid_batch = [(s, ns, r, d) for s, ns, r, d in minibatch if s is not None and ns is not None]
        if not valid_batch: # If no valid experiences in the batch
            return None

        # Prepare data for training
        # current_states_matrices: (batch_size, H, W, C)
        current_states_matrices = np.array([item[0] for item in valid_batch]).reshape(-1, *self.input_shape)
        # next_states_matrices: (batch_size, H, W, C)
        next_states_matrices = np.array([item[1] for item in valid_batch]).reshape(-1, *self.input_shape)
        # rewards: (batch_size,)
        rewards = np.array([item[2] for item in valid_batch], dtype=np.float32)
        # dones: (batch_size,) boolean array
        dones = np.array([item[3] for item in valid_batch], dtype=np.bool_)

        # Predict Q-values for the next states using the TARGET MODEL
        # q_next_target_model will have shape (batch_size,) after [:, 0]
        q_next_target_model = self.target_model.predict(next_states_matrices, verbose=0)[:, 0]
        
        # Calculate target Q-values using the Bellman equation:
        # Q_target = R + gamma * max_a' Q_target(S', a')
        # Since our model outputs Q(S) directly, q_next_target_model represents Q_target(S').
        # If 'done' is True, the future reward component is zero.
        target_q_values = rewards + self.discount * q_next_target_model * (~dones)

        # Train the main model (self.model) to approximate these target_q_values
        # target_q_values need to be reshaped to (batch_size, 1) for Keras model.fit
        history = self.model.fit(current_states_matrices, target_q_values.reshape(-1, 1),
                                 batch_size=batch_size, epochs=epochs, verbose=0)

        # Decay epsilon after training
        if self.epsilon_decay > 0 and self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon) # Ensure epsilon doesn't go below min

        # Periodically update the target model
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self._update_target_model()
            # This print statement can be used for debugging target model updates
            # print(f"Target model updated at train step {self.train_step_counter}")

        # Return the loss from the training history
        if history and 'loss' in history.history:
            return history.history['loss'][-1]
        return None # Return None if training did not occur or loss is unavailable

    def save_model(self, filepath: str) -> None:
        """Saves the main Keras model to a file."""
        try:
            self.model.save(filepath)
            # This print statement confirms model saving
            print(f"Model saved to {filepath}")
        except Exception as e: # Handle errors during model saving
            # This print statement indicates an error during saving
            print(f"Error saving model to {filepath}: {e}")