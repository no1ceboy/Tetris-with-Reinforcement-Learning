import random
import cv2
import numpy as np
from PIL import Image
from time import sleep
from typing import List, Tuple, Dict, Any, Optional

# Type aliases for clarity
Board = List[List[int]]
PieceData = List[Tuple[int, int]] # List of (dx, dy) coordinates relative to pivot
RotationDict = Dict[int, PieceData] # {rotation_angle: piece_data}
TetrominoShapes = Dict[int, RotationDict] # {piece_id: rotation_dict}
Color = Tuple[int, int, int] # RGB color tuple

class Tetris:
    """
    Tetris game environment.
    Handles game logic, state representation, and rendering.
    """
    # --- Game Board Constants ---
    MAP_EMPTY: int = 0    # Represents an empty cell on the board
    MAP_BLOCK: int = 1    # Represents a landed block on the board
    MAP_PLAYER: int = 2   # Represents the currently falling player-controlled piece (for rendering only)

    BOARD_WIDTH: int = 10 # Width of the game board in cells
    BOARD_HEIGHT: int = 20 # Height of the game board in cells

    # --- Tetromino Definitions ---
    # Each key is a piece ID.
    # Each piece ID maps to a dictionary of rotations (0, 90, 180, 270 degrees).
    # Each rotation is a list of (col_offset, row_offset) tuples from the piece's pivot point.
    TETROMINOS: TetrominoShapes = {
        0: { # I piece
            0:   [(0,0), (1,0), (2,0), (3,0)],
            90:  [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)], # Or [(0,0), (1,0), (2,0), (3,0)] if mirrored
            270: [(1,3), (1,2), (1,1), (1,0)], # Or [(1,0), (1,1), (1,2), (1,3)] if mirrored
        },
        1: { # T piece
            0:   [(1,0), (0,1), (1,1), (2,1)],
            90:  [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L piece
            0:   [(1,0), (1,1), (1,2), (2,2)],
            90:  [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J piece
            0:   [(1,0), (1,1), (1,2), (0,2)],
            90:  [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z piece
            0:   [(0,0), (1,0), (1,1), (2,1)],
            90:  [(0,2), (0,1), (1,1), (1,0)], 
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)], 
        },
        5: { # S piece
            0:   [(2,0), (1,0), (1,1), (0,1)],
            90:  [(0,0), (0,1), (1,1), (1,2)], 
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)], 
        },
        6: { # O piece (same for all rotations)
            0:   [(1,0), (2,0), (1,1), (2,1)],
            90:  [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    # --- Color Definitions for Rendering ---
    COLORS: Dict[int, Color] = {
        MAP_EMPTY: (255, 255, 255),  # White
        MAP_BLOCK: (247, 64, 99),    # Reddish-pink for landed blocks
        MAP_PLAYER: (0, 167, 247),   # Blue for the falling piece
    }

    # Initial spawn position for new pieces (column, row) on the board
    INITIAL_SPAWN_POS: List[int] = [3, 0] # Default Tetris often spawns around col 3-4, row 0

    def __init__(self):
        self.cv2 = cv2  # Keep a reference to cv2 for run_model.py to call destroyAllWindows
        # Initialize game state variables
        self.board: Board  # The main game board (list of lists of ints)
        self.current_piece: int # ID of the currently falling piece
        self.current_rotation: int # Current rotation angle (0, 90, 180, 270) of the piece
        self.current_pos: List[int] # [column, row] of the piece's pivot on the board
        self.next_piece: int # ID of the next piece to fall
        self.bag: List[int] # For the 7-bag randomizer system to ensure piece variety
        self.score: int # Current game score
        self.game_over: bool # Flag indicating if the game has ended
        # Attributes for logging per episode
        self.lines_cleared_this_episode: int
        self.reward_accumulated_this_episode: float
        self.reset() # Initialize the game state

    def reset(self) -> np.ndarray:
        """Resets the game to its initial state and returns the initial board state matrix."""
        # Create an empty game board
        self.board = [[Tetris.MAP_EMPTY for _ in range(Tetris.BOARD_WIDTH)] for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.score = 0
        # Reset per-episode counters for logging
        self.lines_cleared_this_episode = 0
        self.reward_accumulated_this_episode = 0.0

        self._initialize_bag() # Initialize or refill the piece bag
        self.next_piece = self._draw_from_bag() # Get the first 'next' piece
        self._new_round() # Start the first round (spawns current_piece)
        return self._get_board_state_matrix(self.board) # Return the initial state as a NumPy array

    def _initialize_bag(self) -> None:
        """Initializes or refills the 7-bag randomizer with all tetromino types."""
        self.bag = list(range(len(Tetris.TETROMINOS))) # Get all piece IDs
        random.shuffle(self.bag) # Shuffle them

    def _draw_from_bag(self) -> int:
        """Draws a piece ID from the bag, refilling the bag if it's empty."""
        if not self.bag: # If the bag is empty
            self._initialize_bag() # Refill and reshuffle
        return self.bag.pop() # Return and remove the last piece ID from the bag

    def _get_rotated_piece_coords(self) -> PieceData:
        """Returns the relative (dx, dy) coordinates of the current piece's current rotation."""
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]

    def _get_board_state_matrix(self, board_to_convert: Board) -> np.ndarray:
        """
        Converts a game board (list of lists) to a NumPy array (H, W, 1)
        suitable for CNN input. Values are 0.0 for empty, 1.0 for landed blocks.
        MAP_PLAYER is treated as empty for the state matrix as it's not a fixed block.
        """
        state_matrix = np.array(board_to_convert, dtype=np.float32)
        # Ensure only MAP_EMPTY (0) and MAP_BLOCK (1) exist in the state matrix.
        state_matrix[state_matrix == Tetris.MAP_PLAYER] = Tetris.MAP_EMPTY
        return state_matrix.reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 1)

    def _get_render_board(self) -> Board:
        """
        Returns a copy of the board with the current falling piece (MAP_PLAYER) drawn on it.
        This board is used solely for rendering purposes.
        """
        render_b = [row[:] for row in self.board] # Create a deep copy of the current board
        piece_coords = self._get_rotated_piece_coords()
        for dx, dy in piece_coords: # Iterate over the blocks of the current piece
            col, row = self.current_pos[0] + dx, self.current_pos[1] + dy
            # Draw the piece block if it's within board boundaries
            if 0 <= row < Tetris.BOARD_HEIGHT and 0 <= col < Tetris.BOARD_WIDTH:
                # Only draw on empty cells to avoid overwriting landed blocks visually before solidification
                if render_b[row][col] == Tetris.MAP_EMPTY:
                    render_b[row][col] = Tetris.MAP_PLAYER
        return render_b

    def get_game_score(self) -> int:
        """Returns the current cumulative game score."""
        return self.score

    def _new_round(self) -> None:
        """
        Starts a new round: sets the current piece to the next piece, gets a new next piece,
        and places the current piece at the initial spawn position.
        Sets game_over to True if the new piece collides immediately upon spawning (board is full at top).
        """
        self.current_piece = self.next_piece # The 'next' piece becomes the 'current'
        self.next_piece = self._draw_from_bag() # Get a new 'next' piece
        self.current_pos = list(Tetris.INITIAL_SPAWN_POS) # Reset position to spawn point (make a copy)
        self.current_rotation = 0 # Reset rotation

        # Check for collision immediately after spawning the new piece
        if self._check_collision(self._get_rotated_piece_coords(), self.current_pos):
            self.game_over = True # If collision, game over

    def _check_collision(self, piece_coords: PieceData, piece_pos: List[int]) -> bool:
        """
        Checks if the given piece_coords at piece_pos collides with board boundaries
        or existing MAP_BLOCKs on the main game board (self.board).
        """
        for dx, dy in piece_coords: # Iterate over each block of the piece
            col, row = piece_pos[0] + dx, piece_pos[1] + dy # Calculate absolute board coordinates
            # Check for out-of-bounds collision
            if not (0 <= col < Tetris.BOARD_WIDTH and 0 <= row < Tetris.BOARD_HEIGHT):
                return True  # Collision with boundary
            # Check for collision with an existing landed block on the board
            if self.board[row][col] == Tetris.MAP_BLOCK:
                return True
        return False # No collision detected

    def _add_piece_to_board_copy(self, board_copy: Board, piece_coords: PieceData, piece_pos: List[int]) -> Board:
        """
        Adds a piece to a *copy* of the board by marking its cells as MAP_BLOCK.
        Modifies and returns the board copy. Used for simulating piece placement.
        """
        for dx, dy in piece_coords: # Iterate over each block of the piece
            col, row = piece_pos[0] + dx, piece_pos[1] + dy # Calculate absolute board coordinates
            # Place the block if within bounds
            if 0 <= row < Tetris.BOARD_HEIGHT and 0 <= col < Tetris.BOARD_WIDTH:
                board_copy[row][col] = Tetris.MAP_BLOCK # Solidify the piece by marking as MAP_BLOCK
        return board_copy

    def _clear_lines_on_board(self, board_to_modify: Board) -> Tuple[int, Board]:
        """
        Clears completed lines in the given board (modifies it in place by returning a new board).
        Returns the number of lines cleared and the modified board.
        """
        lines_cleared_count = 0
        # Create a new list to hold the rows of the board after clearing lines
        final_board_rows: List[List[int]] = []

        # Iterate through the board rows from bottom to top
        for r_idx in range(Tetris.BOARD_HEIGHT - 1, -1, -1):
            # Check if the current row is full (all cells are MAP_BLOCK)
            if all(cell == Tetris.MAP_BLOCK for cell in board_to_modify[r_idx]):
                lines_cleared_count += 1 # Increment lines cleared count
            else:
                # If the row is not full, add it to the top of our new board structure
                final_board_rows.insert(0, board_to_modify[r_idx])
        
        # Add new empty rows at the top for each line cleared
        for _ in range(lines_cleared_count):
            final_board_rows.insert(0, [Tetris.MAP_EMPTY] * Tetris.BOARD_WIDTH)
        
        return lines_cleared_count, final_board_rows


    def get_next_states(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Generates all possible board states resulting from valid moves of the current piece.
        A "move" is defined by a target column (for the piece's pivot) and a rotation angle.
        For each move, the piece is simulated to drop to its lowest possible position.
        The resulting board state (after placement and line clearing) is then calculated.

        Returns:
            A dictionary where keys are action tuples (target_column, rotation_angle)
            and values are the resulting board state matrices (np.ndarray HxWx1).
        """
        possible_states: Dict[Tuple[int, int], np.ndarray] = {}
        current_piece_id = self.current_piece

        # Determine relevant rotations to try for the current piece type
        # (e.g., 'O' piece has 1 unique rotation, 'I/S/Z' effectively 2, others 4)
        if current_piece_id == 6:  # O piece
            rotations_to_try = [0]
        elif current_piece_id in [0, 4, 5]:  # I, S, Z pieces
            rotations_to_try = [0, 90] # These often have only 2 distinct shapes after 180/270 deg rotations
        else:  # T, L, J pieces
            rotations_to_try = [0, 90, 180, 270]

        for rotation in rotations_to_try: # Iterate through each valid rotation
            rotated_piece_coords = Tetris.TETROMINOS[current_piece_id][rotation]
            
            # Determine the horizontal extents of the piece to find valid columns for placement
            min_col_offset = min(p[0] for p in rotated_piece_coords) # How far left the piece extends from pivot
            max_col_offset = max(p[0] for p in rotated_piece_coords) # How far right

            # Iterate through all possible columns where the piece's pivot could be placed
            # such that the entire piece remains within horizontal board boundaries.
            for target_col_for_pivot in range(-min_col_offset, Tetris.BOARD_WIDTH - max_col_offset):
                # Simulate dropping the piece from the top for this (column, rotation)
                sim_pos = [target_col_for_pivot, 0] # Start at top row, specified column for pivot
                
                # Move piece down one step at a time until it would collide one step below current
                # We check collision with the *current self.board* (board before this piece lands)
                while not self._check_collision(rotated_piece_coords, [sim_pos[0], sim_pos[1] + 1]):
                    sim_pos[1] += 1 # Move down
                    # Safety break: if the piece's lowest point (considering its shape) is at board bottom
                    # This prevents infinite loops if collision logic is imperfect for bottom edge.
                    max_row_offset_from_pivot = 0
                    if rotated_piece_coords: # ensure not empty
                         max_row_offset_from_pivot = max(p[1] for p in rotated_piece_coords)
                    if sim_pos[1] + max_row_offset_from_pivot >= Tetris.BOARD_HEIGHT -1 :
                        break
                
                # Now, sim_pos[1] is the final row for the pivot before collision or at bottom.
                # Create a temporary board copy to simulate placing the piece and clearing lines.
                temp_board = [row[:] for row in self.board] # Deep copy current board
                temp_board_after_placement = self._add_piece_to_board_copy(temp_board, rotated_piece_coords, sim_pos)
                
                # Simulate line clearing on this temporary board
                _, board_after_clear = self._clear_lines_on_board(temp_board_after_placement)
                
                # The action is defined by the initial column of the pivot and the rotation
                action_tuple = (target_col_for_pivot, rotation)
                # Store the resulting board state matrix for this action
                possible_states[action_tuple] = self._get_board_state_matrix(board_after_clear)
        
        return possible_states

    def get_state_size(self) -> Tuple[int, int, int]:
        """Returns the shape of the board state matrix (height, width, channels), used for DQNAgent input_shape."""
        return (Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 1)

    def play(self, piece_target_col: int, rotation_angle: int,
             render: bool = False, render_delay: Optional[float] = None
             ) -> Tuple[np.ndarray, float, bool]:
        """
        Executes a play:
        1. Sets the current piece's rotation and initial column (piece_target_col).
        2. Drops the piece to its final resting position.
        3. Updates the main game board (self.board) with the landed piece.
        4. Clears any completed lines and updates self.board again.
        5. Calculates the game score increase and the reward for the agent for this move.
        6. Starts a new round (spawns the next piece).
        7. Checks for game over condition.

        Args:
            piece_target_col: The target column for the piece's pivot point when it starts falling.
            rotation_angle: The target rotation (0, 90, 180, 270) for the piece.
            render: Whether to render intermediate steps of the piece falling (for visualization).
            render_delay: Delay in seconds between renders if rendering is enabled.

        Returns:
            A tuple containing:
                - new_board_state_matrix (np.ndarray): State of the board *after* the move and *after* the next piece has spawned. This is the S' for the *next* decision.
                - reward_for_this_move (float): The reward obtained by the agent from this specific action.
                - game_over_flag (bool): True if the game ended as a result of this move or the new piece spawning, False otherwise.
        """
        # Set the piece's initial horizontal position (column of pivot) and rotation
        self.current_pos = [piece_target_col, 0] # Start at top row, specified column
        self.current_rotation = rotation_angle
        rotated_piece_coords = self._get_rotated_piece_coords() # Get current shape

        # --- Drop the piece ---
        # Move down as long as the position one step below is not a collision
        while not self._check_collision(rotated_piece_coords, [self.current_pos[0], self.current_pos[1] + 1]):
            self.current_pos[1] += 1 # Move piece down by one row
            if render: # If rendering is enabled for this move
                self.render()
                if render_delay:
                    sleep(render_delay)
            # Safety break: if the piece's lowest block is at the very bottom row
            max_row_offset = 0
            if rotated_piece_coords: max_row_offset = max(p[1] for p in rotated_piece_coords)
            if self.current_pos[1] + max_row_offset >= Tetris.BOARD_HEIGHT -1:
                 break
        
        # --- Solidify piece onto the main board ---
        # self.current_pos[1] is now the final row for the pivot
        self.board = self._add_piece_to_board_copy(self.board, rotated_piece_coords, self.current_pos)
        
        # --- Clear lines and update board ---
        lines_cleared_this_move, self.board = self._clear_lines_on_board(self.board)
        self.lines_cleared_this_episode += lines_cleared_this_move # Accumulate lines for episode logging

        # --- Update Game Score (as per original game logic) ---
        current_move_game_score_increase = 1  # Base score for successfully placing a piece
        if lines_cleared_this_move > 0:
            # Add bonus score for clearing lines (e.g., (lines_cleared^2) * board_width)
            current_move_game_score_increase += (lines_cleared_this_move ** 2) * Tetris.BOARD_WIDTH
        self.score += current_move_game_score_increase # Add to total game score

        # --- Calculate Agent's Reward for this move ---
        # This reward is what the agent uses for learning. It can be shaped differently from game score.
        reward_for_this_move = 0.1  # Small positive reward for surviving/making a move

        # Reward shaping based on lines cleared
        if lines_cleared_this_move == 1: reward_for_this_move += 5
        elif lines_cleared_this_move == 2: reward_for_this_move += 15
        elif lines_cleared_this_move == 3: reward_for_this_move += 40
        elif lines_cleared_this_move >= 4: reward_for_this_move += 100 # Tetris!
        
        # --- Prepare for the next piece ---
        self._new_round() # This will set self.current_piece, self.next_piece, and importantly,
                          # it will set self.game_over if the newly spawned piece immediately collides.

        # --- Handle Game Over Condition ---
        if self.game_over: # If _new_round caused a game over (or if it was already game over)
            self.score -= 2 # Penalty to game score for game over
            reward_for_this_move = -20 # Large negative reward for the agent when game ends
        
        self.reward_accumulated_this_episode += reward_for_this_move # Accumulate reward for episode logging

        # Get the state matrix of the board *after* the new piece has spawned (for the next decision)
        new_board_state_matrix_for_agent = self._get_board_state_matrix(self.board)
        
        return new_board_state_matrix_for_agent, reward_for_this_move, self.game_over

    def get_lines_cleared_this_episode(self) -> int:
        """Gets the total number of lines cleared in the current episode."""
        return self.lines_cleared_this_episode

    def get_reward_accumulated_this_episode(self) -> float:
        """Gets the total reward accumulated by the agent in the current episode."""
        return self.reward_accumulated_this_episode

    def render(self, window_name: str = 'Tetris AI') -> None:
        """Renders the current game state using OpenCV and PIL."""
        # Get the board that includes the currently falling player piece for display
        display_board = self._get_render_board()

        # Convert the board data (integers) to an RGB color image
        img_pixels = [[Tetris.COLORS[cell_value] for cell_value in row] for row in display_board]
        img_np_array = np.array(img_pixels, dtype=np.uint8) # Convert to NumPy array

        # Use PIL for resizing with NEAREST resampling (to keep pixels sharp)
        img_pil = Image.fromarray(img_np_array, 'RGB') # PIL uses RGB order
        scale_factor = 25 # Scale factor to make the game board larger on screen
        img_resized_pil = img_pil.resize(
            (Tetris.BOARD_WIDTH * scale_factor, Tetris.BOARD_HEIGHT * scale_factor),
            Image.Resampling.NEAREST
        )
        img_resized_np = np.array(img_resized_pil) # Convert back to NumPy array for OpenCV

        # Add score text to the image
        cv2.putText(img_resized_np, f"Score: {self.score}", (10, 20), # Position of text
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA) # Font, scale, color, thickness

        # Display the image in an OpenCV window
        cv2.imshow(window_name, img_resized_np)
        cv2.waitKey(1) # Essential for imshow to work correctly in a loop and process GUI events