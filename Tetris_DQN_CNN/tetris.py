import random
import cv2
import numpy as np
from PIL import Image
from time import sleep

class Tetris:
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: {0: [(0,0),(1,0),(2,0),(3,0)], 90: [(1,0),(1,1),(1,2),(1,3)], 180: [(3,0),(2,0),(1,0),(0,0)], 270: [(1,3),(1,2),(1,1),(1,0)]},
        1: {0: [(1,0),(0,1),(1,1),(2,1)], 90: [(0,1),(1,2),(1,1),(1,0)], 180: [(1,2),(2,1),(1,1),(0,1)], 270: [(2,1),(1,0),(1,1),(1,2)]},
        2: {0: [(1,0),(1,1),(1,2),(2,2)], 90: [(0,1),(1,1),(2,1),(2,0)], 180: [(1,2),(1,1),(1,0),(0,0)], 270: [(2,1),(1,1),(0,1),(0,2)]},
        3: {0: [(1,0),(1,1),(1,2),(0,2)], 90: [(0,1),(1,1),(2,1),(2,2)], 180: [(1,2),(1,1),(1,0),(2,0)], 270: [(2,1),(1,1),(0,1),(0,0)]},
        4: {0: [(0,0),(1,0),(1,1),(2,1)], 90: [(0,2),(0,1),(1,1),(1,0)], 180: [(2,1),(1,1),(1,0),(0,0)], 270: [(1,0),(1,1),(0,1),(0,2)]},
        5: {0: [(2,0),(1,0),(1,1),(0,1)], 90: [(0,0),(0,1),(1,1),(1,2)], 180: [(0,1),(1,1),(1,0),(2,0)], 270: [(1,2),(1,1),(0,1),(0,0)]},
        6: {0: [(1,0),(2,0),(1,1),(2,1)], 90: [(1,0),(2,0),(1,1),(2,1)], 180: [(1,0),(2,0),(1,1),(2,1)], 270: [(1,0),(2,0),(1,1),(2,1)]}
    }

    COLORS = {0: (255,255,255), 1: (247,64,99), 2: (0,167,247)}

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[Tetris.MAP_EMPTY] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.score = 0
        self.lines_cleared_this_episode = 0
        self.reward_accumulated_this_episode = 0.0
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        return self._get_board_state_matrix(self.board)

    def _get_rotated_piece(self):
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]

    def _get_complete_board(self):
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            if 0 <= y < Tetris.BOARD_HEIGHT and 0 <= x < Tetris.BOARD_WIDTH:
                board[y][x] = Tetris.MAP_PLAYER
        return board

    def get_game_score(self):
        return self.score

    def get_lines_cleared_this_episode(self):
        return self.lines_cleared_this_episode

    def get_reward_accumulated_this_episode(self):
        return self.reward_accumulated_this_episode

    def _new_round(self):
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0
        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True

    def _check_collision(self, piece, pos):
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH or y < 0 or y >= Tetris.BOARD_HEIGHT or self.board[y][x] == Tetris.MAP_BLOCK:
                return True
        return False

    def _add_piece_to_board(self, piece, pos):
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
        return board

    def _clear_lines(self, board):
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board

    def _number_of_holes(self, board):
        holes = 0
        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            holes += len([x for x in col[i+1:] if x == Tetris.MAP_EMPTY])
        return holes

    def _bumpiness(self, board):
        min_ys = []
        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            min_ys.append(i)
        total_bumpiness = sum(abs(min_ys[i] - min_ys[i+1]) for i in range(len(min_ys) - 1))
        max_bumpiness = max([abs(min_ys[i] - min_ys[i+1]) for i in range(len(min_ys) - 1)] or [0])
        return total_bumpiness, max_bumpiness

    def _height(self, board):
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT
        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == Tetris.MAP_EMPTY:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            max_height = max(max_height, height)
            min_height = min(min_height, height)
        return sum_height, max_height, min_height

    def _get_board_state_matrix(self, board_to_convert):
        state_matrix = np.zeros((Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 4), dtype=np.float32)
        for r in range(Tetris.BOARD_HEIGHT):
            for c in range(Tetris.BOARD_WIDTH):
                if board_to_convert[r][c] == Tetris.MAP_BLOCK:
                    state_matrix[r, c, 0] = 1.0
                if board_to_convert[r][c] == Tetris.MAP_PLAYER:
                    state_matrix[r, c, 1] = 1.0
        col_heights = np.zeros((Tetris.BOARD_WIDTH,), dtype=np.float32)
        for c in range(Tetris.BOARD_WIDTH):
            for r in range(Tetris.BOARD_HEIGHT):
                if board_to_convert[r][c] != Tetris.MAP_EMPTY:
                    col_heights[c] = (Tetris.BOARD_HEIGHT - r) / Tetris.BOARD_HEIGHT
                    break
        for r in range(Tetris.BOARD_HEIGHT):
            state_matrix[r, :, 2] = col_heights[:]
        holes = sum(
            1 for c in range(Tetris.BOARD_WIDTH)
            for r in range(1, Tetris.BOARD_HEIGHT)
            if board_to_convert[r][c] == Tetris.MAP_EMPTY and any(board_to_convert[r2][c] == Tetris.MAP_BLOCK for r2 in range(r))
        )
        holes_norm = holes / (Tetris.BOARD_WIDTH * Tetris.BOARD_HEIGHT)
        state_matrix[:, :, 3] = holes_norm
        return state_matrix

    def get_next_states(self):
        possible_states = {}
        piece_id = self.current_piece
        if piece_id == 6:
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    board_matrix = self._get_board_state_matrix(board)
                    possible_states[(x, rotation)] = board_matrix
        return possible_states

    def get_state_size(self):
        return (Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 4)

    def play(self, x, rotation, render=False, render_delay=None):
        self.current_pos = [x, 0]
        self.current_rotation = rotation
        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        # ----- Reward shaping tối ưu -----
        reward = 1.0  # Đặt thành công mảnh nào cũng có thưởng
        if lines_cleared == 1:
            reward += 50
        elif lines_cleared == 2:
            reward += 150
        elif lines_cleared == 3:
            reward += 400
        elif lines_cleared >= 4:
            reward += 1000
        holes = self._number_of_holes(self.board)
        sum_height, _, _ = self._height(self.board)
        total_bumpiness, _ = self._bumpiness(self.board)
        reward -= holes * 0.5
        reward -= max(0, sum_height - 12) * 0.2
        reward -= total_bumpiness * 0.1

        self.score += 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        self.lines_cleared_this_episode += lines_cleared
        self.reward_accumulated_this_episode += reward

        self._new_round()
        if self.game_over:
            reward = -20  # Penalty khi chết, không quá lớn, không quá nhỏ
        next_state = self._get_board_state_matrix(self.board)
        return next_state, reward, self.game_over

    def render(self):
        img = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25), Image.NEAREST)
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)