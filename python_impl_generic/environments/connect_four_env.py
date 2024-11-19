import numpy as np
from numba import jit

import helpers


def create_env_func():
    return ConnectFour()


ROWS = 6
COLS = 7
WINCNT = 4

def ___create_check_ranges():
    rw1 = ROWS - WINCNT + 1
    cw1 = COLS - WINCNT + 1
    return np.array([
        [0, ROWS,          0,  cw1, 0,  1],
        [0,  rw1,          0, COLS, 1,  0],
        [0,  rw1,          0,  cw1, 1,  1],
        [0,  rw1, WINCNT - 1, COLS, 1, -1],
    ])
CHECK_RANGES = ___create_check_ranges()


class HashableState:
    def __init__(self, board, player_idx) -> None:
        self.board = board
        self.player_idx = player_idx
        self.h = hash((board.data.tobytes(), player_idx))

    def get_valid_actions_mask(self):
        board = self.board
        return 1 - board[0, :, 0] - board[0, :, 1]

    def flip_horizontally(self):
        return HashableState(self.board[:, ::-1, :].copy(), self.player_idx)

    def __eq__(self, another):
        return self.board.shape == another.board.shape \
            and np.allclose(self.board, another.board) \
            and self.player_idx == another.player_idx

    def __hash__(self):
        return self.h

    def get_rotated_board(self):
        """
        Make current player to have coins at `board[:, :, 0]`, and opponent at `board[:, :, 1]`
        """
        board = self.board
        if self.player_idx == 1:
            board = np.flip(board, axis=2).copy()
        return board


@jit(nopython=True)
def check_win(board: np.ndarray, pi: int):
    for sr, er, sc, ec, dr, dc in CHECK_RANGES:
        A = board[sr : er, sc : ec, pi].copy()
        for _ in range(1, WINCNT):
            sr += dr
            er += dr
            sc += dc
            ec += dc
            A += board[sr : er, sc : ec, pi]
        if np.max(A) == WINCNT:
            return True

    return False


class ConnectFour(helpers.BaseEnv):
    state: HashableState
    done: bool
    action_mask: np.ndarray
    move_counter: int

    def __init__(self):
        self.reset()

    def copy(self):
        env = ConnectFour()
        env.state = self.state
        env.done = self.done
        env.action_mask = self.action_mask.copy()
        env.move_counter = self.move_counter
        return env

    def reset(self):
        self.state = HashableState(np.zeros((ROWS, COLS, 2), dtype=np.int32), 0)
        self.done = False
        self.action_mask = np.ones(COLS, dtype=np.int32)
        self.move_counter = 0
        return self.state, self.action_mask

    def get_rotated_encoded_state(self):
        return [self.state.get_rotated_board().astype(np.float32)]

    def get_rotated_encoded_states_with_symmetry__value_model(self):
        encs = self.get_rotated_encoded_state()
        return [
            encs,
        ]

    def get_rotated_encoded_states_with_symmetry__q_value_model(self, action_values):
        "Should return [*encoded env, action_values, action mask]"
        assert action_values.shape == self.action_mask.shape

        enc = self.get_rotated_encoded_state()[0]
        action_mask = np.copy(self.get_valid_actions_mask())

        return [
            [enc, action_values, action_mask],
        ]

    def step(self, action):
        board = self.state.board
        player_idx = self.state.player_idx
        assert self.done == False

        if self.action_mask[action] == 0:
            print(f"Player {player_idx}, action {action}, mask {self.action_mask}")
            self.render_ascii()
            raise Exception(f"Illegal move")

        self.move_counter += 1

        ri = ROWS - 1
        while board[ri, action, 0] + board[ri, action, 1] > 0 and ri >= 0:
            ri -= 1
        assert ri >= 0

        board = np.copy(board)
        board[ri, action, player_idx] = 1
        if ri == 0:
            self.action_mask[action] = 0

        if check_win(board, player_idx):
            self.done = True
            reward = 1
        elif np.sum(self.action_mask) == 0:
            self.done = True
            reward = 0
        else:
            reward = 0

        self.state = HashableState(board, 1 - player_idx)
        return reward, self.done

    def get_valid_actions_mask(self) -> np.ndarray:
        return self.action_mask

    def is_white_to_move(self) -> bool:
        return self.state.player_idx == 0

    def render_ascii(self):
        board = self.state.board
        player_idx = self.state.player_idx
        done = self.done

        print_tabs = [[] for _ in range(2)]

        for ri in range(board.shape[0]):
            print_tabs[0].append(''.join([ ('0' if board[ri, ci, 0] == 1 else '1' if board[ri, ci, 1] == 1 else '.') for ci in range(board.shape[1]) ]))

        if done:
            print_tabs[1].append('Game finished')
        else:
            print_tabs[1].append(f" Player {player_idx} to move" if ri == board.shape[0] - 1 else '')

        helpers.print_tabs_content(print_tabs)