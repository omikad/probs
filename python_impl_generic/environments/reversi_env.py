import numpy as np
from numba import jit
from typing import Tuple

import helpers


def create_env_func():
    return Reversi()


ROWS = 8
COLS = 8
PASS_ACTION = ROWS * COLS
DIRS = np.array([
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1],
], dtype=np.int32)

@jit(nopython=True)
def _compute_action_mask_and_reward_done(board: np.ndarray, player_idx: int, action_mask: np.ndarray) -> Tuple[int, bool]:
    fig = 2 * player_idx - 1

    action_mask.fill(0)

    any_move = False

    fig_where = np.where(board == fig)
    for r, c in zip(*fig_where):
        for dr, dc in DIRS:
            found_enemy = False
            nr = r + dr
            nc = c + dc
            while 0 <= nr < ROWS and 0 <= nc < COLS:
                if board[nr, nc] == -fig:
                    found_enemy = True
                    nr += dr
                    nc += dc
                else:
                    if found_enemy and board[nr, nc] == 0:
                        action_mask[nr * COLS + nc] = 1
                        any_move = True
                    break

    if any_move:
        return 0, False

    else:
        cnt_empty = np.sum(board == 0)
        if cnt_empty == 0:
            return np.sign(np.sum(board == fig) - PASS_ACTION // 2), True

        else:
            action_mask[-1] = 1  # Pass action
            return 0, False

@jit(nopython=True)
def _step(board: np.ndarray, player_idx: int, action_mask: np.ndarray, action: int) -> Tuple[int, bool]:
    fig = 2 * player_idx - 1

    if action < PASS_ACTION:   # Not a pass action
        r = action // COLS
        c = action % COLS

        board[r, c] = fig

        for dr, dc in DIRS:
            nr = r + dr
            nc = c + dc
            found_self = False
            while 0 <= nr < ROWS and 0 <= nc < COLS:
                if board[nr, nc] == 0:
                    break
                if board[nr, nc] == fig:
                    found_self = True
                    break
                nr += dr
                nc += dc
            if found_self:
                while nr != r or nc != c:
                    board[nr, nc] = fig
                    nr -= dr
                    nc -= dc

    reward, done = _compute_action_mask_and_reward_done(board, 1 - player_idx, action_mask)

    if action == PASS_ACTION and action_mask[-1] == 1 and not done:   # Both players can't move
        return np.sign(np.sum(board == fig) - PASS_ACTION // 2), True

    return reward, done


class Reversi(helpers.BaseEnv):
    board: np.ndarray  # -1 white, 1 black
    player_idx: int  # 0 white, 1 black
    done: bool
    action_mask: np.ndarray

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), np.int32)
        self.action_mask = np.zeros(ROWS * COLS + 1, np.int32)   # last action is pass
        self.reset()

    def reset(self):
        self.board.fill(0)
        self.board[3, 4] = 1
        self.board[4, 3] = 1
        self.board[3, 3] = -1
        self.board[4, 4] = -1
        self.player_idx = 0
        self.done = False
        _compute_action_mask_and_reward_done(self.board, 0, self.action_mask)

    def step(self, action):
        assert self.done == False
        assert self.action_mask[action] == 1
        assert action == PASS_ACTION or self.board[action // COLS, action % COLS] == 0

        reward, done = _step(self.board, self.player_idx, self.action_mask, action)
        self.player_idx = 1 - self.player_idx
        self.done = done
        return reward, done

    def copy(self):
        env = Reversi()
        env.board = self.board.copy()
        env.player_idx = self.player_idx
        env.done = self.done
        env.action_mask = self.action_mask.copy()
        return env

    def get_valid_actions_mask(self) -> np.ndarray:
        return self.action_mask

    def is_white_to_move(self) -> bool:
        return self.player_idx == 0

    def get_rotated_encoded_state(self):
        board = self.board
        action_mask = self.action_mask

        if self.player_idx == 1:
            board = np.copy(-board)   # make current player to have -1 as board values

        return [board, action_mask[:-1], action_mask[-1]]

    def render_ascii(self):
        board = self.board
        action_mask = self.action_mask
        done = self.done
        player_idx = self.player_idx

        white_name = "\u2591\u2591"
        black_name = "\u2588\u2588"

        player_name = [white_name, black_name][player_idx]

        player0_cnt = np.sum(board == -1)
        player1_cnt = np.sum(board == 1)

        for r in range(ROWS):
            syms = []
            for c in range(COLS):
                if action_mask[r * COLS + c] == 1:
                    syms.append(f" {r * COLS + c:>2}")
                else:
                    syms.append([" " + white_name, "  .", " " + black_name][board[r, c] + 1])

            if r == 0:
                playerline = f" Player {white_name} score {player0_cnt}"
            elif r == 1:
                playerline = f" Player {black_name} score {player1_cnt}"
            elif r == ROWS - 3 and action_mask[-1] == 1:
                playerline = f" Player {player_name} must pass, action = {PASS_ACTION}"
            elif r == ROWS - 1:
                playerline = " Game finished" if done else f" Player {player_name} to move"
            else:
                playerline = ""

            print("".join(syms) + playerline)