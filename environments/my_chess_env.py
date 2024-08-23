import numpy as np
from numba import jit

import helpers


SIDE_WHITE = 0
SIDE_BLACK = 1

PIECE_IDX_KING = 0
PIECE_IDX_QUEEN = 1
PIECE_IDX_ROOK = 2
PIECE_IDX_BISHOP = 3
PIECE_IDX_KNIGHT = 4
PIECE_IDX_PAWN = 5
PIECE_BOUNDARY = 12

PIECE_TO_SYM = [
    '\u2654',     # WHITE CHESS KING
    '\u2655',     # WHITE CHESS QUEEN
    '\u2656',     # WHITE CHESS ROOK
    '\u2657',     # WHITE CHESS BISHOP
    '\u2658',     # WHITE CHESS KNIGHT
    '\u2659',     # WHITE CHESS PAWN
    '\u265A',     # BLACK CHESS KING
    '\u265B',     # BLACK CHESS QUEEN
    '\u265C',     # BLACK CHESS ROOK
    '\u265D',     # BLACK CHESS BISHOP
    '\u265E',     # BLACK CHESS KNIGHT
    '\u265F',     # BLACK CHESS PAWN
]

PATHS_KING = np.array(
    [
        [[dr, dc]] for dr in range(-1, 2) for dc in range(-1, 2) if dr != 0 or dc != 0
    ], dtype=np.int32)
PATHS_QUEEN = np.array(
    [
        [[dr * i, dc * i] for i in range(1, 10)] for dr in range(-1, 2) for dc in range(-1, 2) if dr != 0 or dc != 0
    ], dtype=np.int32)
PATHS_ROOK = np.array(
    [
        [[dr * i, dc * i] for i in range(1, 10)] for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ], dtype=np.int32)
PATHS_BISHOP = np.array(
    [
        [[dr * i, dc * i] for i in range(1, 10)] for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    ], dtype=np.int32)
PATHS_KNIGHT = np.array(
    [
        [[-2, -1]],
        [[-2, 1]],
        [[2, -1]],
        [[2, 1]],
        [[-1, -2]],
        [[-1, 2]],
        [[1, -2]],
        [[1, 2]],
    ], dtype=np.int32)
PATHS_WHITE_PAWN = np.array(
    [
        [[-1, 0], [-2, 0]],
        [[-1, -1], [-1, -1]],
        [[-1, 1], [-1, 1]],
    ], dtype=np.int32)
PATHS_BLACK_PAWN = np.array(
    [
        [[1, 0], [2, 0]],
        [[1, -1], [1, -1]],
        [[1, 1], [1, 1]],
    ], dtype=np.int32)


N_CHANNELS = 16
SYMMETRY_INDEXES_SWAP_TO = list(range(N_CHANNELS))
SYMMETRY_INDEXES_SWAP_FROM = [6,7,8,9,10,11, 0,1,2,3,4,5, 12,13,14,15]

SUB_MOVES = [np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)]


def create_env_45_func():
    return ChessEnv(45)

def create_env_88_func():
    return ChessEnv(88)

def create_env_86_func():
    return ChessEnv(86)

def create_env_66_func():
    return ChessEnv(66)


@jit(nopython=True)
def _yield_moves(board: np.ndarray, color: int, figure: int, row: int, col: int):
    _, rows, cols = board.shape

    if figure == PIECE_IDX_KING:
        paths = PATHS_KING
    elif figure == PIECE_IDX_QUEEN:
        paths = PATHS_QUEEN
    elif figure == PIECE_IDX_ROOK:
        paths = PATHS_ROOK
    elif figure == PIECE_IDX_BISHOP:
        paths = PATHS_BISHOP
    elif figure == PIECE_IDX_KNIGHT:
        paths = PATHS_KNIGHT
    elif figure == PIECE_IDX_PAWN and color == 0:
        paths = PATHS_WHITE_PAWN
    elif figure == PIECE_IDX_PAWN and color == 1:
        paths = PATHS_BLACK_PAWN
    else:
        raise Exception(f"Incorrect figure {figure}")

    result = []

    for path in paths:
        for dr, dc in path:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                other_channel = np.argmax(board[:, nr, nc])

                if board[other_channel, nr, nc] == 0:
                    other_channel = -1
                elif other_channel // 6 == color:
                    break
                if figure == PIECE_IDX_PAWN:
                    if dc == 0 and other_channel >= 0:
                        break

                result.append((other_channel, nr, nc))

                if other_channel >= 0:
                    break
                if figure == PIECE_IDX_PAWN:
                    if color == 0 and row != rows - 2:
                        break
                    if color == 1 and row != 1:
                        break

    return result


@jit(nopython=True)
def _is_king_under_attack(board: np.ndarray, color: int, king_row: int, king_col: int):
    for attacker_channel, ar, ac in _yield_moves(board, color, PIECE_IDX_QUEEN, king_row, king_col):
        if attacker_channel >= 0:
            attacker_fig = attacker_channel % 6
            if (
                    attacker_fig == PIECE_IDX_QUEEN or
                    (attacker_fig == PIECE_IDX_ROOK and (ar == king_row or ac == king_col)) or
                    (attacker_fig == PIECE_IDX_BISHOP and (ar != king_row and ac != king_col)) or
                    (attacker_fig == PIECE_IDX_PAWN and color == 0 and ar + 1 == king_row and abs(ac - king_col) == 1) or
                    (attacker_fig == PIECE_IDX_PAWN and color == 1 and ar - 1 == king_row and abs(ac - king_col) == 1) or
                    (attacker_fig == PIECE_IDX_KING and max(abs(ar - king_row), abs(ac - king_col)) == 1)
                ):
                return True
    for attacker_channel, ar, ac in _yield_moves(board, color, PIECE_IDX_KNIGHT, king_row, king_col):
        if attacker_channel >= 0:
            attacker_fig = attacker_channel % 6
            if attacker_fig == PIECE_IDX_KNIGHT:
                return True
    return False


@jit(nopython=True)
def _compute_action_mask(board: np.ndarray, player_idx: int, action_mask: np.ndarray, castling: np.ndarray, pawn_advance2_col: int) -> np.ndarray:
    _, rows, cols = board.shape

    action_mask.fill(0)

    king_row, king_col = np.where(board[player_idx * 6 + PIECE_IDX_KING, :, :] > 0)
    assert len(king_row) == 1
    king_row = king_row[0]
    king_col = king_col[0]

    moves = []

    for channel, row, col in zip(*np.where(board > 0)):
        color, figure = divmod(channel, 6)
        if color == player_idx:
            pick_i = row * cols + col
            for other_channel, nr, nc in _yield_moves(board, color, figure, row, col):

                kill_en_passant = False
                if figure == PIECE_IDX_PAWN and abs(col - nc) == 1:
                    if nc == pawn_advance2_col and ((color == 0 and row == 3) or (color == 1 and row == rows - 4)) and board[(1 - player_idx) * 6 + PIECE_IDX_PAWN, row, nc] == 1:
                        kill_en_passant = True
                    if not kill_en_passant and other_channel < 0:
                        continue

                board[channel, row, col] = 0
                board[channel, nr, nc] = 1
                if other_channel >= 0:
                    board[other_channel, nr, nc] = 0
                if kill_en_passant:
                    board[(1 - player_idx) * 6 + PIECE_IDX_PAWN, row, nc] = 0

                kr = king_row if figure != PIECE_IDX_KING else nr
                kc = king_col if figure != PIECE_IDX_KING else nc
                king_under_attack = _is_king_under_attack(board, color, kr, kc)

                if kill_en_passant:
                    board[(1 - player_idx) * 6 + PIECE_IDX_PAWN, row, nc] = 1
                if other_channel >= 0:
                    board[other_channel, nr, nc] = 1
                board[channel, nr, nc] = 0
                board[channel, row, col] = 1

                if king_under_attack:
                    continue

                action_mask[pick_i] = 1
                moves.append((pick_i, nr * cols + nc))

            if figure == PIECE_IDX_KING and (castling[color * 2] == 1 or castling[color * 2 + 1] == 1) and (not _is_king_under_attack(board, color, king_row, king_col)):
                # castling = [white queen side, white king side, black queen side, black king side]
                for side in range(2):
                    if castling[color * 2 + side] == 1:
                        dc = -1 + side * 2
                        if ((board[color * 6 + PIECE_IDX_ROOK, king_row, (0 if side == 0 else (cols - 1))] == 1)
                            and ((side == 0 and np.sum(board[:, king_row, 1:king_col]) == 0) or (side == 1 and np.sum(board[:, king_row, king_col+1:-1]) == 0))
                            and (not _is_king_under_attack(board, color, king_row, king_col + dc))
                            and (not _is_king_under_attack(board, color, king_row, king_col + dc + dc))):
                            moves.append((pick_i, king_row * cols + king_col + dc + dc))

    if len(moves) == 0:
        return np.empty((0, 2), dtype=np.int32)

    return np.array(moves, dtype=np.int32)


@jit(nopython=True)
def _step(board: np.ndarray, player_idx: int, action_mask: np.ndarray, picked_figure_action: int, action: int, castling: np.ndarray) -> tuple[int, bool, np.ndarray]:
    _, rows, cols = board.shape
    sr, sc = divmod(picked_figure_action, cols)
    tr, tc = divmod(action, cols)
    color, figure = divmod(np.argmax(board[:, sr, sc]), 6)
    assert color == player_idx
    assert board[color * 6 + figure, sr, sc] == 1

    # pawn promotion
    fig_channel_start = color * 6 + figure
    fig_channel_to = fig_channel_start
    if figure == PIECE_IDX_PAWN and ((color == 0 and tr == 0) or (color == 1 and tr == rows - 1)):
        fig_channel_to = color * 6 + PIECE_IDX_QUEEN

    if figure == PIECE_IDX_KING and abs(sc - tc) == 2:   # castling
        if tc < sc:   # queen side
            board[:, tr, 0] = 0
            board[color * 6 + PIECE_IDX_ROOK, tr, tc + 1] = 1
        else:
            board[:, tr, -1] = 0
            board[color * 6 + PIECE_IDX_ROOK, tr, tc - 1] = 1

    elif figure == PIECE_IDX_PAWN and abs(sc - tc) == 1 and np.sum(board[:, tr, tc]) == 0:   # en passant
        board[:, sr, tc] = 0

    # disable castling
    if figure == PIECE_IDX_KING:
        castling[color * 2] = 0
        castling[color * 2 + 1] = 0
    if figure == PIECE_IDX_ROOK and sr == ((rows - 1) if color == 0 else 0):
        if sc == 0:
            castling[color * 2] = 0
        elif sc == cols - 1:
            castling[color * 2 + 1] = 0
    if board[(1 - color) * 6 + PIECE_IDX_ROOK, tr, tc] == 1 and ((color == 0 and tr == 0) or (color == 1 and tr == rows - 1)):  # killed rook on enemy first line
        if tc == 0:
            castling[(1 - color) * 2] = 0
        elif tc == cols - 1:
            castling[(1 - color) * 2 + 1] = 0

    # prepare en passant
    pawn_advance2_col = -1
    if figure == PIECE_IDX_PAWN and abs(tr - sr) == 2:
        pawn_advance2_col = tc

    board[fig_channel_start, sr, sc] = 0
    board[:, tr, tc] = 0
    board[fig_channel_to, tr, tc] = 1

    moves = _compute_action_mask(board, 1 - player_idx, action_mask, castling, pawn_advance2_col)

    # channel_counts = np.sum(board, axis=(1,2))  # not supported by numba
    channel_counts = np.sum(board.reshape((len(board), -1)), axis=1)
    total_figures = np.sum(channel_counts)
    # Insufficient material
    if total_figures <= 4:
        white_figures = np.sum(channel_counts[:6])
        black_figures = np.sum(channel_counts[6:])
        is_white_kk = white_figures == 2 and channel_counts[PIECE_IDX_KNIGHT] == 1
        is_white_kb = white_figures == 2 and channel_counts[PIECE_IDX_BISHOP] == 1
        is_black_kk = black_figures == 2 and channel_counts[6 + PIECE_IDX_KNIGHT] == 1
        is_black_kb = black_figures == 2 and channel_counts[6 + PIECE_IDX_BISHOP] == 1
        if (white_figures == 1 or is_white_kk or is_white_kb) and (black_figures == 1 or is_black_kk or is_black_kb):
            return 0, True, moves

    if len(moves) == 0:
        king_row, king_col = np.where(board[(1 - color) * 6 + PIECE_IDX_KING, :, :] > 0)
        assert len(king_row) == 1
        king_row = king_row[0]
        king_col = king_col[0]
        if _is_king_under_attack(board, 1 - color, king_row, king_col):
            return 1, True, moves
        return 0, True, moves

    return 0, False, moves


# https://greenchess.net/variants.php?cat=2
def __create_init_boards():
    board45 = np.zeros((12, 5, 4), dtype=np.int32)
    board45[SIDE_WHITE * 6 + PIECE_IDX_ROOK, -1, 0] = 1
    board45[SIDE_WHITE * 6 + PIECE_IDX_KNIGHT, -1, 1] = 1
    board45[SIDE_WHITE * 6 + PIECE_IDX_BISHOP, -1, 2] = 1
    board45[SIDE_WHITE * 6 + PIECE_IDX_KING, -1, 3] = 1
    board45[SIDE_BLACK * 6 + PIECE_IDX_ROOK, 0, 0] = 1
    board45[SIDE_BLACK * 6 + PIECE_IDX_KNIGHT, 0, 1] = 1
    board45[SIDE_BLACK * 6 + PIECE_IDX_BISHOP, 0, 2] = 1
    board45[SIDE_BLACK * 6 + PIECE_IDX_KING, 0, 3] = 1

    board88 = np.zeros((12, 8, 8), dtype=np.int32)
    board88[SIDE_WHITE * 6 + PIECE_IDX_ROOK, -1, 0] = 1
    board88[SIDE_WHITE * 6 + PIECE_IDX_KNIGHT, -1, 1] = 1
    board88[SIDE_WHITE * 6 + PIECE_IDX_BISHOP, -1, 2] = 1
    board88[SIDE_WHITE * 6 + PIECE_IDX_QUEEN, -1, 3] = 1
    board88[SIDE_WHITE * 6 + PIECE_IDX_KING, -1, 4] = 1
    board88[SIDE_WHITE * 6 + PIECE_IDX_BISHOP, -1, 5] = 1
    board88[SIDE_WHITE * 6 + PIECE_IDX_KNIGHT, -1, 6] = 1
    board88[SIDE_WHITE * 6 + PIECE_IDX_ROOK, -1, 7] = 1
    board88[SIDE_WHITE * 6 + PIECE_IDX_PAWN, -2, :] = 1
    board88[SIDE_BLACK * 6 + PIECE_IDX_ROOK, 0, 0] = 1
    board88[SIDE_BLACK * 6 + PIECE_IDX_KNIGHT, 0, 1] = 1
    board88[SIDE_BLACK * 6 + PIECE_IDX_BISHOP, 0, 2] = 1
    board88[SIDE_BLACK * 6 + PIECE_IDX_QUEEN, 0, 3] = 1
    board88[SIDE_BLACK * 6 + PIECE_IDX_KING, 0, 4] = 1
    board88[SIDE_BLACK * 6 + PIECE_IDX_BISHOP, 0, 5] = 1
    board88[SIDE_BLACK * 6 + PIECE_IDX_KNIGHT, 0, 6] = 1
    board88[SIDE_BLACK * 6 + PIECE_IDX_ROOK, 0, 7] = 1
    board88[SIDE_BLACK * 6 + PIECE_IDX_PAWN, 1, :] = 1

    board86 = np.zeros((12, 6, 8), dtype=np.int32)
    board86[SIDE_WHITE * 6 + PIECE_IDX_ROOK, -1, 0] = 1
    board86[SIDE_WHITE * 6 + PIECE_IDX_KNIGHT, -1, 1] = 1
    board86[SIDE_WHITE * 6 + PIECE_IDX_BISHOP, -1, 2] = 1
    board86[SIDE_WHITE * 6 + PIECE_IDX_QUEEN, -1, 3] = 1
    board86[SIDE_WHITE * 6 + PIECE_IDX_KING, -1, 4] = 1
    board86[SIDE_WHITE * 6 + PIECE_IDX_BISHOP, -1, 5] = 1
    board86[SIDE_WHITE * 6 + PIECE_IDX_KNIGHT, -1, 6] = 1
    board86[SIDE_WHITE * 6 + PIECE_IDX_ROOK, -1, 7] = 1
    board86[SIDE_WHITE * 6 + PIECE_IDX_PAWN, -2, :] = 1
    board86[SIDE_BLACK * 6 + PIECE_IDX_ROOK, 0, 0] = 1
    board86[SIDE_BLACK * 6 + PIECE_IDX_KNIGHT, 0, 1] = 1
    board86[SIDE_BLACK * 6 + PIECE_IDX_BISHOP, 0, 2] = 1
    board86[SIDE_BLACK * 6 + PIECE_IDX_QUEEN, 0, 3] = 1
    board86[SIDE_BLACK * 6 + PIECE_IDX_KING, 0, 4] = 1
    board86[SIDE_BLACK * 6 + PIECE_IDX_BISHOP, 0, 5] = 1
    board86[SIDE_BLACK * 6 + PIECE_IDX_KNIGHT, 0, 6] = 1
    board86[SIDE_BLACK * 6 + PIECE_IDX_ROOK, 0, 7] = 1
    board86[SIDE_BLACK * 6 + PIECE_IDX_PAWN, 1, :] = 1

    board66 = np.zeros((12, 6, 6), dtype=np.int32)
    board66[SIDE_WHITE * 6 + PIECE_IDX_ROOK, -1, 0] = 1
    board66[SIDE_WHITE * 6 + PIECE_IDX_KNIGHT, -1, 1] = 1
    board66[SIDE_WHITE * 6 + PIECE_IDX_QUEEN, -1, 2] = 1
    board66[SIDE_WHITE * 6 + PIECE_IDX_KING, -1, 3] = 1
    board66[SIDE_WHITE * 6 + PIECE_IDX_KNIGHT, -1, 4] = 1
    board66[SIDE_WHITE * 6 + PIECE_IDX_ROOK, -1, 5] = 1
    board66[SIDE_WHITE * 6 + PIECE_IDX_PAWN, -2, :] = 1
    board66[SIDE_BLACK * 6 + PIECE_IDX_ROOK, 0, 0] = 1
    board66[SIDE_BLACK * 6 + PIECE_IDX_KNIGHT, 0, 1] = 1
    board66[SIDE_BLACK * 6 + PIECE_IDX_QUEEN, 0, 2] = 1
    board66[SIDE_BLACK * 6 + PIECE_IDX_KING, 0, 3] = 1
    board66[SIDE_BLACK * 6 + PIECE_IDX_KNIGHT, 0, 4] = 1
    board66[SIDE_BLACK * 6 + PIECE_IDX_ROOK, 0, 5] = 1
    board66[SIDE_BLACK * 6 + PIECE_IDX_PAWN, 1, :] = 1

    result = {
        45: board45,
        88: board88,
        66: board66,
        86: board86,
    }

    for code in list(result.keys()):
        board = result[code]
        _, rows, cols = board.shape
        action_mask = np.zeros((rows * cols), dtype=np.int32)

        moves = _compute_action_mask(board, 0, action_mask, np.zeros(4, dtype=np.int32), -1)

        boundary = np.zeros((N_CHANNELS, rows + 2, cols + 2), dtype=np.float32)
        boundary[PIECE_BOUNDARY, 0, :] = 1
        boundary[PIECE_BOUNDARY, -1, :] = 1
        boundary[PIECE_BOUNDARY, :, 0] = 1
        boundary[PIECE_BOUNDARY, :, -1] = 1

        result[code] = (board, action_mask, moves, boundary)

    return result
INIT_BOARDS_ACTION_PLACES = __create_init_boards()


class ChessEnv(helpers.BaseEnv):
    variant: int
    board: np.ndarray
    player_idx: int
    move_counter: int
    sub_move: int   # 0 - pick from, 1 - pick to
    picked_figure_action: int
    done: bool
    action_mask: np.ndarray
    moves: np.ndarray
    castling: np.ndarray

    def __init__(self, variant: int, other=None):
        if other is None:
            self.variant = variant
            self.reset()
        else:
            self.variant = other.variant
            self.board = np.copy(other.board)
            self.player_idx = other.player_idx
            self.move_counter = other.move_counter
            self.sub_move = other.sub_move
            self.picked_figure_action = other.picked_figure_action
            self.done = other.done
            self.action_mask = np.copy(other.action_mask)
            self.moves = np.copy(other.moves)
            self.castling = np.copy(other.castling)

    def reset(self):
        init_board, init_action_mask, init_moves, boundary_example = INIT_BOARDS_ACTION_PLACES[self.variant]
        self.board = np.copy(init_board)
        self.player_idx = 0
        self.move_counter = 0
        self.sub_move = 0
        self.picked_figure_action = 0
        self.done = False
        self.action_mask = np.copy(init_action_mask)
        self.moves = np.copy(init_moves)
        self.castling = np.ones(4, dtype=np.int32)

    def copy(self):
        return ChessEnv(self.variant, other=self)

    def get_valid_actions_mask(self) -> np.ndarray:
        if self.player_idx == 1:
            _, rows, cols = self.board.shape
            return self.action_mask.reshape((rows, cols))[::-1, :].reshape(-1)
        return self.action_mask

    def is_white_to_move(self) -> bool:
        return self.player_idx == 0

    def step(self, action: int) -> tuple[int, bool]:
        if self.player_idx == 1:
            _, rows, cols = self.board.shape
            r, c = divmod(action, cols)
            action = (rows - 1 - r) * cols + c

        assert self.done == False
        if self.action_mask[action] == 0:
            print("self.player_idx", self.player_idx)
            print("self.move_counter", self.move_counter)
            print("self.sub_move", self.sub_move)
            print("self.picked_figure_action", self.picked_figure_action)
            print("self.done", self.done)
            print("self.action_mask", self.action_mask)
            print("self.moves", self.moves)
            print("self.castling", self.castling)
            print("action", action)
            self.render_ascii()
            raise Exception("Picked impossible action")

        if self.sub_move == 0:
            self.picked_figure_action = action
            self.action_mask.fill(0)
            for pick_i, put_i in self.moves:
                if pick_i == action:
                    self.action_mask[put_i] = 1
            self.moves = np.empty((0, 2), dtype=np.int32)
            self.sub_move = 1
            return 0, False

        reward, done, moves = _step(self.board, self.player_idx, self.action_mask, self.picked_figure_action, action, self.castling)
        self.player_idx = 1 - self.player_idx
        self.move_counter += 1
        self.sub_move = 0
        self.picked_figure_action = 0
        self.done = done
        self.moves = moves
        return reward, done

    def chess_move_to_pick_put(self, chess_move: str) -> tuple[int, int]:
        sc, sr, ec, er = chess_move[:4]
        sr, sc = 8 - int(sr), ord(sc) - ord('a')
        er, ec = 8 - int(er), ord(ec) - ord('a')

        if self.player_idx == 1:
            sr = 7 - sr
            er = 7 - er

        cols = self.board.shape[2]
        pick_action = sr * cols + sc
        put_action = er * cols + ec
        return pick_action, put_action

    def to_chess_move(self, pick_action: int, put_action: int) -> str:
        sr, sc = divmod(pick_action, 8)
        er, ec = divmod(put_action, 8)

        if self.player_idx == 1:
            sr = 7 - sr
            er = 7 - er

        promo = ''
        rows = self.board.shape[1]
        is_white = self.is_white_to_move()
        if (is_white and er == 0 and self.board[SIDE_WHITE * 6 + PIECE_IDX_PAWN, sr, sc] == 1) or \
            (not is_white and er == rows - 1 and self.board[SIDE_BLACK * 6 + PIECE_IDX_PAWN, sr, sc] == 1):
                promo = 'q'

        sr, sc = str(8 - sr), chr(ord('a') + sc)
        er, ec = str(8 - er), chr(ord('a') + ec)

        return f"{sc}{sr}{ec}{er}{promo}"

    def get_rotated_encoded_state(self):
        castling = self.castling
        board = self.board

        # encoded channels:
        # 0-5: 6 player pieces
        # 6-11: 6 enemy pieces
        # 12: board boundary
        # 13: castling flag (1 is on rook which is available for castling)
        # 14: picked figure flag
        # 15: move counter (submove adds 0.5)
        enc = np.copy(INIT_BOARDS_ACTION_PLACES[self.variant][3])  # copy boundary example
        enc[:12, 1:-1, 1:-1] = board

        # castling = [white queen side, white king side, black queen side, black king side]
        if castling[0] == 1:
            enc[13, -2, 1] = 1
        if castling[1] == 1:
            enc[13, -2, -2] = 1
        if castling[2] == 1:
            enc[13, 1, 1] = 1
        if castling[3] == 1:
            enc[13, 1, -2] = 1

        if self.sub_move == 1:
            row, col = divmod(self.picked_figure_action, board.shape[2])
            enc[14, 1 + row, 1 + col] = 1

        enc[15, :, :] = (self.move_counter - 100.0) / 100

        if self.player_idx == 1:
            enc[SYMMETRY_INDEXES_SWAP_TO, :, :] = enc[SYMMETRY_INDEXES_SWAP_FROM, ::-1, :]

        return [enc, SUB_MOVES[self.sub_move]]

    def get_rotated_encoded_states_with_symmetry__value_model(self):
        enc, sub = self.get_rotated_encoded_state()
        enc_sym = np.flip(enc, axis=2).copy()
        return [
            [enc, sub],
            [enc_sym, sub],
        ]

    def get_rotated_encoded_states_with_symmetry__q_value_model(self, action_values):
        "Should return [*encoded env, action_values, action mask]"
        assert action_values.shape == self.action_mask.shape
        _, rows, cols = self.board.shape

        enc, sub = self.get_rotated_encoded_state()
        action_mask = np.copy(self.get_valid_actions_mask())
        enc_sym = np.flip(enc, axis=2).copy()
        action_values_sym = np.flip(action_values.reshape((rows, cols)), axis=1).reshape(-1)
        action_mask_sym = np.flip(action_mask.reshape((rows, cols)), axis=1).reshape(-1)

        return [
            [enc, sub, action_values, action_mask],
            [enc_sym, sub, action_values_sym, action_mask_sym],
        ]

    def get_state_dict(self):
        board_argmax = (np.argmax(self.board, axis=0) + 1) * (np.sum(self.board, axis=0) > 0) - 1
        board_argmax = board_argmax.astype(int).tolist()
        return {
            'variant': self.variant,
            'board': board_argmax,
            'player_idx': self.player_idx,
            'move_counter': self.move_counter,
            'sub_move': self.sub_move,
            'picked_figure_action': self.picked_figure_action,
            'done': self.done,
            'action_mask': list(map(int, self.get_valid_actions_mask())),
            'moves': [[int(a), int(b)] for a, b in self.moves],
            'castling': list(map(int, self.castling)),
        }

    def load_from_state_dict(self, state):
        assert self.variant == state['variant']
        self.board.fill(0)
        for r in range(len(state['board'])):
            for c in range(len(state['board'][r])):
                if (val := state['board'][r][c]) >= 0:
                    self.board[val, r, c] = 1
        self.player_idx = state['player_idx']
        self.move_counter = state['move_counter']
        self.sub_move = state['sub_move']
        self.picked_figure_action = state['picked_figure_action']
        self.done = state['done']
        self.action_mask[:] = np.array(state['action_mask'])
        self.moves = state['moves']
        self.castling = np.array(state['castling'], dtype=np.int32)
        if self.player_idx == 1:
            _, rows, cols = self.board.shape
            self.action_mask = self.action_mask.reshape((rows, cols))[::-1, :].reshape(-1)

    def render_ascii(self):
        board = self.board
        done = self.done
        player_idx = self.player_idx
        sub_move = self.sub_move
        action_mask = self.action_mask

        _, rows, cols = board.shape
        print_tabs = [[] for _ in range(3)]

        print_tabs[0].append('Board (not rotated):')
        print_tabs[0].append('    ' + 'abcdefgh'[:board.shape[2]])
        for row in range(board.shape[1]):
            row_content = []
            for col in range(board.shape[2]):
                channel = np.argmax(board[:, row, col])
                sym = PIECE_TO_SYM[channel] if board[channel, row, col] > 0 else '.'
                row_content.append(sym)

            print_tabs[0].append(f"  {rows-row} " + ''.join(row_content))

        print_tabs[1].append('Actions (rotated):')
        who = ['White', 'Black'][player_idx]
        turn_description = ['Pick figure', 'Pick end position'][sub_move]
        print_tabs[1].append(f"Turn {self.move_counter}. {who} to move. {turn_description}:")

        if done:
            print_tabs[1].append("Game finished")
        elif sub_move == 0:
            for row in range(board.shape[1]):
                row_content = []
                for col in range(board.shape[2]):
                    action_i = row * cols + col
                    if action_mask[action_i] == 1:
                        sym = PIECE_TO_SYM[np.argmax(board[:, row, col])]
                        rotated_action_i = action_i if self.is_white_to_move() else ((rows - 1 - row) * cols + col)
                        row_content.append(f"{sym}:{rotated_action_i}")
                print_tabs[1].append(' '.join(row_content))
        else:
            for row in range(board.shape[1]):
                row_content = []
                for col in range(board.shape[2]):
                    action_i = row * cols + col
                    if action_mask[action_i] == 1:
                        rotated_action_i = action_i if self.is_white_to_move() else ((rows - 1 - row) * cols + col)
                        row_content.append(f"{rotated_action_i:>2}")
                    else:
                        row_content.append('..')
                print_tabs[1].append(' '.join(row_content))

        print_tabs[2].append('Debug:')
        print_tabs[2].append(f"variant = {self.variant}")
        print_tabs[2].append(f"player_idx = {self.player_idx}")
        print_tabs[2].append(f"move_counter = {self.move_counter}")
        print_tabs[2].append(f"sub_move = {self.sub_move}")
        print_tabs[2].append(f"picked_figure_action = {self.picked_figure_action}")
        print_tabs[2].append(f"done = {self.done}")
        print_tabs[2].append(f"moves = length ({len(self.moves)})")
        print_tabs[2].append(f"castling = {self.castling}")

        helpers.print_tabs_content(print_tabs)


def go_test_chess_encoding_correctness():
    np.random.seed(34)
    for test_i in range(1000):
        print("test_i", test_i)
        env = create_env_88_func()
        action = -1
        for step_i in range(200):
            enc, submoves = env.get_rotated_encoded_state()
            assert np.sum(enc[12, 0, :]) == 10
            assert np.sum(enc[12, -1, :]) == 10
            assert np.sum(enc[12, :, 0]) == 10
            assert np.sum(enc[12, :, -1]) == 10
            assert np.sum(enc[12, :, :]) == 36
            assert np.allclose(submoves, np.ones(1, dtype=np.float32) * (step_i % 2))

            board = np.zeros((12, 8, 8), dtype=np.int32)
            board[:] = enc[:12, 1:-1, 1:-1]

            is_enc_rotated = step_i % 4 >= 2
            assert is_enc_rotated == (env.player_idx == 1)

            if is_enc_rotated:
                board[:6, :, :], board[6:12, :, :] = board[6:12, ::-1, :].copy(), board[:6, ::-1, :].copy()

            errors = []

            expected_castling = np.zeros(4, dtype=np.int32)
            for rook_row, rook_col in zip(*np.where(enc[13, 1:-1, 1:-1] == 1)):
                if is_enc_rotated:
                    rook_row = 7 - rook_row
                if rook_row == 7 and rook_col == 0: expected_castling[0] = 1
                elif rook_row == 7 and rook_col == 7: expected_castling[1] = 1
                elif rook_row == 0 and rook_col == 0: expected_castling[2] = 1
                elif rook_row == 0 and rook_col == 7: expected_castling[3] = 1
                else:
                    errors.append(f"Incorrect rook position for castling (rook moved?)")
                # 2 - channel rook
                if board[(6 if rook_row == 0 else 0) + 2, rook_row, rook_col] != 1:
                    errors.append(f"Castling rook err, expected rook at ({(6 if rook_row == 0 else 0) + 2, rook_row, rook_col})")

            if step_i % 2 == 0:  # step is to pick figure
                assert np.sum(enc[14, :, :]) == 0
            else:
                assert np.sum(enc[14, :, :]) == 1
                picked_row, picked_col = divmod(action, 8)
                assert enc[14, 1 + picked_row, 1 + picked_col] == 1

            assert np.allclose(enc[15, :, :], (np.ones((10, 10), dtype=np.float32) * (step_i // 2) - 100) / 100)

            if not np.allclose(expected_castling, env.castling):
                errors.append("castling not equal")
            if not np.allclose(board, env.board):
                errors.append("board not equal")

            if len(errors) > 0:
                print("ERRORS:", ', '.join(errors))
                print("test_i, step_i, is_enc_rotated", test_i, step_i, is_enc_rotated)
                print("env.player_idx, env.sub_move", env.player_idx, env.sub_move)
                print("restored board:")
                print(np.argmax(board, axis=0))
                print("env board:")
                env.render_ascii()
                print(np.argmax(env.board, axis=0))
                print("expected castling", expected_castling)
                print("env.castling", env.castling)
                raise 1

            action = env.get_random_action()
            reward, done = env.step(action)
            if done:
                break
    print('Ok!')



def _get_chess_moves(chess_board):
    chess_moves = []
    for move in chess_board.legal_moves:
        move_str = str(move)

        # NOTE: changes to the chess rules:
        # no promotions to anything except queen
        if len(move_str) == 5:
            if move_str[-1] != 'q':
                continue
            move_str = move_str[:-1]
        chess_moves.append((move_str, move))
    chess_moves.sort()
    return chess_moves


def go_test_chess_correctness(n_games=10000, n_steps=200, focus_test=None):
    import chess

    rng = np.random.default_rng(seed=33)

    def __action_to_str(action):
        r, c = divmod(action, 8)
        r = 8 - r
        c = "abcdefgh"[c]
        return f"{c}{r}"

    for test_i in range(n_games):
        print(f"test_i={test_i}")
        env = create_env_88_func()
        chess_board = chess.Board()

        for step_i in range(n_steps):
            env_board_max_figures = np.max(np.sum(env.board, axis=0))
            assert env_board_max_figures == 1

            player_idx = env.player_idx
            assert player_idx == (0 if chess_board.turn == chess.WHITE else 1)

            env_moves = []
            for pick_action in env.get_valid_actions_iter():
                env_copy = env.copy()
                env_copy.step(pick_action)
                for put_action in env_copy.get_valid_actions_iter():
                    if player_idx == 1:   # env flipped vertically
                        fr, fc = divmod(pick_action, 8)
                        tr, tc = divmod(put_action, 8)
                        fr = 7 - fr
                        tr = 7 - tr
                        pick = fr * 8 + fc
                        put = tr * 8 + tc
                    else:
                        pick = pick_action
                        put = put_action

                    env_moves.append((__action_to_str(pick) + __action_to_str(put), pick_action, put_action))
            env_moves.sort()

            chess_moves = _get_chess_moves(chess_board)

            err = ''
            if len(env_moves) != len(chess_moves):
                err = "Lengths are not equal"
            elif any(em[0] != cm[0] for (em, cm) in zip(env_moves, chess_moves)):
                err = "Moves are different"
            elif len(env_moves) == 0:
                err = "Empty moves!"

            if len(err):
                print("ERROR", err, "test_i", test_i, "step_i", step_i)
                env.render_ascii()
                print(chess_board)
                print("myenv:", [t[0] for t in env_moves])
                print("chess:", [t[0] for t in chess_moves])
                print(chess_board.fen())
                raise 1

            picked_move_i = rng.integers(len(chess_moves))

            if focus_test is not None and focus_test == test_i:
                env.render_ascii()
                print(f"Picked move: {chess_moves[picked_move_i]}")

            reward, done = env.step(env_moves[picked_move_i][1])  # pick move
            assert reward == 0 and done == False
            reward, done = env.step(env_moves[picked_move_i][2])  # put move

            chess_board.push(chess_moves[picked_move_i][1])

            if done != chess_board.is_game_over():
                if done and np.sum(env.board) <= 4:   # our insufficient check is more powerful (and more correct)
                    break

                env.render_ascii()
                print("ERROR", err, "test_i", test_i, "step_i", step_i)
                print("player_idx, reward, done, chess_board.outcome()", player_idx, reward, done, chess_board.outcome())
                print("myenv:", [t[0] for t in env_moves])
                print("chess:", [t[0] for t in chess_moves])
                print(chess_board.fen())
                raise 1

            if done:
                if player_idx == 0 and reward == 1:
                    expected_winner = chess.WHITE
                elif player_idx == 0 and reward == -1:
                    expected_winner = chess.BLACK
                elif player_idx == 1 and reward == 1:
                    expected_winner = chess.BLACK
                elif player_idx == 1 and reward == -1:
                    expected_winner = chess.WHITE
                else:
                    expected_winner = None

                if expected_winner != chess_board.outcome().winner:
                    env.render_ascii()
                    print("ERROR", err, "test_i", test_i, "step_i", step_i)
                    print("Unmatched outcome:")
                    print("player_idx, reward, done, chess_board.outcome()", player_idx, reward, done, chess_board.outcome())
                    print("myenv:", [t[0] for t in env_moves])
                    print("chess:", [t[0] for t in chess_moves])
                    print(chess_board.fen())
                    raise 1
                break

    print("Ok!")