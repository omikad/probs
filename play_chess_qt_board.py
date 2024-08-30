# Adapted from https://github.com/h-nasir/PyQtChess

import numpy as np

from PyQt5.QtCore import pyqtSignal, QRegExp, Qt, QThread
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QFrame, QGridLayout, QLabel, QMessageBox, QSizePolicy, QWidget

SQR_SIZE = 60


def san_to_cell_index(san: str):
    assert len(san) == 2
    col = 'abcdef'.index(san[0])
    row = '654321'.index(san[1])
    return row * 6 + col


def cell_index_to_rowcol(idx: int):
    return divmod(idx, 6)


def cell_index_to_san(idx: int):
    row, col = cell_index_to_rowcol(idx)
    return 'abcdef'[col] + '654321'[row]


class ChessBoard(QFrame):
    def __init__(self, parent, env):
        super().__init__(parent)

        self.env = env
        self.parent = parent

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setContentsMargins(0, 0, 0, 0)

        self.layout = QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.draw_squares()
        self.setLayout(self.layout)

        self.user_is_white = self.parent.user_is_white
        self.agent = None
        self.search_thread = SearchThread(self)

        self.sqr_size = SQR_SIZE

    def resizeEvent(self, event):
        if event.size().width() > event.size().height():
            self.resize(event.size().height(), event.size().height())
            self.sqr_size = int(event.size().height() / 6)
        else:
            self.resize(event.size().width(), event.size().width())
            self.sqr_size = int(event.size().width() / 6)

    def start_game(self):
        if self.env.is_white_to_move() != self.user_is_white:
            self.disable_pieces()
            self.search_thread.start()
        else:
            self.enable_pieces()

    def draw_squares(self):
        for row, rank in enumerate('654321'):
            for col, file in enumerate('abcdef'):
                square = QWidget(self)
                square.setObjectName(file + rank)
                square.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                if row % 2 == col % 2:
                    square.setStyleSheet('background-color: #F0D9B5')
                else:
                    square.setStyleSheet('background-color: #B58863')
                self.layout.addWidget(square, row, col)

    def clear(self):
        all_pieces = self.findChildren(QLabel)
        for piece in all_pieces:
            piece.setParent(None)  # Delete piece

    def refresh_from_state(self):
        QApplication.processEvents()

        self.clear()

        board = self.env.board
        for row in range(board.shape[1]):
            for col in range(board.shape[2]):
                piece = np.argmax(board[:, row, col])
                if board[piece, row, col] > 0:
                    piece_label = PieceLabel(self, piece)
                    self.layout.addWidget(piece_label, row, col)

    def reset(self):
        self.env.reset()
        self.refresh_from_state()

    def highlight(self, idx: int):
        square = self.findChild(QWidget, cell_index_to_san(idx))

        row, col = cell_index_to_rowcol(idx)

        if row % 2 == col % 2:
            square.setStyleSheet('background-color: #F7EC74')
        else:
            square.setStyleSheet('background-color: #DAC34B')

    def unhighlight(self, idx: int):
        square = self.findChild(QWidget, cell_index_to_san(idx))

        row, col = cell_index_to_rowcol(idx)

        if row % 2 == col % 2:  # light square
            square.setStyleSheet('background-color: #F0D9B5')
        else:  # dark square
            square.setStyleSheet('background-color: #B58863')

    def unhighlight_all(self):
        for sqr_index in range(36):
            self.unhighlight(sqr_index)

    def disable_pieces(self):
        for piece in self.findChildren(QLabel):
            piece.is_enabled = False

    def enable_pieces(self):
        for piece in self.findChildren(QLabel):
            piece.is_enabled = True

    def maybe_flip(self, idx: int):
        if not self.env.is_white_to_move():
            row, col = divmod(idx, 6)
            row = 6 - 1 - row
            idx = row * 6 + col
        return idx

    def player_move(self, src_index, dst_index):
        self.disable_pieces()

        is_white = self.env.is_white_to_move()

        reward, done = self.env.step(self.maybe_flip(src_index))
        assert done == False
        reward, done = self.env.step(self.maybe_flip(dst_index))

        print(f"Player picked move {src_index, dst_index}. Reward, done = {reward, done}")

        self.refresh_from_state()

        if done:
            self.game_over(is_white, reward)
        else:
            self.search_thread.start()  # Start search thread for computer's move

    def computer_move(self, args):
        pick_action, put_action, is_white, reward, done = args

        self.refresh_from_state()

        if done:
            self.game_over(is_white, reward)
        else:
            self.enable_pieces()

    def game_over(self, is_white, reward):
        if reward == 0:
            text = "Draw!"
        else:
            text = f"{'White' if (is_white == (reward == 1)) else 'Black'} wins!"

        msg_box = QMessageBox()
        msg_box.resize(400, 200)
        font = QFont("Arial", 30)
        msg_box.setFont(font)
        msg_box.setWindowIcon(QIcon('./assets/icons/pawn_icon.png'))
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Chess")
        msg_box.setText(text)

        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


# Search algorithm must be run in a separate thread to the main event loop, to prevent the GUI from freezing
class SearchThread(QThread):
    move_signal = pyqtSignal(tuple)

    def __init__(self, board):
        super().__init__()

        self.board = board
        self.move_signal.connect(self.board.computer_move)

    def run(self):
        env = self.board.env
        agent = self.board.agent

        self.board.disable_pieces()

        is_white = env.is_white_to_move()

        pick_action = agent.get_action(env)

        if getattr(agent, 'last_search_nodes_cnt', None) is not None:
            search_nodes_cnt = agent.last_search_nodes_cnt
            search_time = agent.last_search_time
        else:
            search_nodes_cnt = None
            search_time = None

        reward, done = env.step(pick_action)
        assert done == False

        put_action = agent.get_action(env)

        reward, done = env.step(put_action)

        if search_nodes_cnt is not None:
            search_nodes_cnt += agent.last_search_nodes_cnt
            search_time += agent.last_search_time
            print(f"Agent {agent.get_name()} picked move {pick_action, put_action}, search nodes count {search_nodes_cnt}, search time {search_time} seconds. Reward, done = {reward, done}")
        else:
            print(f"Agent {agent.get_name()} picked move {pick_action, put_action}. Reward, done = {reward, done}")

        self.move_signal.emit((pick_action, put_action, is_white, reward, done))


class PieceLabel(QLabel):
    def __init__(self, parent, piece):
        super().__init__(parent)

        self.piece = piece

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(1, 1)

        # Make label transparent, so square behind piece is visible
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.board = parent
        self.is_white = self.piece < 6
        self.is_enabled = True

        self.src_pos = None
        self.mouse_pos = None
        self.src_square = None
        self.dst_square = None
        self.legal_moves = None
        self.legal_dst_squares = None

        # Store original piece image
        pixmap = QPixmap('./assets/pieces/{}{}.png'.format('w' if self.is_white else 'b', "kqrbnp"[self.piece % 6]))
        self.setPixmap(pixmap)

        # When label is scaled, also scale image inside the label
        self.setScaledContents(True)

        self.setMouseTracking(True)

        self.show()

    def resizeEvent(self, event):
        if event.size().width() > event.size().height():
            self.resize(event.size().height(), event.size().height())
        else:
            self.resize(event.size().width(), event.size().width())

    def enterEvent(self, event):
        if self.is_enabled:
            if self.board.user_is_white == self.is_white:
                # Set open hand cursor while hovering over a piece
                QApplication.setOverrideCursor(Qt.OpenHandCursor)

    def leaveEvent(self, event):
        # Set arrow cursor while not hovering over a piece
        QApplication.setOverrideCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if self.is_enabled:
            if event.buttons() == Qt.LeftButton:
                if self.board.user_is_white == self.is_white:

                    # Store mouse position and square position, relative to the chessboard
                    self.mouse_pos = self.mapToParent(self.mapFromGlobal(event.globalPos()))
                    self.src_pos = self.mapToParent(self.rect().topLeft())

                    # Identify origin square
                    all_squares = self.board.findChildren(QWidget, QRegExp(r'[a-h][1-8]'))
                    for square in all_squares:
                        if square.pos() == self.src_pos:
                            self.src_square = square
                            break

                    # Identify legal moves
                    src_index = san_to_cell_index(self.src_square.objectName())

                    # Set closed hand cursor while dragging a piece
                    QApplication.setOverrideCursor(Qt.ClosedHandCursor)

                    # Raise piece to the front
                    self.raise_()

                    # Snap to cursor
                    offset = self.rect().topLeft() - self.rect().center()
                    self.move(self.mouse_pos + offset)

                    orig_env = self.board.env
                    assert orig_env.sub_move == 0
                    if orig_env.get_valid_actions_mask()[self.board.maybe_flip(src_index)] == 1:
                        env_copy = orig_env.copy()
                        env_copy.step(self.board.maybe_flip(src_index))

                        # Highlight origin and destination squares
                        self.board.highlight(src_index)
                        for dst_square in env_copy.get_valid_actions_iter():
                            self.board.highlight(dst_square)

    def mouseMoveEvent(self, event):
        if self.is_enabled:
            if event.buttons() == Qt.LeftButton:
                if self.board.user_is_white == self.is_white:
                    # Update mouse position, relative to the chess board
                    self.mouse_pos = self.mapToParent(self.mapFromGlobal(event.globalPos()))

                    # Calculate offset from centre to top-left of square
                    offset = self.rect().topLeft() - self.rect().center()

                    # Calculate new x position, not allowing the piece to go outside the board
                    if self.mouse_pos.x() < self.board.rect().left():
                        new_pos_x = self.board.rect().left() + offset.x()
                    elif self.mouse_pos.x() > self.board.rect().right():
                        new_pos_x = self.board.rect().right() + offset.x()
                    else:
                        new_pos_x = self.mouse_pos.x() + offset.x()

                    # Calculate new y position, not allowing the piece to go outside the board
                    if self.mouse_pos.y() < self.board.rect().top():
                        new_pos_y = self.board.rect().top() + offset.y()
                    elif self.mouse_pos.y() > self.board.rect().bottom():
                        new_pos_y = self.board.rect().right() + offset.y()
                    else:
                        new_pos_y = self.mouse_pos.y() + offset.y()

                    # Move piece to new position
                    self.move(new_pos_x, new_pos_y)

    def mouseReleaseEvent(self, event):
        if self.is_enabled:
            if self.board.user_is_white == self.is_white:

                # Set open hand cursor when piece is released
                QApplication.setOverrideCursor(Qt.OpenHandCursor)

                self.board.unhighlight_all()

                # If mouse not released on board, move piece back to origin square, and return
                if not self.board.rect().contains(self.board.mapFromGlobal(event.globalPos())):
                    self.move(self.src_pos)
                    return

                # Identify destination square
                all_squares = self.board.findChildren(QWidget, QRegExp(r'[a-h][1-8]'))
                for square in all_squares:
                    if square.rect().contains(square.mapFromGlobal(event.globalPos())):
                        self.dst_square = square
                        break

                src_index = san_to_cell_index(self.src_square.objectName())
                dst_index = san_to_cell_index(self.dst_square.objectName())

                orig_env = self.board.env
                assert orig_env.sub_move == 0

                if orig_env.get_valid_actions_mask()[self.board.maybe_flip(src_index)] == 1:
                    env_copy = self.board.env.copy()
                    env_copy.step(self.board.maybe_flip(src_index))
                    if env_copy.get_valid_actions_mask()[self.board.maybe_flip(dst_index)] == 1:
                        self.board.player_move(src_index, dst_index)
                        return

                # Illegal move - snap back to origin square
                self.move(self.src_pos)
