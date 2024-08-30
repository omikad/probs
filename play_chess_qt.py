# Adapted from https://github.com/h-nasir/PyQtChess

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFrame, QHBoxLayout, QVBoxLayout, QWidget, QGroupBox, QRadioButton, QPushButton, QSizePolicy

from play_chess_qt_board import ChessBoard

from environments.my_chess_env import create_env_66_func
import helpers
import probs_impl_common


class Info(QFrame):
    def __init__(self, parent):
        super().__init__(parent)

        self.board = parent.board

        # Select human side radio:
        group_select_human = QGroupBox("Human player:", self)
        self.radio_white = QRadioButton("White (first turn)")
        self.radio_black = QRadioButton("Black (second turn)")
        self.radio_white.setChecked(True)

        radio_select_human_layout = QVBoxLayout()
        radio_select_human_layout.setSpacing(5)
        radio_select_human_layout.addWidget(self.radio_white)
        radio_select_human_layout.addWidget(self.radio_black)

        group_select_human.setLayout(radio_select_human_layout)

        # Select AI enemy radio:
        group_select_ai = QGroupBox("AI player:", self)
        self.radio_ai_random = QRadioButton("Random")
        self.radio_ai_two = QRadioButton("Two step lookahead ")
        self.radio_ai_ai1 = QRadioButton("AI Model 1")
        self.radio_ai_ai2 = QRadioButton("AI Model 2")
        self.radio_ai_ai2.setChecked(True)

        radio_select_ai_layout = QVBoxLayout()
        radio_select_ai_layout.setSpacing(5)
        radio_select_ai_layout.addWidget(self.radio_ai_random)
        radio_select_ai_layout.addWidget(self.radio_ai_two)
        radio_select_ai_layout.addWidget(self.radio_ai_ai1)
        radio_select_ai_layout.addWidget(self.radio_ai_ai2)

        group_select_ai.setLayout(radio_select_ai_layout)

        # Select AI thinking params
        group_thinking_params = QGroupBox("AI params:", self)
        self.radio_one_shot = QRadioButton("One-shot")
        self.radio_beam_search_0_5 = QRadioButton("Beam search 0.5 sec")
        self.radio_beam_search_1 = QRadioButton("Beam search 1 sec")
        self.radio_beam_search_3 = QRadioButton("Beam search 3 sec")
        self.radio_beam_search_0_5.setChecked(True)

        radio_params_layout = QVBoxLayout()
        radio_params_layout.setSpacing(5)
        radio_params_layout.addWidget(self.radio_one_shot)
        radio_params_layout.addWidget(self.radio_beam_search_0_5)
        radio_params_layout.addWidget(self.radio_beam_search_1)
        radio_params_layout.addWidget(self.radio_beam_search_3)

        group_thinking_params.setLayout(radio_params_layout)

        # Button reset
        reset_button = QPushButton("Start new game")
        reset_button.clicked.connect(self.reset)

        # Main layout
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setStyleSheet('background-color: white')

        main_v_layout = QVBoxLayout()
        main_v_layout.setContentsMargins(10, 10, 10, 10)
        main_v_layout.setSpacing(10)

        main_v_layout.addWidget(group_select_human)
        main_v_layout.addWidget(group_select_ai)
        main_v_layout.addWidget(group_thinking_params)
        main_v_layout.addWidget(reset_button)
        self.setLayout(main_v_layout)

    def reset(self):
        self.board.user_is_white = self.radio_white.isChecked()
        if self.radio_ai_random.isChecked():
            self.board.agent = helpers.RandomAgent()
        elif self.radio_ai_two.isChecked():
            self.board.agent = helpers.TwoStepLookaheadAgent()
        else:
            model_str = None
            if self.radio_ai_ai1.isChecked():
                model_str = "V=ValueModel66_v11,SL=SelfLearningModel66_v11,CKPT=environments/mychess6x6_v11_checkpoint_20240822-132834.ckpt"
            elif self.radio_ai_ai2.isChecked():
                model_str = "V=ValueModel66_v11,SL=SelfLearningModel66_v11,CKPT=environments/mychess6x6_v11_checkpoint_20240828-033905.ckpt"

            if model_str is not None:
                model_keeper = probs_impl_common.create_model_keeper(None, "mychess6x6", train_params=None, model_str=model_str)
                model_keeper.eval()

                if self.radio_one_shot.isChecked():
                    self.board.agent = probs_impl_common.SelfLearningAgent("one shot agent", model_keeper=model_keeper, device='cpu')
                elif self.radio_beam_search_0_5.isChecked():
                    self.board.agent = probs_impl_common.SelfLearningAgent_TreeScan("tree search 0.1 sec", model_keeper=model_keeper, device='cpu')
                    self.board.agent.action_time_budget = 0.5 / 2 * 0.9   # divide by two because there are two submoves in each turn. Add a bit of time for eval
                    self.board.agent.expand_tree_budget = 500
                elif self.radio_beam_search_1.isChecked():
                    self.board.agent = probs_impl_common.SelfLearningAgent_TreeScan("tree search 1 sec", model_keeper=model_keeper, device='cpu')
                    self.board.agent.action_time_budget = 1 / 2 * 0.9
                    self.board.agent.expand_tree_budget = 100000
                elif self.radio_beam_search_3.isChecked():
                    self.board.agent = probs_impl_common.SelfLearningAgent_TreeScan("tree search 3 sec", model_keeper=model_keeper, device='cpu')
                    self.board.agent.action_time_budget = 3 / 2 * 0.9
                    self.board.agent.expand_tree_budget = 100000

        self.board.reset()
        self.board.start_game()


class GameFrame(QFrame):
    def __init__(self, parent):
        super().__init__()

        self.parent = parent

        env = create_env_66_func()

        self.setStyleSheet('background-color: #4B4945')

        self.user_is_white = True

        self.board = ChessBoard(self, env)
        self.info = Info(self)

        game_widget = QWidget()
        game_layout = QHBoxLayout()
        game_layout.setContentsMargins(0, 0, 0, 0)
        game_layout.setSpacing(30)
        game_layout.addWidget(self.board, 8)
        game_layout.addWidget(self.info, 3)
        game_widget.setLayout(game_layout)

        vbox_widget = QWidget()
        vbox_layout = QVBoxLayout()
        vbox_layout.addWidget(game_widget, 16)
        vbox_widget.setLayout(vbox_layout)

        self.setLayout(vbox_layout)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # Instantiate frames for different pages
        self.game_frame = GameFrame(self)

        self.game_frame.board.user_is_white = True
        self.game_frame.board.reset()

        self.game_frame.info.reset()

        # Set window details
        self.setCentralWidget(self.game_frame)
        self.setWindowTitle("Chess 6x6")
        self.setWindowIcon(QIcon('./assets/icons/pawn_icon.png'))
        self.setMinimumSize(1200, 900)
        self.show()


