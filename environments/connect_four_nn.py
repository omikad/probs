import torch

import helpers


DEFAULT_TRAIN_PARAMS = {
    'value_lr': 0.003,
    'value_batch_size': 128,

    'self_learning_lr': 0.003,
    'self_learning_batch_size': 128,

    'dirichlet_alpha': 0.5,
    'exploration_fraction': 0.25,
}


def create_value_model(ARGS, value_model_class_name):
    return globals()[value_model_class_name]()


def create_self_learning_model(ARGS, self_learning_model_class_name):
    return globals()[self_learning_model_class_name]()


class ValueModel1(helpers.BaseValueModel):
    def __init__(self) -> None:
        super().__init__()

        self.model_state_value = torch.nn.Sequential(   # input (B, 2, 7, 6)
            torch.nn.Conv2d(in_channels=2, out_channels=16, kernel_size=4, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            torch.nn.Flatten(),   # -> 64
            torch.nn.Linear(64, 42),
            torch.nn.ReLU(),
            torch.nn.Linear(42, 1),
        )

    def forward(self, board):
        board = torch.permute(board, (0, 3, 2, 1))
        state_value = self.model_state_value(board)

        # state_value.shape == (B, 1)
        return state_value


class SelfLearningModel1(helpers.BaseSelfLearningModel):
    def __init__(self) -> None:
        super().__init__()

        self.model_action_values = torch.nn.Sequential(   # input (B, 2, 7, 6)
            torch.nn.Conv2d(in_channels=2, out_channels=16, kernel_size=4, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            torch.nn.Flatten(),   # -> 64
            torch.nn.Linear(64, 42),
            torch.nn.ReLU(),
            torch.nn.Linear(42, 7),
        )

    def forward(self, board):
        assert board.shape == (board.shape[0], 6, 7, 2)

        board = torch.permute(board, (0, 3, 2, 1))
        action_values = self.model_action_values(board)

        # action_values.shape == (B, 7)
        return action_values


class ValueModel2(helpers.BaseValueModel):
    def __init__(self) -> None:
        super().__init__()

        self.model_state_value = torch.nn.Sequential(   # input (B, 2, 7, 6)
            torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1),  # -> (B, 64, 5, 4)
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # -> (B, 64, 3, 2)
            torch.nn.Flatten(),   # -> 64*3*2=384
            torch.nn.Linear(384, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, board):
        board = torch.permute(board, (0, 3, 2, 1))   # (B, 2, 7, 6)
        state_value = self.model_state_value(board)

        # state_value.shape == (B, 1)
        return state_value


class SelfLearningModel2(helpers.BaseSelfLearningModel):
    def __init__(self) -> None:
        super().__init__()

        self.model_action_values = torch.nn.Sequential(   # input (B, 2, 7, 6)
            torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1),  # -> (B, 64, 5, 4)
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # -> (B, 64, 3, 2)
            torch.nn.Flatten(),   # -> 64*3*2=384
            torch.nn.Linear(384, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 7),
        )

    def forward(self, board):
        assert board.shape == (board.shape[0], 6, 7, 2)

        board = torch.permute(board, (0, 3, 2, 1))   # (B, 2, 7, 6)
        action_values = self.model_action_values(board)

        # action_values.shape == (B, 7)
        return action_values
