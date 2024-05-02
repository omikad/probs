import torch

import helpers


DEFAULT_TRAIN_PARAMS = {
    'value_lr': 1e-3,
    'value_batch_size': 256,

    'self_learning_lr': 1e-3,
    'self_learning_batch_size': 256,

    'dirichlet_alpha': 0.2,
    'exploration_fraction': 0.25,
}


def create_value_model(ARGS, value_model_class_name):
    return globals()[value_model_class_name]()


def create_self_learning_model(ARGS, self_learning_model_class_name):
    return globals()[self_learning_model_class_name]()


def forward_common_model1(board, action_mask, pass_action_flag, model_cnn):
    assert board.shape == (board.shape[0], 8, 8)
    assert action_mask.shape == (board.shape[0], 64)
    assert pass_action_flag.shape == (board.shape[0],)

    board = torch.nn.functional.one_hot(board.long() + 1, num_classes=3)  # (B, 8, 8, 3)

    action_mask = action_mask.view((-1, 8, 8, 1))

    cnn_inp = torch.concatenate([board, action_mask], dim=3)
    cnn_inp = torch.permute(cnn_inp, (0, 3, 1, 2))

    cnn_out = model_cnn(cnn_inp.float())

    pass_action_flag = pass_action_flag.float().view((-1, 1))

    common_inp = torch.concat([cnn_out, pass_action_flag], dim=1)

    return common_inp


class ValueModel1(helpers.BaseValueModel):
    def __init__(self) -> None:
        super().__init__()

        self.model_cnn = torch.nn.Sequential(   # input (B, 4, 8, 8)
            torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=1),   # -> (B, 64, 5, 5)
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),   # -> (B, 64, 3, 3)
            torch.nn.Flatten(),
        )

        self.model_state_value = torch.nn.Sequential(   # input (B, 8*8+65 = 129)
            torch.nn.LeakyReLU(),
            torch.nn.Linear(577, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, board, action_mask, pass_action_flag):
        common_inp = forward_common_model1(board, action_mask, pass_action_flag, self.model_cnn)  # (B, 577)

        state_value = self.model_state_value(common_inp)

        return state_value


class SelfLearningModel1(helpers.BaseSelfLearningModel):
    def __init__(self) -> None:
        super().__init__()

        self.model_cnn = torch.nn.Sequential(   # input (B, 4, 8, 8)
            torch.nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=1),   # -> (B, 64, 5, 5)
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),   # -> (B, 64, 3, 3)
            torch.nn.Flatten(),
        )

        self.model_action_values = torch.nn.Sequential(   # input (B, 64*3*3+1=577)
            torch.nn.LeakyReLU(),
            torch.nn.Linear(577, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 65),
        )

    def forward(self, board, action_mask, pass_action_flag):
        common_inp = forward_common_model1(board, action_mask, pass_action_flag, self.model_cnn)  # (B, 577)

        action_values = self.model_action_values(common_inp)

        return action_values
