import torch

import helpers


N_ACTIONS = 7


#[-------------------------------------------------------------------------------]
#[--------------------------------- Model CF 7x6 v1 -----------------------------]
#[-------------------------------------------------------------------------------]

WIDTH_v1 = 64

class ValueModelCF_v1(helpers.BaseValueModel):   # 409089 parameters
    def __init__(self) -> None:
        super().__init__()

        self.model_state_value = torch.nn.Sequential(   # (B, 2, 7, 6) -> (B, COMMON_v1)
            torch.nn.Conv2d(in_channels=2, out_channels=WIDTH_v1, kernel_size=4, stride=1),
            torch.nn.LeakyReLU(0.01),
            ResBlock(planes=WIDTH_v1),
            ResBlock(planes=WIDTH_v1),
            ResBlock(planes=WIDTH_v1),
            ResBlock(planes=WIDTH_v1),
            ResBlock(planes=WIDTH_v1),
            torch.nn.Conv2d(in_channels=WIDTH_v1, out_channels=WIDTH_v1, kernel_size=3, stride=1),
            torch.nn.Flatten(),
            torch.nn.Linear(WIDTH_v1 * 2, 1),
        )

    def forward_common(self, board):

        common = self.model_common(board)
        return common

    def forward(self, board):
        B = board.shape[0]
        assert board.shape == (B, 6, 7, 2)

        board = torch.permute(board, (0, 3, 2, 1))

        state_value = self.model_state_value(board)
        return state_value


class SelfLearningModelCF_v1(helpers.BaseSelfLearningModel):   # 409863 parameters
    def __init__(self) -> None:
        super().__init__()

        self.model_action_values = torch.nn.Sequential(   # (B, 2, 7, 6) -> (B, COMMON)
            torch.nn.Conv2d(in_channels=2, out_channels=WIDTH_v1, kernel_size=4, stride=1),
            torch.nn.LeakyReLU(0.01),
            ResBlock(planes=WIDTH_v1),
            ResBlock(planes=WIDTH_v1),
            ResBlock(planes=WIDTH_v1),
            ResBlock(planes=WIDTH_v1),
            ResBlock(planes=WIDTH_v1),
            torch.nn.Conv2d(in_channels=WIDTH_v1, out_channels=WIDTH_v1, kernel_size=3, stride=1),
            torch.nn.Flatten(),
            torch.nn.Linear(WIDTH_v1 * 2, N_ACTIONS),
        )

    def forward(self, board):
        B = board.shape[0]
        assert board.shape == (B, 6, 7, 2)

        board = torch.permute(board, (0, 3, 2, 1))
        action_values = self.model_action_values(board)

        return action_values


# Got from https://github.com/geochri/AlphaZero_Chess/blob/master/src/alpha_net.py
class ResBlock(torch.nn.Module):
    def __init__(self, planes, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = torch.nn.functional.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = torch.nn.functional.relu(out)
        return out
