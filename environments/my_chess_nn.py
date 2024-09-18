import torch

import helpers


#[-------------------------------------------------------------------------------]
#[--------------------------------- Model 66 v11 --------------------------------]
#[-------------------------------------------------------------------------------]

WIDTH_V11 = 80

class ValueModel66_v11(helpers.BaseValueModel):   # 2016721 parameters
    def __init__(self) -> None:
        super().__init__()

        self.model_common = torch.nn.Sequential(   # input (B, 16, 8, 8)
            FirstConvBlock(inplanes=16, outplanes=WIDTH_V11, input_rows_cols=8),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),

            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),

            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),

            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
        )

        self.model_state_value = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=WIDTH_V11, out_channels=32, kernel_size=4, stride=2),   # -> (B, 32, 2, 2)
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32*2*2, 1),
        )

    def forward_common(self, board, sub_move):
        assert board.shape == (board.shape[0], 16, 8, 8)
        assert sub_move.shape == (board.shape[0], 1)
        common = self.model_common(board)
        return common

    def forward(self, board, sub_move):
        common = self.forward_common(board, sub_move)
        state_value = self.model_state_value(common)
        return state_value


class SelfLearningModel66_v11(helpers.BaseSelfLearningModel):   # 1975762 parameters
    def __init__(self) -> None:
        super().__init__()

        self.model_state = torch.nn.Sequential(   # input (B, 16, 8, 8)
            FirstConvBlock(inplanes=16, outplanes=WIDTH_V11, input_rows_cols=8),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),

            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),

            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),

            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
            ResBlock(inplanes=WIDTH_V11, planes=WIDTH_V11),
        )

        self.model_action_values = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=WIDTH_V11, out_channels=2, kernel_size=1, stride=1),   # -> (B, 2, 6, 6)
        )

    def forward(self, board, sub_move):
        assert board.shape == (board.shape[0], 16, 8, 8)
        assert sub_move.shape == (board.shape[0], 1)

        inp = self.model_state(board)  # -> (B, 64, 6, 6)

        action_values = self.model_action_values(inp)

        pick, put = torch.split(action_values, (1, 1), dim=1)
        pick = torch.flatten(pick, start_dim=1)
        put = torch.flatten(put, start_dim=1)

        result = pick * (1 - sub_move) + put * sub_move

        return result


#[-------------------------------------------------------------------------------]
#[--------------------------------- Model 88 v1 ---------------------------------]
#[-------------------------------------------------------------------------------]

WIDTH_8_v1 = 128
HEIGHT_8_v1 = 20

class ValueModel88_v1(helpers.BaseValueModel):   # 5993153 parameters
    def __init__(self) -> None:
        super().__init__()

        self.model_common = torch.nn.Sequential(   # input (B, 16, 10, 10)
            FirstConvBlock(inplanes=16, outplanes=WIDTH_8_v1, input_rows_cols=10),
            *[ ResBlock(inplanes=WIDTH_8_v1, planes=WIDTH_8_v1) for _ in range(HEIGHT_8_v1) ]
        )

        self.model_state_value = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=WIDTH_8_v1, out_channels=WIDTH_8_v1 // 4, kernel_size=4, stride=2),   # -> (B, 32, 3, 3)
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(WIDTH_8_v1 // 4 * 3 * 3, 1),
        )

    def forward_common(self, board, sub_move):
        assert board.shape == (board.shape[0], 16, 10, 10)
        assert sub_move.shape == (board.shape[0], 1)
        common = self.model_common(board)
        return common

    def forward(self, board, sub_move):
        common = self.forward_common(board, sub_move)
        state_value = self.model_state_value(common)
        return state_value


class SelfLearningModel88_v1(helpers.BaseSelfLearningModel):   # 5927554 parameters
    def __init__(self) -> None:
        super().__init__()

        self.model_state = torch.nn.Sequential(   # input (B, 16, 10, 10)
            FirstConvBlock(inplanes=16, outplanes=WIDTH_8_v1, input_rows_cols=10),
            *[ ResBlock(inplanes=WIDTH_8_v1, planes=WIDTH_8_v1) for _ in range(HEIGHT_8_v1) ]
        )

        self.model_action_values = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=WIDTH_8_v1, out_channels=2, kernel_size=1, stride=1),   # -> (B, 2, 8, 8)
        )

    def forward(self, board, sub_move):
        assert board.shape == (board.shape[0], 16, 10, 10)
        assert sub_move.shape == (board.shape[0], 1)

        inp = self.model_state(board)

        action_values = self.model_action_values(inp)

        pick, put = torch.split(action_values, (1, 1), dim=1)
        pick = torch.flatten(pick, start_dim=1)
        put = torch.flatten(put, start_dim=1)

        result = pick * (1 - sub_move) + put * sub_move

        return result


class FirstConvBlock(torch.nn.Module):
    def __init__(self, inplanes, outplanes, input_rows_cols):
        super(FirstConvBlock, self).__init__()
        self.input_rows_cols = input_rows_cols
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = torch.nn.Conv2d(inplanes, outplanes, 3, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(outplanes)

    def forward(self, s):
        s = s.view(s.shape[0], self.inplanes, self.input_rows_cols, self.input_rows_cols)
        s = torch.nn.functional.relu(self.bn1(self.conv1(s)))
        return s


# Copied from https://github.com/geochri/AlphaZero_Chess/blob/master/src/alpha_net.py
class ResBlock(torch.nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
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
