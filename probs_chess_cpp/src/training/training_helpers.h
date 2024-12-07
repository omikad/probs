#pragma once

#include <vector>
#include <math.h>
#include <torch/torch.h>
#include <ATen/Device.h>

#include "chess/bitboard.h"
#include "utils/exception.h"
#include "infra/config_parser.h"


namespace probs {


struct MoveEstimation {
    lczero::Move move;
    float score;
};

lczero::Move GetMoveWithExploration(const std::vector<MoveEstimation>& moves_estimation, int env_ply, bool exploration_full_random, int exploration_num_first_moves);

at::Device GetDeviceFromConfig(const ConfigParser& config_parser);

}  // namespace probs
