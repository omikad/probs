#pragma once
#include <vector>
#include <utility>
#include <ATen/Device.h>
#include <torch/torch.h>

#include "chess/position.h"
#include "infra/config_parser.h"
#include "infra/env_player.h"
#include "neural/encoder.h"
#include "neural/network.h"
#include "neural/torch_encoder.h"
#include "utils/torch_utils.h"


namespace probs {

std::vector<std::pair<torch::Tensor, float>> SelfPlay(ResNet q_model, const ConfigParser& config_parser, const int n_games);

void TrainV(const ConfigParser& config_parser, ResNet v_model, torch::optim::AdamW& v_optimizer, std::vector<std::pair<torch::Tensor, float>>& v_dataset);

}  // namespace probs
