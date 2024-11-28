#pragma once
#include <vector>
#include <memory>
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

std::vector<std::pair<lczero::InputPlanes, float>> SelfPlay(ResNet q_model, at::Device& device, const ConfigParser& config_parser, const int n_games);

void TrainV(const ConfigParser& config_parser, ResNet v_model, at::Device& device, torch::optim::AdamW& v_optimizer, std::vector<std::pair<lczero::InputPlanes, float>>& v_dataset);

}  // namespace probs
