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
#include "utils/exception.h"


namespace probs {

using QDataset = std::vector<std::pair<lczero::InputPlanes, std::vector<std::pair<lczero::Move, float>>>>;

QDataset GetQDataset(ResNet v_model, ResNet q_model, at::Device& device, const ConfigParser& config_parser, const int n_games);

}  // namespace probs
