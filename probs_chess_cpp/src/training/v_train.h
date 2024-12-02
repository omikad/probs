#pragma once
#include <vector>
#include <memory>
#include <utility>
#include <ATen/Device.h>
#include <torch/torch.h>

#include "chess/position.h"
#include "chess/game_tree.h"
#include "infra/config_parser.h"
#include "neural/encoder.h"
#include "neural/network.h"
#include "neural/torch_encoder.h"
#include "utils/torch_utils.h"
#include "training/training_helpers.h"


namespace probs {

using VDataset = std::vector<std::pair<lczero::InputPlanes, float>>;

VDataset SelfPlay(ResNet q_model, at::Device& device, const ConfigParser& config_parser, const int n_games);

void TrainV(const ConfigParser& config_parser, ResNet v_model, at::Device& device, torch::optim::AdamW& v_optimizer, VDataset& v_dataset);

}  // namespace probs
