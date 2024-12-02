#pragma once
#include <vector>
#include <memory>
#include <utility>
#include <ATen/Device.h>
#include <torch/torch.h>

#include "chess/position.h"
#include "chess/policy_map.h"
#include "infra/config_parser.h"
#include "infra/env_player.h"
#include "neural/encoder.h"
#include "neural/network.h"
#include "neural/torch_encoder.h"
#include "utils/torch_utils.h"
#include "utils/exception.h"
#include "training/training_helpers.h"


namespace probs {


struct QDatasetRow {
    lczero::InputPlanes input_planes;
    int transform;
    std::vector<MoveEstimation> target;
};

using QDataset = std::vector<QDatasetRow>;

QDataset GetQDataset(ResNet v_model, ResNet q_model, at::Device& device, const ConfigParser& config_parser, const int n_games);

void TrainQ(const ConfigParser& config_parser, ResNet q_model, at::Device& device, torch::optim::AdamW& q_optimizer, QDataset& q_dataset);

}  // namespace probs
