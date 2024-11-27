#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>

#include "neural/network.h"
#include "infra/config_parser.h"


namespace probs {

class ModelKeeper {
    public:
        ModelKeeper(const ConfigParser& config_parser, const std::string& config_v_key, const std::string& config_q_key, const std::string& training_key);
        ResNet v_model;
        ResNet q_model;
        torch::optim::AdamW v_optimizer;
        torch::optim::AdamW q_optimizer;
        std::string checkpoints_dir;

        void SaveCheckpoint();
};

}   // namespace probs