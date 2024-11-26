#pragma once

#include <ATen/Device.h>
#include <torch/torch.h>
#include <iostream>

#include "infra/config_parser.h"
#include "neural/encoder.h"
#include "utils/exception.h"


namespace probs {

struct ResNet : torch::nn::Module {
    public:
        ResNet(const ConfigParser& config_parser, const std::string& config_key_prefix);
        torch::Tensor forward(torch::Tensor x);
    private:
        torch::nn::Conv2d m_conv_first{nullptr};
        torch::nn::Sequential res_tower{nullptr};
        torch::nn::Conv2d m_conv_last{nullptr};
};

}  // namespace probs
