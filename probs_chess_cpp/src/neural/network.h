#pragma once

#include <ATen/Device.h>
#include <torch/torch.h>
#include <iostream>

#include "infra/config_parser.h"
#include "neural/encoder.h"
#include "utils/exception.h"
#include "utils/torch_utils.h"


namespace probs {

struct ResNetImpl : torch::nn::Module {
    public:
        ResNetImpl(const ConfigParser& config_parser, const std::string& config_key_prefix, const bool true_if_v_else_q);
        torch::Tensor forward(torch::Tensor x);
    private:
        bool true_if_v_else_q;
        torch::nn::Conv2d m_conv_first{nullptr};
        torch::nn::Sequential m_res_tower{nullptr};
        torch::nn::Conv2d m_conv_last{nullptr};
        torch::nn::Conv2d m_v_conv_last{nullptr};
        torch::nn::Linear m_v_fc{nullptr};
};
TORCH_MODULE(ResNet);

}  // namespace probs
