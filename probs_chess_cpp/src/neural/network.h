#pragma once

#include <torch/torch.h>
#include <iostream>

#include "infra/config_parser.h"
#include "neural/encoder.h"

using namespace std;


namespace probs {

struct ResNet : torch::nn::Module {
    public:
        ResNet(const ConfigParser& config_parser, const string& config_key_prefix);
        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Conv2d m_conv_first{nullptr};
        torch::nn::Conv2d m_conv_last{nullptr};

        // torch::nn::BatchNorm2d m_bn1{nullptr};
        // torch::nn::ReLU m_relu{nullptr};
        // torch::nn::MaxPool2d m_maxpool{nullptr};
        // torch::nn::Sequential m_layer1{nullptr}, m_layer2{nullptr}, m_layer3{nullptr}, m_layer4{nullptr};
        // torch::nn::AdaptiveAvgPool2d m_avgpool{nullptr};
        // torch::nn::Linear m_fc{nullptr};

        torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride);
};

}  // namespace probs
