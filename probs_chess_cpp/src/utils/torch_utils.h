#pragma once

#include <torch/torch.h>
#include <iostream>


namespace probs {

std::string DebugString(const torch::Tensor& tensor);

std::string DebugString(const torch::nn::Module& net);

}       // namespace probs
