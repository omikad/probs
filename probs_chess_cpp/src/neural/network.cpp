/*
    Based on: https://github.com/leimao/LibTorch-ResNet-CIFAR/blob/main/src/resnet.cpp
*/

#include "neural/network.h"

using namespace std;


namespace probs {

torch::nn::Conv2dOptions
create_conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                    int64_t stride = 1, int64_t padding = 0, int64_t groups = 1,
                    int64_t dilation = 1, bool bias = false) {
    torch::nn::Conv2dOptions conv_options =
        torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size)
            .stride(stride)
            .padding(padding)
            .bias(bias)
            .groups(groups)
            .dilation(dilation);

    return conv_options;
}


torch::nn::Conv2dOptions create_conv3x3_options(int64_t in_planes,
                                                int64_t out_planes,
                                                int64_t stride = 1,
                                                int64_t groups = 1,
                                                int64_t dilation = 1) {
    torch::nn::Conv2dOptions conv_options = create_conv_options(
        in_planes, out_planes, /*kerner_size = */ 3, stride,
        /*padding = */ dilation, groups, /*dilation = */ dilation, false);
    return conv_options;
}


torch::nn::Conv2dOptions create_conv1x1_options(int64_t in_planes,
                                                int64_t out_planes,
                                                int64_t stride = 1) {
    torch::nn::Conv2dOptions conv_options = create_conv_options(
        in_planes, out_planes,
        /*kerner_size = */ 1, stride,
        /*padding = */ 0, /*groups = */ 1, /*dilation = */ 1, false);
    return conv_options;
}


struct ResNetBlock : torch::nn::Module {
    ResNetBlock(int64_t planes, int64_t stride = 1) {
        m_conv1 = register_module("conv1", torch::nn::Conv2d{create_conv3x3_options(planes, planes, stride)});
        m_bn1 = register_module("bn1", torch::nn::BatchNorm2d{planes});
        m_relu = register_module("relu", torch::nn::ReLU{true});
        m_conv2 = register_module("conv2", torch::nn::Conv2d{create_conv3x3_options(planes, planes)});
        m_bn2 = register_module("bn2", torch::nn::BatchNorm2d{planes});
        m_stride = stride;
    }

    static const int64_t m_expansion = 1;

    torch::nn::Conv2d m_conv1{nullptr}, m_conv2{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr}, m_bn2{nullptr};
    torch::nn::ReLU m_relu{nullptr};

    int64_t m_stride;

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor identity = x;

        torch::Tensor out = m_conv1->forward(x);
        out = m_bn1->forward(out);
        out = m_relu->forward(out);

        out = m_conv2->forward(out);
        out = m_bn2->forward(out);

        out += identity;
        out = m_relu->forward(out);

        return out;
    }
};


ResNetImpl::ResNetImpl(const ConfigParser& config_parser, const string& config_key_prefix, const bool true_if_v_else_q) : true_if_v_else_q(true_if_v_else_q) {
    int res_blocks = config_parser.GetInt(config_key_prefix + ".res_blocks");
    int filters = config_parser.GetInt(config_key_prefix + ".filters");
    cout << "[NETWORK] ResNet " << (true_if_v_else_q ? "V" : "Q") << " config_key_prefix: " << config_key_prefix << endl;
    cout << "[NETWORK] ResNet " << (true_if_v_else_q ? "V" : "Q") << " res_blocks: " << res_blocks << endl;
    cout << "[NETWORK] ResNet " << (true_if_v_else_q ? "V" : "Q") << " filters: " << filters << endl;

    m_conv_first = register_module(
        "m_conv_first",
        torch::nn::Conv2d{create_conv_options(
            /*in_planes = */   lczero::kInputPlanes,
            /*out_planes = */  filters,
            /*kerner_size = */ 3,
            /*stride = */      1,
            /*padding = */     1,
            /*bias = */        true)});

    torch::nn::Sequential m_res_tower_seq;
    for (int li = 0; li < res_blocks; li++)
        m_res_tower_seq->push_back(ResNetBlock(filters));
    m_res_tower = register_module("m_res_tower", m_res_tower_seq);

    m_conv_last = register_module(
        "m_conv_last",
        torch::nn::Conv2d{create_conv_options(
            /*in_planes = */   filters,
            /*out_planes = */  lczero::kNumOutputPolicyFilters,
            /*kerner_size = */ 3,
            /*stride = */      1,
            /*padding = */     1,
            /*bias = */        true)});

    if (true_if_v_else_q) {
        m_v_conv_last = register_module(
            "m_v_conv_last",
            torch::nn::Conv2d{create_conv_options(
                /*in_planes = */   lczero::kNumOutputPolicyFilters,
                /*out_planes = */  1,
                /*kerner_size = */ 3,
                /*stride = */      1,
                /*padding = */     1,
                /*bias = */        true)});

        m_v_fc = register_module("m_v_fc", torch::nn::Linear(64, 1));
    }

    // auto all_modules = modules(false);
    // https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#_CPPv4NK5torch2nn6Module7modulesEb
    for (auto m : modules(false)) {
        // if (m->name() == "torch::nn::Conv2dImpl") {
        //     torch::OrderedDict<std::string, torch::Tensor> named_parameters = m->named_parameters(false);
        //     torch::Tensor* ptr_w = named_parameters.find("weight");
        //     torch::nn::init::kaiming_normal_(*ptr_w, 0, torch::kFanOut, torch::kReLU);
        // } else
        if ((m->name() == "torch::nn::BatchNormImpl") ||
                 (m->name() == "torch::nn::GroupNormImpl")) {
            torch::OrderedDict<std::string, torch::Tensor> named_parameters = m->named_parameters(false);
            torch::Tensor* ptr_w = named_parameters.find("weight");
            torch::nn::init::constant_(*ptr_w, 1.0);
            torch::Tensor* ptr_b = named_parameters.find("bias");
            torch::nn::init::constant_(*ptr_b, 0.0);
        }
    }
}


torch::Tensor ResNetImpl::forward(torch::Tensor x) {
    x = m_conv_first->forward(x);
    x = m_res_tower->forward(x);
    x = m_conv_last->forward(x);

    if (true_if_v_else_q) {
        x = m_v_conv_last->forward(x);
        x = x.view({x.sizes()[0], -1});
        x = m_v_fc->forward(x);
    }

    return x;
}

}  // namespace probs
