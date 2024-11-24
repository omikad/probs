#include <torch/torch.h>
#include <iostream>

#include "neural/network.h"
#include "infra/config_parser.h"

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
    ResNetBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
               torch::nn::Sequential downsample = torch::nn::Sequential()) {
        m_conv1 = register_module("conv1", torch::nn::Conv2d{create_conv3x3_options(inplanes, planes, stride)});
        m_bn1 = register_module("bn1", torch::nn::BatchNorm2d{planes});
        m_relu = register_module("relu", torch::nn::ReLU{true});
        m_conv2 = register_module("conv2", torch::nn::Conv2d{create_conv3x3_options(planes, planes)});
        m_bn2 = register_module("bn2", torch::nn::BatchNorm2d{planes});
        if (!downsample->is_empty()) {
            m_downsample = register_module("downsample", downsample);
        }
        m_stride = stride;
    }

    static const int64_t m_expansion = 1;

    torch::nn::Conv2d m_conv1{nullptr}, m_conv2{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr}, m_bn2{nullptr};
    torch::nn::ReLU m_relu{nullptr};
    torch::nn::Sequential m_downsample = torch::nn::Sequential();

    int64_t m_stride;

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor identity = x;

        torch::Tensor out = m_conv1->forward(x);
        out = m_bn1->forward(out);
        out = m_relu->forward(out);

        out = m_conv2->forward(out);
        out = m_bn2->forward(out);

        if (!m_downsample->is_empty())
        {
            identity = m_downsample->forward(x);
        }

        out += identity;
        out = m_relu->forward(out);

        return out;
    }
};

    // ResNet(const std::vector<int64_t> layers, int64_t num_classes = 1000,
    //        bool zero_init_residual = false, int64_t groups = 1,
    //        int64_t width_per_group = 64,
    //        std::vector<int64_t> replace_stride_with_dilation = {})

ResNet::ResNet(const ConfigParser& config_parser, const string& config_key_prefix) {
    int res_blocks = config_parser.GetInt(config_key_prefix + ".res_blocks");
    int filters = config_parser.GetInt(config_key_prefix + ".filters");
    cout << "ResNet config_key_prefix: " << config_key_prefix << endl;
    cout << "ResNet res_blocks: " << res_blocks << endl;
    cout << "ResNet filters: " << filters << endl;

    m_conv_first = register_module(
        "m_conv_first",
        torch::nn::Conv2d{create_conv_options(
            /*in_planes = */   lczero::kInputPlanes,
            /*out_planes = */  filters,
            /*kerner_size = */ 3,
            /*stride = */      1,
            /*padding = */     1,
            /*bias = */        true)});

    m_conv_last = register_module(
        "m_conv_last",
        torch::nn::Conv2d{create_conv_options(
            /*in_planes = */   filters,
            /*out_planes = */  lczero::kNumOutputPolicyFilters,
            /*kerner_size = */ 3,
            /*stride = */      1,
            /*padding = */     1,
            /*bias = */        true)});

//     m_bn1 = register_module("bn1", torch::nn::BatchNorm2d{m_inplanes});
//     m_relu = register_module("relu", torch::nn::ReLU{true});
//     m_maxpool = register_module("maxpool", torch::nn::MaxPool2d{torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding({1, 1})});

//     m_layer1 = register_module("layer1", _make_layer(64, layers.at(0)), 1);
//     m_layer2 = register_module("layer2", _make_layer(128, layers.at(1), 2), 1);
//     m_layer3 = register_module("layer3", _make_layer(256, layers.at(2), 2), 1);
//     m_layer4 = register_module("layer4", _make_layer(512, layers.at(3), 2), 1);

//     m_avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d( torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
//     m_fc = register_module("fc", torch::nn::Linear(512 * Block::m_expansion, num_classes));

//     // auto all_modules = modules(false);
//     // https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#_CPPv4NK5torch2nn6Module7modulesEb
//     for (auto m : modules(false)) {
//         if (m->name() == "torch::nn::Conv2dImpl") {
//             torch::OrderedDict<std::string, torch::Tensor>
//                 named_parameters = m->named_parameters(false);
//             torch::Tensor* ptr_w = named_parameters.find("weight");
//             torch::nn::init::kaiming_normal_(*ptr_w, 0, torch::kFanOut, torch::kReLU);
//         }
//         else if ((m->name() == "torch::nn::BatchNormImpl") ||
//                  (m->name() == "torch::nn::GroupNormImpl")) {
//             torch::OrderedDict<std::string, torch::Tensor>
//                 named_parameters = m->named_parameters(false);
//             torch::Tensor* ptr_w = named_parameters.find("weight");
//             torch::nn::init::constant_(*ptr_w, 1.0);
//             torch::Tensor* ptr_b = named_parameters.find("bias");
//             torch::nn::init::constant_(*ptr_b, 0.0);
//         }
//     }
}

// torch::nn::Sequential ResNet::_make_layer(int64_t planes, int64_t blocks, int64_t stride) {
//     torch::nn::Sequential downsample = torch::nn::Sequential();
//     if ((stride != 1) || (m_inplanes != planes * ResNetBlock::m_expansion)) {
//         downsample = torch::nn::Sequential(
//             torch::nn::Conv2d(create_conv1x1_options(m_inplanes, planes * ResNetBlock::m_expansion, stride)),
//             torch::nn::BatchNorm2d(planes * ResNetBlock::m_expansion));
//     }

//     torch::nn::Sequential layers;

//     layers->push_back(ResNetBlock(m_inplanes, planes, stride, downsample));
//     m_inplanes = planes * ResNetBlock::m_expansion;
//     for (int64_t i = 0; i < blocks; i++) {
//         layers->push_back(ResNetBlock(m_inplanes, planes, 1,
//                                 torch::nn::Sequential()));
//     }

//     return layers;
// }

torch::Tensor ResNet::forward(torch::Tensor x) {
    x = m_conv_first->forward(x);
    x = m_conv_last->forward(x);

    // x = m_bn1->forward(x);
    // x = m_relu->forward(x);
    // x = m_maxpool->forward(x);

    // x = m_layer1->forward(x);
    // x = m_layer2->forward(x);
    // x = m_layer3->forward(x);
    // x = m_layer4->forward(x);

    // x = m_avgpool->forward(x);
    // x = torch::flatten(x, 1);
    // x = m_fc->forward(x);

    return x;
}

}  // namespace probs
