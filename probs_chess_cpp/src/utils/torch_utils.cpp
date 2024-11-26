#include "utils/torch_utils.h"

using namespace std;


namespace probs {

string DebugString(const torch::Tensor& tensor) {
    ostringstream stream;
    stream << "Tensor " << tensor.toString() << " " << tensor.sizes() << ":" << endl;

    const auto flatten = tensor.contiguous();
    vector<float> content(flatten.data_ptr<float>(), flatten.data_ptr<float>() + flatten.numel());

    if (content.size() > 0) {
        float sum_el = 0;
        float min_el = content[0];
        float max_el = content[0];
        for (int i = 0; i < content.size(); i++) {
            sum_el += content[i];
            min_el = min(min_el, content[i]);
            max_el = max(max_el, content[i]);
        }

        stream << "  Sum: " << sum_el << endl;
        stream << "  Cnt: " << content.size() << endl;
        stream << "  Min: " << min_el << endl;
        stream << "  Max: " << max_el << endl;
    }

    return stream.str();
}

string DebugString(const torch::nn::Module& net) {
    ostringstream stream;

    stream << "Torch Module:" << endl;

    long long total_parameters = 0;

    // auto all_modules = modules(false);
    // https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_module.html#_CPPv4NK5torch2nn6Module7modulesEb
    for (auto m : net.modules(false)) {
        stream << "* " << m->name() << ":" << endl;
        torch::OrderedDict<std::string, torch::Tensor> named_parameters = m->named_parameters(false);
        for (const auto& kvp: named_parameters) {
            stream << "  " << kvp.key() << " parameters: " << kvp.value().numel() << endl;
            total_parameters += kvp.value().numel();
        }
    }

    stream << "Total parameters: " << total_parameters << endl;

    return stream.str();
}

}   // namespace probs