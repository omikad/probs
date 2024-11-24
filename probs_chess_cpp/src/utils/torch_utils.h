#include <torch/torch.h>
#include <iostream>

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

}       // namespace probs