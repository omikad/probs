#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

#include "training/model_keeper.h"


using namespace std;


namespace probs {


torch::optim::AdamW adam_factory(const ConfigParser& config_parser, const string& training_key, ResNet model) {
    float learning_rate = config_parser.GetDouble(training_key + ".learning_rate", true, 0);
    float weight_decay = config_parser.GetDouble(training_key + ".weight_decay", true, 0);

    torch::optim::AdamWOptions adam_options(learning_rate);
    adam_options.weight_decay(weight_decay);

    return torch::optim::AdamW(model->parameters(), adam_options);
}


/// @brief Constructor to load models for evaluation
ModelKeeper::ModelKeeper(const ConfigParser& config_parser, const string& model_key) :
        v_model(ResNet(config_parser, model_key + ".v", true)),
        q_model(ResNet(config_parser, model_key + ".q", false)),
        v_optimizer(torch::optim::AdamW(torch::nn::Sequential()->parameters(), torch::optim::AdamWOptions())),
        q_optimizer(torch::optim::AdamW(torch::nn::Sequential()->parameters(), torch::optim::AdamWOptions())),
        checkpoints_dir("")
{
    if (config_parser.KeyExist(model_key + ".checkpoint")) {
        string filename = config_parser.GetString(model_key + ".checkpoint");

        if (filename.size() > 0 && filename.back() == '*') {
            if (config_parser.GetInt("infra.uci_mode", false, 0) == 0)
                cout << "[MODEL_KEEPER] Models V and Q loaded from " << filename << endl;
            filename.pop_back();
            torch::load(v_model, filename + "_v.ckpt");
            torch::load(q_model, filename + "_q.ckpt");
            return;
        }
    }

    cout << "[MODEL_KEEPER] Models V and Q created from scratch" << endl;
}


/// @brief Constructor to load models for training
ModelKeeper::ModelKeeper(const ConfigParser& config_parser, const string& config_v_key, const string& config_q_key, const string& training_key) :
        v_model(ResNet(config_parser, config_v_key, true)),
        q_model(ResNet(config_parser, config_q_key, false)),
        v_optimizer(adam_factory(config_parser, training_key, v_model)),
        q_optimizer(adam_factory(config_parser, training_key, q_model)),
        checkpoints_dir(config_parser.GetString(training_key + ".checkpoints_dir")) {
    if (config_parser.KeyExist(training_key + ".checkpoint")) {
        string filename = config_parser.GetString(training_key + ".checkpoint");

        if (filename.size() > 0 && filename.back() == '*') {
            cout << "[MODEL_KEEPER] Models V and Q loaded with their optimizers from " << filename << endl;
            filename.pop_back();
            torch::load(v_model, filename + "_v.ckpt");
            torch::load(q_model, filename + "_q.ckpt");
            torch::load(v_optimizer, filename + "_vo.ckpt");
            torch::load(q_optimizer, filename + "_qo.ckpt");
            return;
        }
    }

    cout << "ModelKeeper V model: ";
    cout << DebugString(*v_model);
    cout << "ModelKeeper Q model: ";
    cout << DebugString(*q_model);

    cout << "[MODEL_KEEPER] Models V and Q created from scratch" << endl;
}


void ModelKeeper::SaveCheckpoint() {
    std::time_t t =  std::time(NULL);
    std::tm tm    = *std::localtime(&t);

    std::ostringstream os;
    os << std::put_time(&tm, "%Y%m%d-%H%M%S");

    string path_base = checkpoints_dir + "/ckpt_" + os.str();

    torch::save(v_model, path_base + "_v.ckpt");
    torch::save(q_model, path_base + "_q.ckpt");
    torch::save(v_optimizer, path_base + "_vo.ckpt");
    torch::save(q_optimizer, path_base + "_qo.ckpt");

    cout << "[TRAIN] Checkpoint saved to " << path_base << "*" << endl;
}


void ModelKeeper::SetEvalMode() {
    v_model->eval();
    q_model->eval();
}


void ModelKeeper::SetTrainMode() {
    v_model->train();
    q_model->train();
}


void ModelKeeper::To(const at::Device& device) {
    v_model->to(device);
    q_model->to(device);
}


}   // namespace probs