#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <fstream>
#include <torch/torch.h>

#include "infra/config_parser.h"
#include "infra/battle.h"
#include "training/train.h"
#include "utils/exception.h"
#include "infra/tests.h"
#include "infra/uci_impl.h"

using namespace std;


string getUciEngineConfig() {
    vector<string> config_paths {
        "uci_engine.yaml",
        "configs/uci_engine.yaml",
        "../uci_engine.yaml",
        "../configs/uci_engine.yaml",
    };
    for (int ci = 0; ci < config_paths.size(); ci++) {
        ifstream file(config_paths[ci]);
        if (file.good())
            return config_paths[ci];
    }
    throw probs::Exception("Can't find uci_engine.yaml");
}


int main(int argc, char* argv[]) {
    std::locale::global(std::locale("en_US.UTF-8"));
    srand(time(NULL));
    lczero::InitializeMagicBitboards();

    try {
        if (argc == 1) {
            ConfigParser config(getUciEngineConfig());

            for (int cmdi = 0; cmdi < 10; cmdi++) {
                string command;
                cin >> command;

                if (command == "uci") {
                    probs::UciImpl uci(config);
                    uci.Run();
                    break;
                }
            }
        }
        else if (argc != 3) {
            cerr << "Run as UCI Engine: " << argv[0] << "\n";
            cerr << "Using with command: " << argv[0] << "<cmd> <path_to_yaml_file>\n";
            return 1;
        }
        else {
            string command = argv[1];
            string config_file_path = argv[2];

            ifstream file(config_file_path);
            if (!file.good()) {
                cerr << "Error: Unable to open file at " << config_file_path << "\n";
                return 1;
            }

            ConfigParser config(config_file_path);

            if (command == "battle") {
                probs::GoBattle(config);
            }
            else if (command == "train") {
                probs::ProbsImpl probs(config);
                probs.GoTrain();
            }
            else if (command == "test_V_predict_self_play") {
                probs::V_predict_self_play(config);
            }
            else if (command == "test_show_v_starting_estimation") {
                probs::ShowVModelStartingFenEstimation(config);
            }
            else
                throw probs::Exception("Unknown command " + command);
        }

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}