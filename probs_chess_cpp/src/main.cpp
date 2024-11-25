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

using namespace std;


int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << "<cmd> <path_to_yaml_file>\n";
        return 1;
    }

    srand(time(NULL));

    string command = argv[1];
    string configFilePath = argv[2];

    ifstream file(configFilePath);
    if (!file.good()) {
        cerr << "Error: Unable to open file at " << configFilePath << "\n";
        return 1;
    }

    try {
        ConfigParser config(configFilePath);

        lczero::InitializeMagicBitboards();

        if (command == "battle") {
            probs::Battle::GoBattle(config);
        }
        else if (command == "train") {
            probs::ProbsImpl probs(config);
            probs.GoTrain();
        }
        else if (command == "tests") {
            probs::PositionHistoryTree_eq_PositionHistory();
        }
        else
            throw probs::Exception("Unknown command " + command);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}