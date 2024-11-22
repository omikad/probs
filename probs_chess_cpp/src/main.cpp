#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <fstream>
#include <torch/torch.h>

#include "infra/config_parser.h"
#include "infra/battle.h"

using namespace std;


int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << "<cmd> <path_to_yaml_file>\n";
        return 1;
    }

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

        // torch::Tensor tensor = torch::rand({2, 3});
        // cout << tensor << endl;

        // lczero::ChessBoard board;
        // board.SetFromFen(lczero::ChessBoard::kStartposFen);
        // cout << board.DebugString() << endl;

        if (command == "battle") {
            srand(time(NULL));
            probs::Battle::GoBattle(config);
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}