#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <fstream>
#include <torch/torch.h>

#include "infra/config_parser.h"
#include "chess/bitboard.h"
#include "chess/board.h"

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
        ConfigParser parser(configFilePath);

        cout << "checkpoints_dir: " << parser.get_string("infra.checkpoints_dir") << "\n";
        cout << "n_high_level_iterations: " << parser.get_int("training.n_high_level_iterations") << "\n";
        cout << "dataset_drop_ratio: " << parser.get_double("training.dataset_drop_ratio") << "\n";

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // torch::Tensor tensor = torch::rand({2, 3});
    // cout << tensor << endl;

    lczero::InitializeMagicBitboards();

    // lczero::ChessBoard board;
    // board.SetFromFen(lczero::ChessBoard::kStartposFen);
    // cout << board.DebugString() << endl;
    // for (const auto& move : board.GenerateLegalMoves()) {
    //     cout << move.as_string() << endl;
    // }

    return 0;
}