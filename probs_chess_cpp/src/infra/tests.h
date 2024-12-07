#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <torch/torch.h>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/game_tree.h"
#include "chess/position.h"
#include "infra/player.h"
#include "utils/callbacks.h"
#include "utils/exception.h"
#include "training/model_keeper.h"
#include "training/v_train.h"
#include "neural/encoder.h"
#include "neural/torch_encoder.h"

using namespace std;


namespace probs {

void V_predict_self_play(const ConfigParser& config_parser) {
    srand(333);
    const int N_GAMES = 100;
    const int N_MAX_STEPS = 600;
    const int EXPLORATION_NUM_FIRST_MOVES = 30;

    torch::NoGradGuard no_grad;
    ModelKeeper model_keeper(config_parser, "player1.model");
    model_keeper.SetEvalMode();
    model_keeper.v_model->to(torch::kCPU);
    model_keeper.q_model->to(torch::kCPU);

    map<string, float> game_result_last_pred_sums;
    map<string, float> game_result_last_pred_cnts;

    for (int game_i = 0; game_i < N_GAMES; game_i++) {
        PositionHistoryTree tree(lczero::ChessBoard::kStartposFen, N_MAX_STEPS);

        vector<bool> is_black_to_move_arr;
        vector<float> predictions;

        while (tree.GetGameResult(-1) == lczero::GameResult::UNDECIDED) {
            // cout << "Board at step " << ply << ":\n" << curr_board.DebugString() << endl;

            int transform_out;
            auto input_planes = Encode(tree, tree.LastIndex(), &transform_out);

            torch::Tensor input = torch::zeros({1, lczero::kInputPlanes, 8, 8});
            for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
                const auto& plane = input_planes[pi];
                for (auto bit : lczero::IterateBits(plane.mask))
                    input[0][pi][bit / 8][bit % 8] = plane.value;
            }
            input = input.to(torch::kCPU);

            torch::Tensor pred = model_keeper.v_model->forward(input);

            // cout << "Pred: " << DebugString(pred) << endl;

            is_black_to_move_arr.push_back(tree.LastPosition().IsBlackToMove());
            predictions.push_back(pred[0][0].item<float>());

            vector<PositionHistoryTree*> trees = {&tree};
            vector<int> nodes = {tree.LastIndex()};
            at::Device device = torch::kCPU;
            auto encoded_batch = GetQModelEstimation(trees, nodes, model_keeper.q_model, device);

            auto move = GetMoveWithExploration(encoded_batch->moves_estimation[0], tree.LastPosition().GetGamePly(), false, EXPLORATION_NUM_FIRST_MOVES);

            // cout << "PLY=" << env_player.Ply() << " " << "Player selected move=" << move.as_string() << " V prediction=" << predictions.back() << endl;

            tree.Move(-1, move);
        }

        string gr = tree.GetGameResult(-1) == lczero::GameResult::UNDECIDED ? "UNDECIDED"
            : tree.GetGameResult(-1) == lczero::GameResult::BLACK_WON ? "BLACK_WON"
            : tree.GetGameResult(-1) == lczero::GameResult::DRAW ? "DRAW"
            : tree.GetGameResult(-1) == lczero::GameResult::WHITE_WON ? "WHITE_WON"
            : "HZ";
        cout << "Game result " << gr << " last prediction: " << predictions.back() << endl;

        string key = (is_black_to_move_arr.back() ? "black move, outcome " : "white move, outcome ") + gr;
        game_result_last_pred_sums[key] += predictions.back();
        game_result_last_pred_cnts[key]++;
    }

    for (auto& kvp : game_result_last_pred_sums)
        cout << "Average last prediction for game result " << kvp.first << " = " << game_result_last_pred_sums[kvp.first] / game_result_last_pred_cnts[kvp.first] << endl;
}


void ShowVModelStartingFenEstimation(const ConfigParser& config_parser) {
    torch::NoGradGuard no_grad;

    ModelKeeper model_keeper(config_parser, "player1.model");

    at::Device device = GetDeviceFromConfig(config_parser);
    model_keeper.To(device);
    model_keeper.SetEvalMode();

    float vval = GetVScoreOnStartingBoard(model_keeper.v_model, device);
    cout << "V model estimation for starting fen: " << vval << endl;
}


}   // namespace probs