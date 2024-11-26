#pragma once

#include <iostream>
#include <vector>
#include <string>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/game_tree.h"
#include "chess/position.h"
#include "utils/callbacks.h"
#include "infra/player.h"
#include "infra/env_player.h"
#include "utils/exception.h"
#include "neural/encoder.h"

using namespace std;


namespace probs {

void PositionHistoryTree_eq_PositionHistory() {
    srand(333);
    const int N_GAMES = 1000;
    const int N_MAX_STEPS = 500;

    auto check_correctness = [](const int test_i, const string& test_name, const lczero::PositionHistory& lchistory, const PositionHistoryTree& tree, const int node) {
        lczero::PositionHistory tree_history = tree.ToLczeroHistory(node);

        if (lchistory.GetLength() != tree_history.GetLength()) {
            cout << "Test " << test_i << " " << test_name << ": position lengths are different" << endl;
            cout << "expected length: " << lchistory.GetLength() << endl;
            cout << "actual length: " << tree_history.GetLength() << endl;
            throw Exception("Failed");
        }

        int expected_transforms_out;
        lczero::InputPlanes expected_input_planes = lczero::EncodePositionForNN(
            lczero::InputFormat::INPUT_112_WITH_CANONICALIZATION_V2,
            lchistory,
            8,
            lczero::FillEmptyHistory::FEN_ONLY,
            &expected_transforms_out);

        int actual_transforms_out;
        lczero::InputPlanes actual_input_planes = lczero::EncodePositionForNN(
            lczero::InputFormat::INPUT_112_WITH_CANONICALIZATION_V2,
            tree_history,
            8,
            lczero::FillEmptyHistory::FEN_ONLY,
            &actual_transforms_out);

        if (expected_transforms_out != actual_transforms_out) {
            cout << "Test " << test_i << " " << test_name << ": expected_transforms_out != actual_transforms_out, " << expected_transforms_out << " != " << actual_transforms_out << endl;
            throw Exception("Failed");
        }

        if (expected_input_planes.size() != actual_input_planes.size()) {
            cout << "Test " << test_i << " " << test_name << ": sizes of planes are different" << endl;
            throw Exception("Failed");
        }

        for (int i = 0; i < expected_input_planes.size(); i++) {
            auto expected_input = expected_input_planes[i];
            auto actual_input = actual_input_planes[i];
            if (expected_input.mask != actual_input.mask) {
                cout << "Test " << test_i << " " << test_name << " plane " << i << ": masks are different" << endl;
                throw Exception("Failed");
            }
            if (abs(expected_input.value - actual_input.value) >= 1e-5) {
                cout << "Test " << test_i << " " << test_name << " plane " << i << ": values are different" << endl;
                throw Exception("Failed");                
            }
        }
    };

    RandomPlayer player("random player");
    int test_i = 0;

    for (int game_i = 0; game_i < N_GAMES; game_i++) {
        vector<lczero::Move> moves;

        {
            lczero::PositionHistory lchistory(lczero::ChessBoard::kStartposFen);
            EnvPlayer env_player(lczero::ChessBoard::kStartposFen, N_MAX_STEPS);

            while (env_player.GameResult() == lczero::GameResult::UNDECIDED) {
                vector<PositionHistoryTree*> trees = { &env_player.Tree() };
                auto move = player.GetActions(trees)[0];
                moves.push_back(move);

                env_player.Move(move);

                lchistory.Append(lchistory.Last().GetBoard().GetModernMove(move));

                // cout << "selected move " << move.as_string() << endl;

                check_correctness(test_i, "part 1", lchistory, env_player.Tree(), env_player.Tree().LastIndex());
                test_i++;
            }
        }

        // Play some moves differently
        int trim_i = 1 + rand() % (moves.size() - 2);
        while (moves.size() >= trim_i) moves.pop_back();

        {
            lczero::PositionHistory lchistory(lczero::ChessBoard::kStartposFen);
            EnvPlayer env_player(lczero::ChessBoard::kStartposFen, N_MAX_STEPS);

            // Replay first part
            for (int step_i = 0; step_i < (int)moves.size(); step_i++) {
                auto move = moves[step_i];
                env_player.Move(move);
                assert (env_player.GameResult() == lczero::GameResult::UNDECIDED);

                lchistory.Append(lchistory.Last().GetBoard().GetModernMove(move));

                check_correctness(test_i, "part 2", lchistory, env_player.Tree(), step_i + 1);
                test_i++;
            }

            int node_i = trim_i - 1;   // after replaying this will be a node where moves start to play differently

            // New random moves
            while (env_player.GameResult() == lczero::GameResult::UNDECIDED) {
                vector<PositionHistoryTree*> trees = { &env_player.Tree() };
                auto move = player.GetActions(trees)[0];

                env_player.Move(move);

                lchistory.Append(lchistory.Last().GetBoard().GetModernMove(move));
                node_i = env_player.Tree().LastIndex();   // newly added node

                // cout << "selected move " << move.as_string() << endl;

                check_correctness(test_i, "part 3", lchistory, env_player.Tree(), env_player.Tree().LastIndex());
                test_i++;
            }
        }
    }

    cout << "OK!!!" << endl;
}

}   // namespace probs