#pragma once

#include <string>
#include <optional>
#include <memory>

#include "chess/board.h"
#include "chess/position.h"
#include "chess/game_tree.h"
#include "utils/exception.h"

using namespace std;


namespace probs {
namespace python {

class ChessEnv {
    public:
        ChessEnv(const optional<string> start_fen, optional<int> max_episode_steps) :
                game_tree(
                    start_fen.value_or(lczero::ChessBoard::kStartposFen),
                    max_episode_steps.has_value() && max_episode_steps.value() > 0 ? max_episode_steps.value() : 450) {
        }

        void move(const optional<string> move_str) {
            if (!move_str.has_value())
                return;
            lczero::Move move(move_str.value(), game_tree.LastPosition().IsBlackToMove());
            game_tree.Move(-1, move);
        }

        string game_state() {
            if (game_tree.GetGameResult(-1) == lczero::GameResult::WHITE_WON) return "white_won";
            if (game_tree.GetGameResult(-1) == lczero::GameResult::BLACK_WON) return "black_won";
            if (game_tree.GetGameResult(-1) == lczero::GameResult::DRAW) return "draw";
            return "undecided";
        }

        vector<string> legal_moves() const {
            auto ms = game_tree.node_valid_moves.back();
            bool is_black = game_tree.LastPosition().IsBlackToMove();
            vector<string> result;
                for (auto m : ms) {
                    m = game_tree.LastPosition().GetBoard().GetLegacyMove(m);
                    if (is_black) m.Mirror();
                    result.push_back(m.as_string());
                }
            return result;
        }

        vector<int> policy_indices() const {
            auto ms = game_tree.node_valid_moves.back();
            vector<int> result;
            for (auto m : ms) {
            result.push_back(m.as_nn_index(/* transform= */ 0));
            }
            return result;
        }

        string as_string() const {
            bool is_black = game_tree.LastPosition().IsBlackToMove();
            return (is_black ? game_tree.LastPosition().GetThemBoard() : game_tree.LastPosition().GetBoard())
                .DebugString();
        }

    private:
        PositionHistoryTree game_tree;
};

}  // namespace python
}  // namespace probs