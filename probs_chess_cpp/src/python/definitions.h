#pragma once

#include <string>
#include <optional>
#include <memory>

#include "chess/board.h"
#include "chess/position.h"
#include "utils/exception.h"
#include "infra/env_player.h"

using namespace std;


namespace probs {
namespace python {

class ChessEnv {
    public:
        ChessEnv(const optional<string> start_fen, optional<int> max_episode_steps) :
                env_player(EnvPlayer(
                    start_fen.value_or(lczero::ChessBoard::kStartposFen),
                    max_episode_steps.has_value() && max_episode_steps.value() > 0 ? max_episode_steps.value() : 450)) {
        }

        void move(const optional<string> move_str) {
            if (!move_str.has_value())
                return;
            lczero::Move move(move_str.value(), env_player.History().IsBlackToMove());
            env_player.Move(move);
        }

        string game_state() {
            if (env_player.GameResult() == lczero::GameResult::WHITE_WON) return "white_won";
            if (env_player.GameResult() == lczero::GameResult::BLACK_WON) return "black_won";
            if (env_player.GameResult() == lczero::GameResult::DRAW) return "draw";
            return "undecided";
        }

        vector<string> legal_moves() const {
            auto ms = env_player.LastChessBoard().GenerateLegalMoves();
            bool is_black = env_player.History().IsBlackToMove();
            vector<string> result;
                for (auto m : ms) {
                    m = env_player.LastChessBoard().GetLegacyMove(m);
                    if (is_black) m.Mirror();
                    result.push_back(m.as_string());
                }
            return result;
        }

        vector<int> policy_indices() const {
            auto ms = env_player.LastChessBoard().GenerateLegalMoves();
            vector<int> result;
            for (auto m : ms) {
            result.push_back(m.as_nn_index(/* transform= */ 0));
            }
            return result;
        }

        string as_string() const {
            bool is_black = env_player.History().IsBlackToMove();
            return (is_black ? env_player.History().Last().GetThemBoard() : env_player.History().Last().GetBoard())
                .DebugString();
        }

    private:
        EnvPlayer env_player;
};

}  // namespace python
}  // namespace probs