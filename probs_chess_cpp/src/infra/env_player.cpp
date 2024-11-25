#include <string>

#include "infra/env_player.h"
#include "chess/bitboard.h"
#include "chess/board.h"
#include "utils/callbacks.h"
#include "chess/position.h"
#include "env_player.h"

using namespace std;


namespace probs {

EnvPlayer::EnvPlayer(const string& starting_fen, const int n_max_episode_steps) :
        n_max_episode_steps(n_max_episode_steps),
        tree(starting_fen) {
    ComputeGameResult();
}

void EnvPlayer::Move(const lczero::Move &move) {
    if (game_result != lczero::GameResult::UNDECIDED) return;

    auto& board = tree.Last().GetBoard();
    auto new_move = board.GetModernMove(move);

    tree.Append(tree.LastIndex(), new_move);

    ComputeGameResult();
}

void EnvPlayer::ComputeGameResult() {
    game_result = tree.ComputeGameResult(tree.LastIndex());
    if (game_result == lczero::GameResult::UNDECIDED && tree.Last().GetGamePly() >= n_max_episode_steps)
        game_result = lczero::GameResult::DRAW;
}

} // namespace probs
