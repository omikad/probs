#include "infra/env_player.h"

using namespace std;


namespace probs {

EnvPlayer::EnvPlayer(const string& starting_fen, const int n_max_episode_steps) :
        n_max_episode_steps(n_max_episode_steps),
        tree(starting_fen) {
}


void EnvPlayer::Move(const lczero::Move &move) {
    assert(GameResult(-1) == lczero::GameResult::UNDECIDED);

    auto& board = tree.Last().GetBoard();
    auto new_move = board.GetModernMove(move);

    tree.Append(tree.LastIndex(), new_move);
}


lczero::GameResult EnvPlayer::GameResult(const int node) const {
    auto game_result = tree.ComputeGameResult(node >= 0 ? node : tree.LastIndex());
    if (game_result == lczero::GameResult::UNDECIDED && tree.Last().GetGamePly() >= n_max_episode_steps)
        game_result = lczero::GameResult::DRAW;
    return game_result;
}


} // namespace probs
