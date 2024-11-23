#include <string>

#include "infra/env_player.h"
#include "chess/bitboard.h"
#include "chess/board.h"
#include "utils/callbacks.h"
#include "chess/position.h"
#include "env_player.h"

using namespace std;


namespace probs {

EnvPlayer::EnvPlayer(int n_max_episode_steps) :
        n_max_episode_steps(n_max_episode_steps)
    {}

void EnvPlayer::StartNew(const string &starting_fen) {
    lczero::ChessBoard starting_board;
    int no_capture_ply;
    int full_moves;
    starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);

    history.Reset(starting_board, no_capture_ply, full_moves * 2 - (starting_board.flipped() ? 1 : 2));
    ComputeGameResult();
}

void EnvPlayer::Move(const lczero::Move &move) {
    if (game_result != lczero::GameResult::UNDECIDED) return;

    auto board = history.Last().GetBoard();
    auto new_move = board.GetModernMove(move);

    history.Append(new_move);

    ComputeGameResult();
}

void EnvPlayer::ComputeGameResult() {
    game_result = history.ComputeGameResult();
    if (game_result == lczero::GameResult::UNDECIDED && history.Last().GetGamePly() >= n_max_episode_steps)
        game_result = lczero::GameResult::DRAW;
}

} // namespace probs
