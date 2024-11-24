#include <iostream>
#include <vector>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "infra/player.h"
#include "utils/exception.h"

using namespace std;


namespace probs {

vector<lczero::Move> RandomPlayer::GetActions(const vector<lczero::ChessBoard>& boards) {
    vector<lczero::Move> picked_moves(boards.size());

    for(int bi = 0; bi < boards.size(); bi++) {
        const auto& board = boards[bi];

        auto legal_moves = board.GenerateLegalMoves();

        if (legal_moves.size() == 0)
            throw Exception("No legal moves found");

        int mi = rand() % (legal_moves.size());

        picked_moves[bi] = legal_moves[mi];
    }

    return picked_moves;
}


VQResnetPlayer::VQResnetPlayer(const ConfigParser& config_parser, const string& config_key_prefix, const string& name):
        name(name),
        v_model(config_parser, config_key_prefix + ".model.v"),
        q_model(config_parser, config_key_prefix + ".model.q") {}


vector<lczero::Move> VQResnetPlayer::GetActions(const vector<lczero::ChessBoard>& boards) {
    throw Exception("TODO");
}


}  // namespace probs
