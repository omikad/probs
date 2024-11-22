#include <iostream>
#include <vector>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "infra/agent.h"
#include "utils/exception.h"

using namespace std;


namespace probs {

vector<lczero::Move> RandomAgent::GetActions(const vector<lczero::ChessBoard>& boards) {
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

}  // namespace probs
