#include <iostream>
#include <vector>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "infra/agent.h"

using namespace std;


namespace probs {

vector<lczero::Move> RandomAgent::getActions(const vector<lczero::ChessBoard>& boards) {
    vector<lczero::Move> pickedMoves(boards.size());

    for(int bi = 0; bi < boards.size(); bi++) {
        const auto& board = boards[bi];

        auto legalMoves = board.GenerateLegalMoves();
        for (const auto& move : legalMoves) {
            cout << move.as_string() << endl;
        }
        int mi = rand() % (legalMoves.size());

        cout << board.DebugString() << endl;
        cout << "Random agent picked move " << legalMoves[mi].as_string() << "\n";

        pickedMoves[bi] = legalMoves[mi];
    }

    return pickedMoves;
}

}  // namespace probs
