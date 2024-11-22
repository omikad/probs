#pragma once

#include <vector>

#include "chess/bitboard.h"
#include "chess/board.h"

using namespace std;


namespace probs {

class IAgent {
    public:
        virtual ~IAgent() {}
        virtual vector<lczero::Move> getActions(const vector<lczero::ChessBoard>& boards) = 0;
};


class RandomAgent : public IAgent {
    public:
        virtual vector<lczero::Move> getActions(const vector<lczero::ChessBoard>& boards);
};

}  // namespace probs
