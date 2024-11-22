#pragma once

#include <vector>

#include "chess/bitboard.h"
#include "chess/board.h"

using namespace std;


namespace probs {

class IAgent {
    public:
        virtual ~IAgent() {}
        virtual vector<lczero::Move> GetActions(const vector<lczero::ChessBoard>& boards) = 0;
        virtual string GetName() = 0;
};


class RandomAgent : public IAgent {
    public:
        RandomAgent(const string& name): name(name) {};
        virtual vector<lczero::Move> GetActions(const vector<lczero::ChessBoard>& boards);
        virtual string GetName() {return name;};
        string name;
};

}  // namespace probs
