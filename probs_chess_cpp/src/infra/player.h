#pragma once

#include <vector>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "neural/network.h"

using namespace std;


namespace probs {

class IPlayer {
    public:
        virtual ~IPlayer() {}
        virtual vector<lczero::Move> GetActions(const vector<lczero::ChessBoard>& boards) = 0;
        virtual string GetName() = 0;
};


class RandomPlayer : public IPlayer {
    public:
        RandomPlayer(const string& name): name(name) {};
        virtual vector<lczero::Move> GetActions(const vector<lczero::ChessBoard>& boards);
        virtual string GetName() {return name;};
        string name;
};


class VQResnetPlayer : public IPlayer {
    public:
        VQResnetPlayer(const ConfigParser& config_parser, const string& config_key_prefix, const string& name);
        virtual vector<lczero::Move> GetActions(const vector<lczero::ChessBoard>& boards);
        virtual string GetName() {return name;};
        string name;
    private:
        ResNet v_model;
        ResNet q_model;
};

}  // namespace probs
