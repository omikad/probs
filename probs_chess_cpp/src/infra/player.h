#pragma once

#include <ATen/Device.h>
#include <vector>

#include "chess/position.h"
#include "neural/network.h"

using namespace std;


namespace probs {

class IPlayer {
    public:
        virtual ~IPlayer() {}
        virtual vector<lczero::Move> GetActions(const vector<lczero::PositionHistory>& history) = 0;
        virtual string GetName() = 0;
};


class RandomPlayer : public IPlayer {
    public:
        RandomPlayer(const string& name): name(name) {};
        virtual vector<lczero::Move> GetActions(const vector<lczero::PositionHistory>& history);
        virtual string GetName() {return name;};
        const string name;
};


class NStepLookaheadPlayer : public IPlayer {
    public:
        NStepLookaheadPlayer(const string& name, const int depth): name(name), depth(depth) {};
        virtual vector<lczero::Move> GetActions(const vector<lczero::PositionHistory>& history);
        virtual string GetName() {return name;};
        const string name;
        const int depth;
};


class VQResnetPlayer : public IPlayer {
    public:
        VQResnetPlayer(const ConfigParser& config_parser, const string& config_key_prefix, const string& name);
        virtual vector<lczero::Move> GetActions(const vector<lczero::PositionHistory>& history);
        virtual string GetName() {return name;};
        const string name;
    private:
        ResNet v_model;
        ResNet q_model;
        at::Device device;
};

}  // namespace probs
