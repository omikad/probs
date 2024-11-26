#pragma once

#include <ATen/Device.h>
#include <torch/torch.h>
#include <vector>
#include <iostream>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/game_tree.h"
#include "chess/policy_map.h"
#include "chess/position.h"
#include "neural/torch_encoder.h"
#include "utils/exception.h"
#include "utils/torch_utils.h"


namespace probs {

class IPlayer {
    public:
        virtual ~IPlayer() {}
        virtual std::vector<lczero::Move> GetActions(std::vector<PositionHistoryTree*>& history) = 0;
        virtual std::string GetName() = 0;
};


class RandomPlayer : public IPlayer {
    public:
        RandomPlayer(const std::string& name): name(name) {};
        virtual std::vector<lczero::Move> GetActions(std::vector<PositionHistoryTree*>& history);
        virtual std::string GetName() {return name;};
        const std::string name;
};


class NStepLookaheadPlayer : public IPlayer {
    public:
        NStepLookaheadPlayer(const std::string& name, const int depth): name(name), depth(depth) {};
        virtual std::vector<lczero::Move> GetActions(std::vector<PositionHistoryTree*>& history);
        virtual std::string GetName() {return name;};
        const std::string name;
        const int depth;
};


class VQResnetPlayer : public IPlayer {
    public:
        VQResnetPlayer(const ConfigParser& config_parser, const std::string& config_key_prefix, const std::string& name);
        virtual std::vector<lczero::Move> GetActions(std::vector<PositionHistoryTree*>& history);
        virtual std::string GetName() {return name;};
        const std::string name;
    private:
        ResNet v_model;
        ResNet q_model;
        at::Device device;
};

}  // namespace probs
