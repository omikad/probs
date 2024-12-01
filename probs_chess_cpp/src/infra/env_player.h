#pragma once
#include <string>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/position.h"
#include "chess/game_tree.h"


namespace probs {

class EnvPlayer {
    public:
        EnvPlayer(const std::string& starting_fen, const int n_max_episode_steps);
        void Move(const lczero::Move& move);
        PositionHistoryTree& Tree() {return tree;}
        const lczero::Position& LastPosition() const {return tree.Last();}
        lczero::GameResult GameResult() const {return game_result;}
        const lczero::ChessBoard& LastChessBoard() const {return tree.Last().GetBoard();}
        int Ply() {return tree.Last().GetGamePly();}
        void ComputeGameResult();

    private:
        const int n_max_episode_steps;
        PositionHistoryTree tree;
        lczero::GameResult game_result;
};

}  // namespace probs
