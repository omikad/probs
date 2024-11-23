#pragma once
#include <string>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/position.h"

using namespace std;


namespace probs {

class EnvPlayer {
    public:
        EnvPlayer(string starting_fen, int n_max_episode_steps);
        void Move(const lczero::Move& move);
        const lczero::PositionHistory& History() const {return history;}
        lczero::GameResult GameResult() const {return game_result;}
        const lczero::ChessBoard& LastChessBoard() const {return history.Last().GetBoard();}
        int Ply() const {return history.Last().GetGamePly();}

    private:
        void ComputeGameResult();
        int n_max_episode_steps;
        lczero::PositionHistory history;
        lczero::GameResult game_result;
};

}  // namespace probs
