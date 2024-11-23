#pragma once
#include <string>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/position.h"

using namespace std;


namespace probs {

class EnvPlayer {
    public:
        EnvPlayer(int n_max_episode_steps);
        void StartNew(const string& starting_fen);
        void Move(const lczero::Move& move);
        const lczero::PositionHistory& History() {return history;}
        const lczero::GameResult GameResult() {return game_result;}
        const lczero::ChessBoard LastChessBoard() {return history.Last().GetBoard();}
        int Ply() {return history.Last().GetGamePly();}

    private:
        void ComputeGameResult();
        int n_max_episode_steps;
        lczero::PositionHistory history;
        lczero::GameResult game_result;
};

}  // namespace probs
