#pragma once

#include <vector>
#include <math.h>

#include "chess/bitboard.h"
#include "utils/exception.h"


namespace probs {


struct MoveEstimation {
    lczero::Move move;
    float score;
};


lczero::Move GetMoveWithExploration(const std::vector<MoveEstimation>& moves_estimation, int env_ply, bool exploration_full_random, int exploration_num_first_moves);


}  // namespace probs
