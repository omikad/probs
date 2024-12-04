#pragma once

namespace probs {
struct BattleInfo {
    int games_played = 0;

    // Player1's [win/draw/lose] as [white/black].
    int results[3][2] = {{0, 0}, {0, 0}, {0, 0}};
};
}       // namespace probs
