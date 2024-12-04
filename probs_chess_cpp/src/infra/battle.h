#pragma once

#include <iostream>

#include "chess/game_tree.h"
#include "chess/position.h"
#include "chess/board.h"
#include "infra/config_parser.h"
#include "infra/player.h"
#include "utils/exception.h"
#include "utils/callbacks.h"


namespace probs {

BattleInfo ComparePlayers(IPlayer& player1, IPlayer& player2, int evaluate_n_games, int n_max_episode_steps);

void GoBattle(const ConfigParser &config_parser);

}  // namespace probs