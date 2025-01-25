#pragma once

#include <iostream>
#include <memory>

#include "chess/game_tree.h"
#include "chess/position.h"
#include "chess/board.h"
#include "infra/config_parser.h"
#include "infra/player.h"
#include "utils/exception.h"
#include "utils/callbacks.h"
#include "training/training_helpers.h"


namespace probs {

BattleInfo ComparePlayers(IPlayer& player1, IPlayer& player2, int evaluate_n_games, int n_max_episode_steps, int random_first_turns = 0);

std::unique_ptr<IPlayer> CreatePlayer(const ConfigParser& config_parser, const std::string& player_config_path);

void GoBattle(const ConfigParser &config_parser);

}  // namespace probs