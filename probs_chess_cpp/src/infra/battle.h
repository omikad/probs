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

class Battle {
    public:
        static void GoBattle(const ConfigParser &config_parser);
};

}  // namespace probs