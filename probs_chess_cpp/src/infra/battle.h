#pragma once

#include <iostream>

#include "infra/config_parser.h"
#include "infra/player.h"
#include "utils/exception.h"
#include "utils/callbacks.h"
#include "chess/board.h"
#include "chess/position.h"
#include "infra/env_player.h"


namespace probs {

class Battle {
    public:
        static void GoBattle(const ConfigParser &config_parser);
};

}  // namespace probs