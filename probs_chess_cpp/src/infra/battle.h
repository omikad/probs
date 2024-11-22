#pragma once

#include "utils/callbacks.h"
#include "chess/bitboard.h"
#include "chess/board.h"
#include "infra/config_parser.h"


namespace probs {

class Battle {
    public:
        static void go_battle(const ConfigParser& config_parser);
};

}  // namespace probs