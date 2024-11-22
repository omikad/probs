#pragma once

#include "utils/callbacks.h"
#include "chess/bitboard.h"
#include "chess/board.h"
#include "infra/config_parser.h"


namespace probs {

class Battle {
    public:
        static void GoBattle(const ConfigParser &config_parser);
};

}  // namespace probs