#pragma once

#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <optional>
#include <chrono>

#include "infra/config_parser.h"
#include "infra/uci_player.h"
#include "chess/board.h"


namespace probs {

class UciImpl {
    public:
        UciImpl(ConfigParser& config);
        void run();

    private:
        UciPlayer uci_player;
        void onPositionCommand(std::stringstream& line_stream);
        void onGoCommand(std::stringstream& line_stream);
};

}  // namespace probs
