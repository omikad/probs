#include <iostream>

#include "infra/battle.h"
#include "infra/config_parser.h"
#include "infra/agent.h"
#include "utils/exception.h"
#include "utils/callbacks.h"
#include "chess/board.h"
#include "chess/position.h"

using namespace std;


namespace probs {


void Battle::go_battle(const ConfigParser& config_parser) {
    IAgent* player1;
    IAgent* player2;

    if (config_parser.get_string("player1.kind") == "random")
        player1 = new RandomAgent();
    else
        throw Exception("Unknown player1.kind attribute value");

    if (config_parser.get_string("player2.kind") == "random")
        player2 = new RandomAgent();
    else
        throw Exception("Unknown player2.kind attribute value");

    int evaluate_n_games = config_parser.get_int("infra.evaluate_n_games");
    int n_max_episode_steps = config_parser.get_int("env.n_max_episode_steps");

    BattleInfo battleInfo;

    for (int gi = 0; gi < evaluate_n_games; gi++) {

        lczero::PositionHistory positionHistory;

        for (int si = 0; si < n_max_episode_steps; si++) {

        }
    }
}

}    // namespace probs