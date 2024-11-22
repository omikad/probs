#include <iostream>

#include "infra/battle.h"
#include "infra/config_parser.h"
#include "infra/player.h"
#include "utils/exception.h"
#include "utils/callbacks.h"
#include "chess/board.h"
#include "chess/position.h"
#include "infra/env_player.h"

using namespace std;


namespace probs {


void Battle::GoBattle(const ConfigParser& config_parser) {
    IPlayer* player1;
    IPlayer* player2;

    for (int pi = 1; pi <= 2; pi++) {
        string kind = config_parser.GetString("player" + to_string(pi) + ".kind");
        if (kind == "random")
            (pi == 1 ? player1 : player2) = new RandomPlayer("RandomPlayer" + to_string(pi));
        else if (kind == "vq_resnet_player")
            (pi == 1 ? player1 : player2) = new VQResnetPlayer(config_parser, "player" + to_string(pi), "VQResnetPlayer" + to_string(pi));
        else
            throw Exception("Unknown player1.kind attribute value");
    }

    int evaluate_n_games = config_parser.GetInt("infra.evaluate_n_games");
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps");
    string starting_fen = lczero::ChessBoard::kStartposFen;

    BattleInfo battle_info;

    for (int gi = 0; gi < evaluate_n_games; gi++) {
        EnvPlayer env_player(starting_fen, n_max_episode_steps);

        int start_player_shift = env_player.History().IsBlackToMove() ? 1 : 0;

        while (env_player.GameResult() == lczero::GameResult::UNDECIDED) {
            int ply = env_player.Ply();
            auto history = env_player.History();

            // cout << "Board at step " << ply << ":\n" << curr_board.DebugString() << endl;

            auto move = ((ply + gi) % 2 == 0 ? player1 : player2)->GetActions({history})[0];
            
            // cout << "Player " << ((ply + gi) % 2 == 0 ? player1 : player2)->GetName() << " selected move " << move.as_string() << endl;

            env_player.Move(move);
        }

        auto game_result = env_player.GameResult();
        if ((gi + start_player_shift) % 2 == 0) {   // player1 is white:
            if (game_result == lczero::GameResult::WHITE_WON)
                battle_info.results[0][0]++;
            else if (game_result == lczero::GameResult::BLACK_WON)
                battle_info.results[2][0]++;
            else
                battle_info.results[1][0]++;
        }
        else {   // player 1 is black
            if (game_result == lczero::GameResult::WHITE_WON)
                battle_info.results[2][1]++;
            else if (game_result == lczero::GameResult::BLACK_WON)
                battle_info.results[0][1]++;
            else
                battle_info.results[1][1]++;
        }
        battle_info.games_played++;
    }

    cout << "Battle " << player1->GetName() << " vs " << player2->GetName() << ":" << endl;
    cout << "Games played: " << battle_info.games_played << endl;
    cout << "Player [" << player1->GetName() << "]: as white,black:" << endl;
    cout << "   wins: " << battle_info.results[0][0] << ", " << battle_info.results[0][1] << endl;
    cout << "  draws: " << battle_info.results[1][0] << ", " << battle_info.results[1][1] << endl;
    cout << " losses: " << battle_info.results[2][0] << ", " << battle_info.results[2][1] << endl;
}

}    // namespace probs