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


void Battle::GoBattle(const ConfigParser& config_parser) {
    IAgent* player1;
    IAgent* player2;

    if (config_parser.GetString("player1.kind") == "random")
        player1 = new RandomAgent("Player1_random");
    else
        throw Exception("Unknown player1.kind attribute value");

    if (config_parser.GetString("player2.kind") == "random")
        player2 = new RandomAgent("Player2_random");
    else
        throw Exception("Unknown player2.kind attribute value");

    int evaluate_n_games = config_parser.GetInt("infra.evaluate_n_games");
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps");
    string starting_fen = lczero::ChessBoard::kStartposFen;

    BattleInfo battle_info;

    for (int gi = 0; gi < evaluate_n_games; gi++) {
        lczero::ChessBoard starting_board;
        int no_capture_ply;
        int full_moves;
        starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);

        lczero::PositionHistory position_history;
        position_history.Reset(starting_board, no_capture_ply, full_moves * 2 - (starting_board.flipped() ? 1 : 2));

        int start_player_shift = position_history.IsBlackToMove() ? 1 : 0;

        for (int si = 0; si <= n_max_episode_steps; si++) {
            auto game_result = position_history.ComputeGameResult();
            if (game_result != lczero::GameResult::UNDECIDED || si == n_max_episode_steps - 1) {
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
                battle_info.finished = gi == evaluate_n_games - 1;
                battle_info.games_played++;
                break;
            }

            auto curr_board = position_history.Last().GetBoard();
            // cout << "Board at step " << si << ":\n" << curr_board.DebugString() << endl;

            auto move = ((si + gi) % 2 == 0 ? player1 : player2)->GetActions({curr_board})[0];
            move = curr_board.GetModernMove(move);

            position_history.Append(move);
        }
    }

    cout << "Games played: " << battle_info.games_played << endl;
    cout << "Player [" << player1->GetName() << "]: as white,black:" << endl;
    cout << "   wins: " << battle_info.results[0][0] << ", " << battle_info.results[0][1] << endl;
    cout << "  draws: " << battle_info.results[1][0] << ", " << battle_info.results[1][1] << endl;
    cout << " losses: " << battle_info.results[2][0] << ", " << battle_info.results[2][1] << endl;
}

}    // namespace probs