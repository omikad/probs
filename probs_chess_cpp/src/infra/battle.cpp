#include "infra/battle.h"

using namespace std;


namespace probs {


BattleInfo ComparePlayers(IPlayer& player1, IPlayer& player2, int evaluate_n_games, int n_max_episode_steps) {
    string starting_fen = lczero::ChessBoard::kStartposFen;

    BattleInfo battle_info;

    for (int gi = 0; gi < evaluate_n_games; gi++) {
        PositionHistoryTree tree(starting_fen, n_max_episode_steps);

        int start_player_shift = tree.LastPosition().IsBlackToMove() ? 1 : 0;

        while (tree.GetGameResult(-1) == lczero::GameResult::UNDECIDED) {
            int ply = tree.LastPosition().GetGamePly();

            // cout << "Board at step " << ply << ":\n" << curr_board.DebugString() << endl;

            vector<PositionHistoryTree*> trees = { &tree };
            auto move = ((ply + gi) % 2 == 0 ? player1 : player2).GetActions(trees)[0];
            
            // cout << "PLY " << ply << " " << "Player " << ((ply + gi) % 2 == 0 ? player1 : player2)->GetName() << " selected move " << move.as_string() << " " << (int)env_player.GameResult() << endl;

            tree.Move(-1, move);
        }

        auto game_result = tree.GetGameResult(-1);
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

    return battle_info;
}


void GoBattle(const ConfigParser& config_parser) {
    torch::NoGradGuard no_grad;

    IPlayer* player1;
    IPlayer* player2;

    for (int pi = 1; pi <= 2; pi++) {
        string kind = config_parser.GetString("player" + to_string(pi) + ".kind");
        if (kind == "random")
            (pi == 1 ? player1 : player2) = new RandomPlayer("RandomPlayer" + to_string(pi));
        else if (kind == "one_step_lookahead")
            (pi == 1 ? player1 : player2) = new NStepLookaheadPlayer("NStepLookaheadPlayer" + to_string(pi), 1);
        else if (kind == "two_step_lookahead")
            (pi == 1 ? player1 : player2) = new NStepLookaheadPlayer("NStepLookaheadPlayer" + to_string(pi), 2);
        else if (kind == "three_step_lookahead")
            (pi == 1 ? player1 : player2) = new NStepLookaheadPlayer("NStepLookaheadPlayer" + to_string(pi), 3);
        else if (kind == "v_resnet_player" || kind == "q_resnet_player") {
            ModelKeeper model_keeper(config_parser, "player" + to_string(pi) + ".model");
            if (kind == "v_resnet_player")
                (pi == 1 ? player1 : player2) = new VResnetPlayer(model_keeper, config_parser, "player" + to_string(pi), "VResnetPlayer" + to_string(pi));
            else if (kind == "q_resnet_player")
                (pi == 1 ? player1 : player2) = new QResnetPlayer(model_keeper, config_parser, "player" + to_string(pi), "QResnetPlayer" + to_string(pi));
            model_keeper.SetEvalMode();
        }
        else {
            throw Exception("Unknown player.kind attribute value");
        }
    }

    int evaluate_n_games = config_parser.GetInt("infra.evaluate_n_games", true, 0);
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps", true, 0);

    BattleInfo battle_info = ComparePlayers(*player1, *player2, evaluate_n_games, n_max_episode_steps);

    cout << "Battle " << player1->GetName() << " vs " << player2->GetName() << ":" << endl;
    cout << "Games played: " << battle_info.games_played << endl;
    cout << "Player " << player1->GetName() << " stats [white,black]:" << endl;
    cout << "   wins: " << battle_info.results[0][0] << ", " << battle_info.results[0][1] << endl;
    cout << "  draws: " << battle_info.results[1][0] << ", " << battle_info.results[1][1] << endl;
    cout << " losses: " << battle_info.results[2][0] << ", " << battle_info.results[2][1] << endl;

    delete player1;
    delete player2;
}

}    // namespace probs