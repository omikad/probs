#include "infra/battle.h"

using namespace std;


namespace probs {


BattleInfo ComparePlayers(IPlayer& player1, IPlayer& player2, int evaluate_n_games, int n_max_episode_steps, int random_first_turns) {
    torch::NoGradGuard no_grad;

    RandomPlayer random_player = RandomPlayer("RandomPlayer");

    string starting_fen = lczero::ChessBoard::kStartposFen;

    BattleInfo battle_info;

    vector<PositionHistoryTree*> trees;
    vector<bool> alive;
    for (int gi = 0; gi < evaluate_n_games; gi++) {
        PositionHistoryTree* tree = new PositionHistoryTree(starting_fen, n_max_episode_steps);
        trees.push_back(tree);
        alive.push_back(true);
    }

    int alive_cnt = trees.size();
    for (int step_i = 0; step_i < n_max_episode_steps; step_i++) {
        if (alive_cnt == 0)
            break;

        vector<PositionHistoryTree*> trees1;
        vector<PositionHistoryTree*> trees2;
        for (int i = 0; i < trees.size(); i++)
            if (alive[i])
                (i % 2 == step_i % 2 ? trees1 : trees2).push_back(trees[i]);

        vector<lczero::Move> moves1 = step_i < random_first_turns ? random_player.GetActions(trees1) : player1.GetActions(trees1);
        vector<lczero::Move> moves2 = step_i < random_first_turns ? random_player.GetActions(trees2) : player2.GetActions(trees2);

        for (int i = 0; i < trees1.size(); i++) trees1[i]->Move(-1, moves1[i]);
        for (int i = 0; i < trees2.size(); i++) trees2[i]->Move(-1, moves2[i]);

        for (int i = 0; i < trees.size(); i++) {
            auto game_result = trees[i]->GetGameResult(-1);
            if (alive[i] && game_result != lczero::GameResult::UNDECIDED) {
                alive[i] = false;
                alive_cnt--;

                if (i % 2 == 0) {   // player1 is white:
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
        }
    }

    for (int i = 0; i < trees.size(); i++)
        delete trees[i];

    return battle_info;
}


void GoBattle(const ConfigParser& config_parser) {
    torch::NoGradGuard no_grad;

    int random_first_turns = config_parser.GetInt("battle.random_first_turns", false, 0);

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
            auto device = GetDeviceFromConfig(config_parser);
            cout << "Device " << device << endl;
            model_keeper.To(device);
            model_keeper.SetEvalMode();

            if (kind == "v_resnet_player")
                (pi == 1 ? player1 : player2) = new VResnetPlayer(model_keeper.v_model, device, "VResnetPlayer" + to_string(pi));
            else if (kind == "q_resnet_player")
                (pi == 1 ? player1 : player2) = new QResnetPlayer(model_keeper.q_model, device, "QResnetPlayer" + to_string(pi));
        }
        else {
            throw Exception("Unknown player.kind attribute value");
        }
    }

    int evaluate_n_games = config_parser.GetInt("infra.evaluate_n_games", true, 0);
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps", true, 0);

    BattleInfo battle_info = ComparePlayers(*player1, *player2, evaluate_n_games, n_max_episode_steps, random_first_turns);

    cout << "Battle " << player1->GetName() << " vs " << player2->GetName() << ":" << endl;
    cout << "Number of random first turns: " << random_first_turns << endl;
    cout << "Games played: " << battle_info.games_played << endl;
    cout << "Player " << player1->GetName() << " stats [white,black]:" << endl;
    cout << "   wins: " << battle_info.results[0][0] << ", " << battle_info.results[0][1] << endl;
    cout << "  draws: " << battle_info.results[1][0] << ", " << battle_info.results[1][1] << endl;
    cout << " losses: " << battle_info.results[2][0] << ", " << battle_info.results[2][1] << endl;

    int w1 = battle_info.results[0][0] + battle_info.results[0][1];
    int d1 = battle_info.results[1][0] + battle_info.results[1][1];
    int l1 = battle_info.results[2][0] + battle_info.results[2][1];
    double games = w1 + d1 + l1;
    double player1_score = (w1 + (double)d1 / 2) / games;

    double w2 = games - w1 - d1;
    double player2_score = (w2 + (double)d1 / 2) / games;
    cout << "Player " << player1->GetName() << " score=" << player1_score << endl;
    cout << "Player " << player2->GetName() << " score=" << player2_score << endl;

    delete player1;
    delete player2;
}

}    // namespace probs