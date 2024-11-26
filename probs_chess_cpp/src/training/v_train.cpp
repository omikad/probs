#include "training/v_train.h"

using namespace std;


namespace probs {

vector<pair<lczero::InputPlanes, float>> SelfPlay(const ConfigParser& config_parser, const int n_games) {
    vector<pair<lczero::InputPlanes, float>> rows;

    ResNet q_model(config_parser, "model.q");
    at::Device device(torch::kCPU);                       // TODO: try gpu in threads?

    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps");
    string starting_fen = lczero::ChessBoard::kStartposFen;

    for (int gi = 0; gi < n_games; gi++) {
        int start_rows_index = rows.size();
        EnvPlayer env_player(starting_fen, n_max_episode_steps);

        bool is_first_black = env_player.LastPosition().IsBlackToMove();

        while (true) {
            vector<PositionHistoryTree*> trees = {&env_player.Tree()};
            auto encoded_batch = GetQModelEstimation(trees, {env_player.Tree().LastIndex()}, q_model, device);

            rows.push_back({encoded_batch->planes[0], 0});

            if (env_player.GameResult() != lczero::GameResult::UNDECIDED)
                break;

            auto move = encoded_batch->FindBestMoves()[0];
            env_player.Move(move);
        }

        auto game_result = env_player.GameResult();
        assert(game_result != lczero::GameResult::UNDECIDED);
        float score =
            game_result == lczero::GameResult::DRAW ? 0
            : is_first_black == (game_result == lczero::GameResult::BLACK_WON) ? 1
            : -1;
        for (int ri = start_rows_index; ri < rows.size(); ri++) {
            rows[ri].second = score;
            score = -score;
        }
    }

    return rows;
}

}  // namespace probs