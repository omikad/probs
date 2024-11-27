#include "training/v_train.h"

using namespace std;


namespace probs {

vector<pair<torch::Tensor, float>> SelfPlay(ResNet q_model, const ConfigParser& config_parser, const int n_games, at::Device& device) {
    torch::NoGradGuard no_grad;

    // TODO: check if tensor being copied

    int batch_size = config_parser.GetInt("training.batch_size");
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps");
    double dataset_drop_ratio = config_parser.GetDouble("training.dataset_drop_ratio");

    int game_idx = 0;
    vector<float> game_scores(n_games, 0);

    vector<pair<torch::Tensor, float>> rows;
    vector<int> row_game_indices;

    vector<shared_ptr<EnvPlayer>> envs;
    vector<int> game_indices;

    while (game_idx < n_games || envs.size() > 0) {

        if (envs.size() < batch_size && game_idx < n_games) {
            envs.push_back(make_shared<EnvPlayer>(EnvPlayer(lczero::ChessBoard::kStartposFen, n_max_episode_steps)));
            game_indices.push_back(game_idx);
            game_idx++;
        }

        else {
            vector<PositionHistoryTree*> trees;
            vector<int> nodes;
            for (int ei = 0; ei < envs.size(); ei++) {
                trees.push_back(&envs[ei]->Tree());
                nodes.push_back(envs[ei]->Tree().LastIndex());
            }
            auto encoded_batch = GetQModelEstimation(trees, nodes, q_model, device);

            auto best_moves = encoded_batch->FindBestMoves();

            for (int ei = envs.size() - 1; ei >= 0; ei--) {

                if (rand() % 1000000 > dataset_drop_ratio * 1000000) {
                    rows.push_back({encoded_batch->tensor[ei], 0});   // TODO: check if copy needed
                    row_game_indices.push_back(game_indices[ei]);
                }

                auto game_result = envs[ei]->GameResult();

                if (game_result == lczero::GameResult::UNDECIDED) {
                    auto move = best_moves[ei];
                    envs[ei]->Move(move);
                }
                else {
                    bool is_first_black = envs[ei]->Tree().positions[0].IsBlackToMove();
                    float score =
                        game_result == lczero::GameResult::DRAW ? 0
                        : is_first_black == (game_result == lczero::GameResult::BLACK_WON) ? 1
                        : -1;
                    game_scores[game_indices[ei]] = score;

                    if (ei < envs.size() - 1) {
                        swap(envs[ei], envs[envs.size() - 1]);
                        swap(game_indices[ei], game_indices[game_indices.size() - 1]);
                    }
                    envs.pop_back();
                    game_indices.pop_back();
                }
            }
        }
    }

    for (int ri = 0; ri < rows.size(); ri++)
        rows[ri].second = game_scores[row_game_indices[ri]];


    // for (int gi = 0; gi < n_games; gi++) {
    //     int start_rows_index = rows.size();
    //     EnvPlayer env_player(starting_fen, n_max_episode_steps);

    //     bool is_first_black = env_player.LastPosition().IsBlackToMove();

    //     while (true) {
    //         vector<PositionHistoryTree*> trees = {&env_player.Tree()};
    //         auto encoded_batch = GetQModelEstimation(trees, {env_player.Tree().LastIndex()}, q_model, device);

    //         if (rand() % 1000000 > dataset_drop_ratio * 1000000)
    //             rows.push_back({encoded_batch->tensor, 0});

    //         if (env_player.GameResult() != lczero::GameResult::UNDECIDED)
    //             break;

    //         auto move = encoded_batch->FindBestMoves()[0];
    //         env_player.Move(move);
    //     }

    //     auto game_result = env_player.GameResult();
    //     assert(game_result != lczero::GameResult::UNDECIDED);
    //     float score =
    //         game_result == lczero::GameResult::DRAW ? 0
    //         : is_first_black == (game_result == lczero::GameResult::BLACK_WON) ? 1
    //         : -1;
    //     for (int ri = start_rows_index; ri < rows.size(); ri++) {
    //         rows[ri].second = score;
    //         score = -score;
    //     }
    // }

    return rows;
}


void TrainV(const ConfigParser& config_parser, ResNet v_model, torch::optim::AdamW& v_optimizer, vector<pair<torch::Tensor, float>>& v_dataset) {
    int dataset_size = v_dataset.size();
    cout << "[Train.V] Train V model on dataset with " << dataset_size << " rows" << endl;

    int batch_size = config_parser.GetInt("training.batch_size");

    vector<int> indices(dataset_size);
    for (int i = 0; i < dataset_size; i++) indices[i] = i;
    for (int i = 0; i < dataset_size; i++) swap(indices[i], indices[i + rand() % (dataset_size - i)]);

    for (int end = batch_size; end <= dataset_size; end += batch_size) {
        vector<torch::Tensor> input_arr;
        torch::Tensor target = torch::zeros({batch_size, 1});

        for (int i = end - batch_size; i < end; i++) {
            input_arr.push_back(v_dataset[indices[i]].first);
            target[i - end + batch_size, 0] = v_dataset[indices[i]].second;
        }
        torch::Tensor input = torch::cat(input_arr);

        v_optimizer.zero_grad();

        // cout << "Input: " << DebugString(input) << endl;

        torch::Tensor prediction = v_model->forward(input);

        // cout << "Prediction: " << DebugString(prediction) << endl;

        torch::Tensor loss = torch::mse_loss(prediction, target);

        // cout << "Loss: " << DebugString(loss) << endl;

        loss.backward();

        v_optimizer.step();

        cout << "Loss: " << loss.item<float>() << endl;
    }
}


}  // namespace probs