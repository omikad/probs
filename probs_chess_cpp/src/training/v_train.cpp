#include "training/v_train.h"

using namespace std;


namespace probs {

vector<pair<torch::Tensor, float>> SelfPlay(ResNet q_model, const ConfigParser& config_parser, const int n_games) {
    torch::NoGradGuard no_grad;

    vector<pair<torch::Tensor, float>> rows;

    // TODO: check if tensor being copied

    at::Device device(torch::kCPU);                       // TODO: try gpu in threads?

    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps");
    double dataset_drop_ratio = config_parser.GetDouble("training.dataset_drop_ratio");

    string starting_fen = lczero::ChessBoard::kStartposFen;

    for (int gi = 0; gi < n_games; gi++) {
        int start_rows_index = rows.size();
        EnvPlayer env_player(starting_fen, n_max_episode_steps);

        bool is_first_black = env_player.LastPosition().IsBlackToMove();

        while (true) {
            vector<PositionHistoryTree*> trees = {&env_player.Tree()};
            auto encoded_batch = GetQModelEstimation(trees, {env_player.Tree().LastIndex()}, q_model, device);

            if (rand() % 1000000 > dataset_drop_ratio * 1000000)
                rows.push_back({encoded_batch->tensor, 0});

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