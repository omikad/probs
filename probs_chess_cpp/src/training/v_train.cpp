#include "training/v_train.h"

using namespace std;


namespace probs {


lczero::Move GetMoveWithExploration(shared_ptr<EncodedPositionBatch> encoded_batch, int batch_item_idx, int env_ply, bool exploration_full_random, int exploration_num_first_moves) {
    if (exploration_full_random) {
        int pi = rand() % (encoded_batch->moves_estimation[batch_item_idx].size());
        return encoded_batch->moves_estimation[batch_item_idx][pi].first;
    }

    if (exploration_num_first_moves > 0) {
        vector<pair<lczero::Move, float>>& moves_estimation = encoded_batch->moves_estimation[batch_item_idx];

        double max_score = 0;
        vector<double> scores;
        for (auto& move_and_score : moves_estimation) {
            double score = move_and_score.second;
            scores.push_back(score);
            max_score = max(max_score, score);
        }

        for (int i = 0; i < scores.size(); i++)
            scores[i] = exp(scores[i] - max_score);
        
        double summ = 0;
        for (int i = 0; i < scores.size(); i++)
            summ += scores[i];

        for (int i = 0; i < scores.size(); i++)
            scores[i] /= summ;

        double p = ((double)(rand() % 1000000) / 1000000);
        int idx = 0;
        for (; idx < scores.size(); idx++) {
            p -= scores[idx];
            if (p < 0)
                break;
        }

        return moves_estimation[idx].first;
    }

    return encoded_batch->FindBestMove(batch_item_idx);
}


vector<pair<lczero::InputPlanes, float>> SelfPlay(ResNet q_model, at::Device& device, const ConfigParser& config_parser, const int n_games) {
    torch::NoGradGuard no_grad;
    q_model->eval();

    int batch_size = config_parser.GetInt("training.batch_size");
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps");
    double dataset_drop_ratio = config_parser.GetDouble("training.dataset_drop_ratio");
    int exploration_num_first_moves = config_parser.GetInt("training.exploration_num_first_moves");
    bool exploration_full_random = config_parser.KeyExist("training.exploration_full_random");

    // cout << "[SELFPLAY] batch_size = " << batch_size << endl;
    // cout << "[SELFPLAY] n_max_episode_steps = " << n_max_episode_steps << endl;
    // cout << "[SELFPLAY] dataset_drop_ratio = " << dataset_drop_ratio << endl;
    // cout << "[SELFPLAY] exploration_num_first_moves = " << exploration_num_first_moves << endl;
    // cout << "[SELFPLAY] exploration_full_random = " << (exploration_full_random ? "true" : "false") << endl;


    int game_idx = 0;
    vector<float> game_scores(n_games, 0);

    vector<pair<lczero::InputPlanes, float>> rows;

    vector<shared_ptr<EnvPlayer>> envs;
    vector<vector<int>> env_rows;

    while (game_idx < n_games || envs.size() > 0) {

        if (envs.size() < batch_size && game_idx < n_games) {
            envs.push_back(make_shared<EnvPlayer>(EnvPlayer(lczero::ChessBoard::kStartposFen, n_max_episode_steps)));
            env_rows.push_back({});
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

            for (int ei = envs.size() - 1; ei >= 0; ei--) {

                if (rand() % 1000000 > dataset_drop_ratio * 1000000) {
                    float is_row_black = envs[ei]->Tree().Last().IsBlackToMove() ? 1 : -1;
                    rows.push_back({encoded_batch->planes[ei], is_row_black});
                    env_rows[ei].push_back(rows.size());
                }

                auto game_result = envs[ei]->GameResult();

                if (game_result == lczero::GameResult::UNDECIDED) {
                    auto move = GetMoveWithExploration(encoded_batch, ei, envs[ei]->LastPosition().GetGamePly(), exploration_full_random, exploration_num_first_moves);
                    envs[ei]->Move(move);
                }
                else {
                    bool is_first_black = envs[ei]->Tree().positions[0].IsBlackToMove();
                    float first_player_score =
                        game_result == lczero::GameResult::DRAW ? 0
                        : is_first_black == (game_result == lczero::GameResult::BLACK_WON) ? 1
                        : -1;

                    for (int row_idx : env_rows[ei]) {
                        float is_row_black = rows[row_idx].second;
                        float score = (is_first_black ? 1 : -1) * is_row_black * first_player_score;
                        rows[row_idx].second = score;
                    }

                    if (ei < envs.size() - 1) {
                        swap(envs[ei], envs[envs.size() - 1]);
                        swap(env_rows[ei], env_rows[env_rows.size() - 1]);
                    }
                    envs.pop_back();
                    env_rows.pop_back();
                }
            }
        }
    }

    return rows;
}


void TrainV(const ConfigParser& config_parser, ResNet v_model, at::Device& device, torch::optim::AdamW& v_optimizer, vector<pair<lczero::InputPlanes, float>>& v_dataset) {
    v_model->train();

    int dataset_size = v_dataset.size();
    map<float, int> counter;
    for (auto& row : v_dataset) counter[row.second]++;

    cout << "[Train.V] Train V model on dataset with " << dataset_size << " rows";
    if (counter.size() >= 10)
        cout << endl;
    else {
        cout << ", stats: ";
        for (auto kvp : counter) cout << "score=" << kvp.first << " count=" << kvp.second << "; ";
        cout << "nulls=" << (double)counter[0] / v_dataset.size() << endl;
    }

    int batch_size = config_parser.GetInt("training.batch_size");

    vector<int> indices(dataset_size);
    for (int i = 0; i < dataset_size; i++) indices[i] = i;
    for (int i = 0; i < dataset_size; i++) swap(indices[i], indices[i + rand() % (dataset_size - i)]);

    for (int end = batch_size; end <= dataset_size; end += batch_size) {

        torch::Tensor target = torch::zeros({batch_size, 1});
        torch::Tensor input = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});

        for (int ri = end - batch_size; ri < end; ri++) {
            int bi = ri - end + batch_size;

            target[bi][0] = v_dataset[ri].second;

            for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
                const lczero::InputPlane& plane = v_dataset[ri].first[pi];
                for (auto bit : lczero::IterateBits(plane.mask)) {
                    input[bi][pi][bit / 8][bit % 8] = plane.value;
                }
            }
        }

        input = input.to(device);
        target = target.to(device);

        v_optimizer.zero_grad();

        // cout << "Input: " << DebugString(input) << endl;

        torch::Tensor prediction = v_model->forward(input);

        // cout << "Prediction: " << DebugString(prediction.to(torch::kCPU)) << endl;

        torch::Tensor loss = torch::mse_loss(prediction, target);

        // cout << "Loss: " << DebugString(loss) << endl;

        loss.backward();

        v_optimizer.step();

        cout << "Loss: " << loss.item<float>() << endl;
    }
}


}  // namespace probs