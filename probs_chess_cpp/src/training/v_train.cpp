#include "training/v_train.h"

using namespace std;


namespace probs {


VDataset SelfPlay(ResNet q_model, at::Device& device, const ConfigParser& config_parser, const int n_games) {
    torch::NoGradGuard no_grad;
    q_model->eval();

    int batch_size = config_parser.GetInt("training.batch_size", true, 0);
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps", true, 0);
    double dataset_drop_ratio = config_parser.GetDouble("training.dataset_drop_ratio", true, 0);
    int exploration_num_first_moves = config_parser.GetInt("training.exploration_num_first_moves", true, 0);
    bool exploration_full_random = config_parser.GetInt("training.exploration_full_random", false, 0) > 0;
    bool is_test = config_parser.GetInt("training.is_test", false, 0) > 0;

    // cout << "[SELFPLAY] batch_size = " << batch_size << endl;
    // cout << "[SELFPLAY] n_max_episode_steps = " << n_max_episode_steps << endl;
    // cout << "[SELFPLAY] dataset_drop_ratio = " << dataset_drop_ratio << endl;
    // cout << "[SELFPLAY] exploration_num_first_moves = " << exploration_num_first_moves << endl;
    // cout << "[SELFPLAY] exploration_full_random = " << (exploration_full_random ? "true" : "false") << endl;

    int game_idx = 0;
    VDataset rows;

    vector<PositionHistoryTree*> trees;
    vector<vector<pair<int, int>>> tree_rows;  // env_index -> [(row_idx, node)]

    while (game_idx < n_games || trees.size() > 0) {

        if (trees.size() < batch_size && game_idx < n_games) {
            PositionHistoryTree* tree = new PositionHistoryTree(lczero::ChessBoard::kStartposFen, n_max_episode_steps);
            trees.push_back(tree);
            tree_rows.push_back({});
            game_idx++;
        }

        else {
            vector<int> nodes;
            for (int ei = 0; ei < trees.size(); ei++)
                nodes.push_back(trees[ei]->LastIndex());

            auto encoded_batch = GetQModelEstimation(trees, nodes, q_model, device);

            for (int ei = trees.size() - 1; ei >= 0; ei--) {

                if (rand() % 1000000 >= dataset_drop_ratio * 1000000) {
                    float is_row_black = trees[ei]->LastPosition().IsBlackToMove() ? 1 : -1;
                    tree_rows[ei].push_back({ rows.size(), trees[ei]->LastIndex() });
                    rows.push_back({encoded_batch->planes[ei], is_row_black});
                }

                auto game_result = trees[ei]->GetGameResult(-1);

                if (game_result == lczero::GameResult::UNDECIDED) {
                    auto move = GetMoveWithExploration(encoded_batch->moves_estimation[ei], trees[ei]->LastPosition().GetGamePly(), exploration_full_random, exploration_num_first_moves);
                    trees[ei]->Move(-1, move);
                }
                else {
                    float black_score =
                        game_result == lczero::GameResult::DRAW ? 0
                        : game_result == lczero::GameResult::BLACK_WON ? 1
                        : -1;

                    for (auto [row_idx, node] : tree_rows[ei]) {
                        float is_row_black = rows[row_idx].second;
                        rows[row_idx].second = is_row_black * black_score;
                    }

                    // cout << "env outcome=" << (int)game_result << endl;
                    // cout << "env rows.size()=" << tree_rows[ei].size() << endl;
                    // cout << "env size=" << trees[ei]->positions.size() << endl;
                    // cout << "env rows scores:";
                    // for (int ri : tree_rows[ei])
                    //     cout << " " << rows[ri].second;
                    // cout << endl;

                    if (is_test) {
                        int game_plys = trees[ei]->LastPosition().GetGamePly();
                        assert(trees[ei]->positions.size() == game_plys + 1);

                        for (int start_node = 0; start_node < trees[ei]->positions.size(); start_node++) {
                            assert(trees[ei]->positions[start_node].GetGamePly() == start_node);
                            assert(trees[ei]->parents[start_node] == start_node - 1);                // should be chain-like
                        }

                        for (auto [row_idx, node] : tree_rows[ei]) {
                            float expected_score = (trees[ei]->positions[node].IsBlackToMove() ? 1 : -1) * black_score;
                            float actual_score = rows[row_idx].second;
                            assert(abs(expected_score - actual_score) < 1e-5);
                        }
                    }

                    if (ei < trees.size() - 1) {
                        swap(trees[ei], trees[trees.size() - 1]);
                        swap(tree_rows[ei], tree_rows[tree_rows.size() - 1]);
                    }

                    delete trees.back();
                    trees.pop_back();
                    tree_rows.pop_back();
                }
            }
        }
    }
    assert(trees.size() == 0);
    if (is_test)
        cout << "V train ok" << endl;

    return rows;
}


void TrainV(const ConfigParser& config_parser, ofstream& losses_file, ResNet v_model, at::Device& device, torch::optim::AdamW& v_optimizer, VDataset& v_dataset) {
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
        cout << "zeros=" << (double)counter[0] / v_dataset.size() << endl;
    }

    int batch_size = config_parser.GetInt("training.batch_size", true, 0);

    vector<int> indices(dataset_size);
    for (int i = 0; i < dataset_size; i++) indices[i] = i;
    for (int i = 0; i < dataset_size; i++) swap(indices[i], indices[i + rand() % (dataset_size - i)]);

    for (int end = batch_size; end <= dataset_size; end += batch_size) {
        torch::Tensor target = torch::zeros({batch_size, 1});
        torch::Tensor input = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});

        for (int ri = end - batch_size; ri < end; ri++) {
            int bi = ri - end + batch_size;
            int row_i = indices[ri];

            target[bi][0] = v_dataset[row_i].second;

            for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
                const lczero::InputPlane& plane = v_dataset[row_i].first[pi];
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

        losses_file << "VLoss: " << loss.item<float>() << endl;
    }
}


}  // namespace probs