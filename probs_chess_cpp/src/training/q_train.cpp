#include "training/q_train.h"

using namespace std;


namespace probs {


struct KidEstimation {
    int node;
    lczero::Move move;
    float score;
};


struct EnvExpandState {
    shared_ptr<EnvPlayer> env;
    int top_node;
    map<pair<float, int>, int> beam;        // (priority, node) -> node
    vector<vector<KidEstimation>> kid_moves_estimation;          // node -> [{kid, move, q score of this kid}]
    vector<int> node_depths;   // depth is counted from the node with the last selection action
    int expand_tree_size;
    int n_qsa_calls;

    EnvExpandState(string start_pos, int n_max_episode_steps) {
        env = make_shared<EnvPlayer>(EnvPlayer(lczero::ChessBoard::kStartposFen, n_max_episode_steps));

        // Root node:
        top_node = 0;
        beam.insert({{1000.0, 0}, 0});
        kid_moves_estimation.push_back({});
        node_depths.push_back(0);
        expand_tree_size = 1;
        n_qsa_calls = 0;
    }

    int GetTopPriorityNode() const {
        return beam.rbegin()->second;
    }

    void AppendChildren(const int node, const vector<pair<lczero::Move, float>>& moves_estimation, const int& max_depth) {
        int node_depth = node_depths[node];

        for (int mi = 0; mi < moves_estimation.size(); mi++) {
            int kid_node = env->Tree().Append(node, moves_estimation[mi].first);

            if (node_depth + 1 < max_depth) {
                // float kid_priority = node_depth == 0 ? 1000.0 : moves_estimation[mi].second;    // always expand first turn
                float kid_priority = moves_estimation[mi].second;
                beam.insert({{kid_priority, kid_node}, kid_node});
            }

            assert(kid_node == kid_moves_estimation.size());
            kid_moves_estimation.push_back({});
            kid_moves_estimation[node].push_back({kid_node, moves_estimation[mi].first, moves_estimation[mi].second});

            assert(kid_node == node_depths.size());
            node_depths.push_back(node_depth + 1);

            expand_tree_size++;
        }
    }

    void RecomputeMovesEstimation(const PositionHistoryTree& tree, ResNet v_model, at::Device device, int batch_size) {
        vector<int> nodes;
        vector<lczero::InputPlanes> leaf_input_planes;
        for (int node = 0; node < kid_moves_estimation.size(); node++)
            if (kid_moves_estimation[node].size() == 0) {
                nodes.push_back(node);
                int transform_out;
                leaf_input_planes.push_back(Encode(tree.ToLczeroHistory(node), &transform_out));
            }

        map<int, float> leaf_values;
        for (int start = 0; start < nodes.size(); start += batch_size) {
            int end = min((int)nodes.size(), start + batch_size);

            torch::Tensor input = torch::zeros({end - start, lczero::kInputPlanes, 8, 8});
            for (int bi = 0; bi < end - start; bi++)
                for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
                    const auto& plane = leaf_input_planes[start + bi][pi];
                    for (auto bit : lczero::IterateBits(plane.mask)) {
                        input[bi][pi][bit / 8][bit % 8] = plane.value;
                    }
                }
            input = input.to(device);
            torch::Tensor predictions = v_model->forward(input);
            predictions = predictions.contiguous();
            predictions = predictions.to(torch::kCPU);
            vector<float> pred_values(predictions.data_ptr<float>(), predictions.data_ptr<float>() + predictions.numel());
            assert(pred_values.size() == end - start);

            for (int bi = 0; bi < end - start; bi++)
                leaf_values[nodes[bi + start]] = pred_values[bi];
        }

        auto dfs = [&](auto& self, int node)->float {
            if (kid_moves_estimation[node].size() == 0)
                return leaf_values[node];
            float curr_val = -1000.0;
            for (int ki = 0; ki < kid_moves_estimation[node].size(); ki++) {
                float kid_val = self(self, kid_moves_estimation[node][ki].node);
                kid_moves_estimation[node][ki].score = kid_val;
                curr_val = max(curr_val, -kid_val);
            }
            return curr_val;
        };
        dfs(dfs, 0);
    }

    void ShowDebugStructure() {
        cout << "EnvExpandState: beam size=" << beam.size()
             << "; node depths size=" << node_depths.size()
             << "; expand_tree_size=" << expand_tree_size
             << "; n_qsa_calls=" << n_qsa_calls << endl;

        cout << "Beam:" << endl;
        for (auto& kvp : beam)
            cout << "  Node=" << kvp.second << "; priority=" << kvp.first.first << endl;

        cout << "Tree:" << endl;
        vector<int> queue{top_node};
        for (int qi = 0; qi < queue.size(); qi++) {
            int node = queue[qi];
            cout << "  Node=" << node << "; depth=" << node_depths[node] << "; kid_moves_estimation=" << endl;
            for(auto& kms : kid_moves_estimation[node])
                cout << "     Kid=" << kms.node << "; move=" << kms.move.as_string() << "; " << kms.score << endl;

            for(auto& kms : kid_moves_estimation[node])
                queue.push_back(kms.node);
        }
    }
};


QDataset GetQDataset(ResNet v_model, ResNet q_model, at::Device& device, const ConfigParser& config_parser, const int n_games) {
    torch::NoGradGuard no_grad;
    v_model->eval();
    q_model->eval();

    int batch_size = config_parser.GetInt("training.batch_size");
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps");
    double dataset_drop_ratio = config_parser.GetDouble("training.dataset_drop_ratio");
    int exploration_num_first_moves = config_parser.GetInt("training.exploration_num_first_moves");
    bool exploration_full_random = config_parser.KeyExist("training.exploration_full_random");
    int tree_num_q_s_a_calls = config_parser.GetInt("training.tree_num_q_s_a_calls");
    int tree_max_depth = config_parser.GetInt("training.tree_max_depth");

    int game_idx = 0;
    QDataset rows;

    vector<EnvExpandState> envs;

    while (game_idx < n_games || envs.size() > 0) {

        if (envs.size() < batch_size && game_idx < n_games) {
            envs.push_back(EnvExpandState(lczero::ChessBoard::kStartposFen, n_max_episode_steps));
            game_idx++;
        }

        else {
            cout << "-------------------------------------------" << endl;
            cout << "envs size=" << envs.size() << endl;
            for (int ei = 0; ei < envs.size(); ei++)
                envs[ei].ShowDebugStructure();
            // for (int ei = 0; ei < envs.size(); ei++) envs[ei].ShowDebugStructure();

            vector<PositionHistoryTree*> trees;
            vector<int> nodes;
            for (int ei = 0; ei < envs.size(); ei++) {
                trees.push_back(&envs[ei].env->Tree());
                nodes.push_back(envs[ei].GetTopPriorityNode());
                envs[ei].beam.erase(std::prev(envs[ei].beam.end()));
            }
            auto encoded_batch = GetQModelEstimation(trees, nodes, q_model, device);

            // for (int ei = 0; ei < envs.size(); ei++) {
            //     cout << "ei=" << ei << "; moves size=" << encoded_batch->moves_estimation[ei].size() << endl;
            // }
            // cout << "Go append children" << endl;

            for (int ei = 0; ei < envs.size(); ei++)
                envs[ei].AppendChildren(nodes[ei], encoded_batch->moves_estimation[ei], tree_max_depth);

            // for (int ei = 0; ei < envs.size(); ei++) envs[ei].ShowDebugStructure();

            for (int ei = envs.size() - 1; ei >= 0; ei--) {
                envs[ei].n_qsa_calls++;
                
                if (envs[ei].n_qsa_calls < tree_num_q_s_a_calls)    // need expand more
                    continue;

                envs[ei].ShowDebugStructure();

                envs[ei].RecomputeMovesEstimation(*trees[ei], v_model, device, batch_size);

                cout << "------------------- RECOMPUTED ------------------------" << endl;
                envs[ei].ShowDebugStructure();

                throw Exception("hi");

            //     if (rand() % 1000000 > dataset_drop_ratio * 1000000) {
            //         float is_row_black = envs[ei]->Tree().Last().IsBlackToMove() ? 1 : -1;
            //         rows.push_back({encoded_batch->planes[ei], is_row_black});
            //         env_rows[ei].push_back(rows.size());
            //     }

            //     auto game_result = envs[ei]->GameResult();

            //     if (game_result == lczero::GameResult::UNDECIDED) {
            //         auto move = GetMoveWithExploration(encoded_batch, ei, envs[ei]->LastPosition().GetGamePly(), exploration_full_random, exploration_num_first_moves);
            //         envs[ei]->Move(move);
            //     }
            //     else {
            //         bool is_first_black = envs[ei]->Tree().positions[0].IsBlackToMove();
            //         float first_player_score =
            //             game_result == lczero::GameResult::DRAW ? 0
            //             : is_first_black == (game_result == lczero::GameResult::BLACK_WON) ? 1
            //             : -1;

            //         for (int row_idx : env_rows[ei]) {
            //             float is_row_black = rows[row_idx].second;
            //             float score = (is_first_black ? 1 : -1) * is_row_black * first_player_score;
            //             rows[row_idx].second = score;
            //         }

            //         if (ei < envs.size() - 1) {
            //             swap(envs[ei], envs[envs.size() - 1]);
            //             swap(env_rows[ei], env_rows[env_rows.size() - 1]);
            //         }
            //         envs.pop_back();
            //         env_rows.pop_back();
            //     }
            }
        }
    }


    return rows;
}


}  // namespace probs
