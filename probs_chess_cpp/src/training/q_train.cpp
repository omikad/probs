#include <optional>

#include "training/q_train.h"
#include "utils/usage_counter.h"

using namespace std;

namespace probs {


enum NodeState {
    CREATED,                    // node just created
    TERMINAL,                   // game ends here
    FRONTIER,                   // node is leaf of beam search
    FRONTIER_MOVES_COMPUTED,    // node is leaf of beam search and its legal moves are computed
    FRONTIER_Q_COMPUTED,        // node is leaf of beam search and its q values are estimated
    EXPANDED,                   // node is expanded
};


class QTrainNode {
    public:
        NodeState state;
        vector<int> kids;
        vector<MoveEstimation> moves_estimation;
        int depth;          // depth from top node
        optional<float> v_estimation;

        int transform;
        lczero::InputPlanes input_planes;


        QTrainNode(const PositionHistoryTree& history_tree, const int last_node, int depth) : state(NodeState::CREATED), depth(depth) {
            input_planes = Encode(history_tree, last_node, &transform);
        }

        QTrainNode(const PositionHistoryTree& history_tree, const vector<int>& history_nodes, int depth) : state(NodeState::CREATED), depth(depth) {
            input_planes = Encode(history_tree, history_nodes, &transform);
        }

        void AddValidMoves(const vector<lczero::Move>& valid_moves) {
            assert(state == NodeState::FRONTIER);
            assert(moves_estimation.size() == 0);
            for (auto move : valid_moves)
                moves_estimation.push_back({move, 0.0f});
            state = NodeState::FRONTIER_MOVES_COMPUTED;
        }

        int FindKidByMove(const lczero::Move move) const {
            assert(kids.size() == moves_estimation.size());
            for (int ki = 0; ki < kids.size(); ki++)
                if (moves_estimation[ki].move == move)
                    return kids[ki];
            return -1;
        }

        bool IsLeaf() const {
            return kids.size() == 0;
        }

        void ShowKidsInfo() const {
            assert(kids.size() == moves_estimation.size());
            for (int ki = 0; ki < kids.size(); ki++)
                cout << "     Kid=" << kids[ki] << "; move=" << moves_estimation[ki].move.as_string() << "; " << moves_estimation[ki].score << endl;
        }
};


struct EnvExpandState {
    PositionHistoryTree tree;
    int top_node;
    map<pair<float, int>, int> beam;        // (priority, node) -> node
    vector<QTrainNode*> nodes;
    int n_qsa_calls;
    bool node_going_to_dataset;

    EnvExpandState(string start_pos, int n_max_episode_steps)
            : tree(PositionHistoryTree(lczero::ChessBoard::kStartposFen, n_max_episode_steps)) {

        // Root node:
        top_node = 0;
        beam.insert({{1000.0, 0}, 0});
        QTrainNode* new_node = new QTrainNode(tree, tree.LastIndex(), 0);
        new_node->state = NodeState::FRONTIER;
        nodes.push_back(new_node);
        n_qsa_calls = 0;
    }

    int PopTopPriorityNode() {
        assert(beam.size() > 0);
        int node = beam.rbegin()->second;
        beam.erase(prev(beam.end()));
        assert(nodes[node]->state == NodeState::FRONTIER);
        return node;
    }

    vector<int> BFS(const int start_node) const {
        vector<int> queue {start_node};
        for (int qi = 0; qi < queue.size(); qi++)
            for (auto& kid : nodes[queue[qi]]->kids)
                queue.push_back(kid);
        return queue;
    }

    void ExpandNodeAfterQComputed(const int node, const int& max_depth) {
        // auto& qnode = nodes[node] - lead to error because emplace_back corrupts this pointer
        int node_depth = nodes[node]->depth;

        assert(nodes[node]->state == NodeState::FRONTIER_Q_COMPUTED);
        assert(nodes[node]->moves_estimation.size() > 0);
        assert(nodes[node]->kids.size() == 0);

        vector<int> history_nodes = tree.GetHistoryPathNodes(node);

        for (int ki = 0; ki < nodes[node]->moves_estimation.size(); ki++) {
            auto move = nodes[node]->moves_estimation[ki].move;
            float q_estimation_score = nodes[node]->moves_estimation[ki].score;

            int kid_node = tree.Move(node, move);
            assert(kid_node == nodes.size());

            history_nodes.push_back(kid_node);
            QTrainNode* kid_qnode = new QTrainNode(tree, history_nodes, node_depth + 1);
            history_nodes.pop_back();

            nodes.push_back(kid_qnode);

            optional<int> kid_score = tree.GetRelativePositionScore(kid_node);
            if (kid_score.has_value()) {
                nodes[node]->moves_estimation[ki].score = -kid_score.value();  // overwrite estimation with terminal value
                kid_qnode->state = NodeState::TERMINAL;
            }
            else if (node_depth + 1 < max_depth) {
                // float kid_priority = node_depth == 0 ? 1000.0 : q_estimation_score;    // always expand first turn
                float kid_priority = q_estimation_score;
                beam.insert({{kid_priority, kid_node}, kid_node});
                kid_qnode->state = NodeState::FRONTIER;
            }
            else
                kid_qnode->state = NodeState::FRONTIER;

            nodes[node]->kids.push_back(kid_node);
        }
        nodes[node]->state = NodeState::EXPANDED;
    }

    map<int, float> ComputeLeafValues(ResNet v_model, const at::Device device, const int batch_size) {
        vector<int> bfs = BFS(top_node);

        map<int, float> leaf_values;

        vector<int> to_compute_nodes;
        for (int node : bfs)
            if (nodes[node]->IsLeaf()) {
                if (nodes[node]->v_estimation.has_value())
                    leaf_values[node] = nodes[node]->v_estimation.value();
                else {
                    optional<int> node_score = tree.GetRelativePositionScore(node);
                    if (!node_score.has_value())
                        to_compute_nodes.push_back(node);
                    else
                        leaf_values[node] = node_score.value();
                }
            }

        for (int start = 0; start < to_compute_nodes.size(); start += batch_size) {
            int end = min((int)to_compute_nodes.size(), start + batch_size);

            torch::Tensor input = torch::zeros({end - start, lczero::kInputPlanes, 8, 8});
            auto input_accessor = input.accessor<float, 4>();
            for (int bi = 0; bi < end - start; bi++)
                FillInputTensor(input_accessor, bi, nodes[to_compute_nodes[bi + start]]->input_planes);

            input = input.to(device);
            torch::Tensor predictions = v_model->forward(input);
            predictions = predictions.to(torch::kCPU);
            auto pred_accessor = predictions.accessor<float, 2>();

            for (int bi = 0; bi < end - start; bi++) {
                float predicted_value = pred_accessor[bi][0];

                leaf_values[to_compute_nodes[bi + start]] = predicted_value;
                nodes[to_compute_nodes[bi + start]]->v_estimation = predicted_value;
            }
        }

        return leaf_values;
    }

    void ComputeTreeValues(ResNet v_model, at::Device device, int batch_size) {
        map<int, float> leaf_values = ComputeLeafValues(v_model, device, batch_size);

        auto dfs = [&](auto& self, int node)->float {
            QTrainNode* qnode = nodes[node];

            if (qnode->IsLeaf())
                return leaf_values[node];

            float curr_val = -1000.0;

            assert(qnode->kids.size() == qnode->moves_estimation.size());
            for (int ki = 0; ki < qnode->kids.size(); ki++) {
                float kid_val = self(self, qnode->kids[ki]);
                qnode->moves_estimation[ki].score = -kid_val;
                curr_val = max(curr_val, -kid_val);
            }
            return curr_val;
        };
        dfs(dfs, top_node);
    }

    // Clear memory of all kids of the given `new_top_node`, except kid `keep_kid` and its subtree
    void ClearKids(const int new_top_node, const int keep_kid) {
        vector<int> queue { new_top_node };
        for (int qi = 0; qi < queue.size(); qi++) {
            for (int kid : nodes[queue[qi]]->kids)
                if (kid != keep_kid)
                    queue.push_back(kid);
        }

        for (int qi = 0; qi < queue.size(); qi++)
            if (queue[qi] != new_top_node) {
                delete nodes[queue[qi]];
                nodes[queue[qi]] = nullptr;
            }
    }

    void MoveTopNode(bool exploration_full_random, int exploration_num_first_moves) {
        assert(nodes[top_node]->state == NodeState::EXPANDED);
        auto move = GetMoveWithExploration(nodes[top_node]->moves_estimation, tree.positions[top_node].GetGamePly(), exploration_full_random, exploration_num_first_moves);

        int new_top_node = nodes[top_node]->FindKidByMove(move);
        assert(new_top_node > top_node);
        ClearKids(top_node, new_top_node);

        map<int, float> old_priorities;
        for (const auto& kvp : beam)
            old_priorities[kvp.second] = kvp.first.first;

        beam.clear();

        n_qsa_calls = 0;

        for(const int node : BFS(new_top_node)) {
            if (tree.GetGameResult(node) != lczero::GameResult::UNDECIDED)   // terminal node
                continue;

            if (nodes[node]->state != NodeState::EXPANDED) {
                float priority;
                // if (node == new_top_node || tree.parents[node] == new_top_node)   // always expand first turn
                if (node == new_top_node)
                    priority = 1000;
                else {
                    assert (old_priorities.find(node) != old_priorities.end());
                    priority = old_priorities[node];
                }
                beam.insert({{priority, node}, node});
            }
            else
                n_qsa_calls++;
        }

        top_node = new_top_node;
    }

    void ShowStructure() {
        cout << "EnvExpandState: beam size=" << beam.size()
             << "; n_qsa_calls=" << n_qsa_calls
             << "; top_node=" << top_node << endl;

        cout << "Beam:" << endl;
        for (auto& kvp : beam)
            cout << "  Node=" << kvp.second << "; priority=" << kvp.first.first << endl;

        cout << "Tree (ordered by BFS):" << endl;

        for (int node : BFS(0)) {
            cout << "  Node=" << node << "; depth=" << nodes[node]->depth << "; kids:" << endl;
            nodes[node]->ShowKidsInfo();
        }
    }

    void TestCorrectness() {
        assert(tree.positions.size() > 0);

        vector<vector<int>> computed_kids(tree.positions.size());
        for (int node = 0; node < tree.positions.size(); node++)
            if (tree.parents[node] >= 0)
                computed_kids[tree.parents[node]].push_back(node);

        auto dfs_test = [&](auto& self, int node)->float {
            optional<int> node_score = tree.GetRelativePositionScore(node);
            float expected_score;

            if (node_score.has_value())
                expected_score = node_score.value();
            else if (computed_kids[node].size() == 0) {
                assert(nodes[node]->kids.size() == 0);
                assert(nodes[node]->v_estimation.has_value());
                expected_score = nodes[node]->v_estimation.value();
            }
            else {
                assert(computed_kids[node].size() > 0);
                assert(nodes[node]->kids.size() == computed_kids[node].size());
                assert(nodes[node]->moves_estimation.size() == computed_kids[node].size());

                expected_score = -1000;
                for (int ki = 0; ki < computed_kids[node].size(); ki++) {
                    int kid_node = computed_kids[node][ki];

                    float kid_expected_score = self(self, kid_node);
                    float kid_actual_score = -nodes[node]->moves_estimation[ki].score;
                    // cout << " inside " << ki << " " << kid_expected_score << " " << kid_actual_score << endl;
                    assert(abs(kid_actual_score - kid_expected_score) < 1e-5);

                    expected_score = max(expected_score, -kid_actual_score);
                }
            }
            return expected_score;
        };

        for (int ki = 0; ki < nodes[top_node]->kids.size(); ki++) {
            int top_kid_node = nodes[top_node]->kids[ki];
            float actual = nodes[top_node]->moves_estimation[ki].score;
            float expected = -dfs_test(dfs_test, top_kid_node);
            // cout << ki << " " << actual << " " << expected << endl;
            assert(abs(actual - expected) < 1e-5);
        }
    }
};


QDataset GetQDataset(ResNet v_model, ResNet q_model, at::Device& device, const ConfigParser& config_parser, const int n_games) {
    torch::NoGradGuard no_grad;
    v_model->eval();
    q_model->eval();

    int batch_size = config_parser.GetInt("training.q_self_play_batch_size", true, 0);
    int n_max_episode_steps = config_parser.GetInt("env.n_max_episode_steps", true, 0);
    double dataset_drop_ratio = config_parser.GetDouble("training.dataset_drop_ratio", false, 0);
    int exploration_num_first_moves = config_parser.GetInt("training.exploration_num_first_moves", true, 0);
    bool exploration_full_random = config_parser.GetInt("training.exploration_full_random", false, 0) > 0;
    int tree_num_q_s_a_calls = config_parser.GetInt("training.tree_num_q_s_a_calls", true, 0);
    int tree_max_depth = config_parser.GetInt("training.tree_max_depth", true, 0);
    bool is_test = config_parser.GetInt("training.is_test", false, 0) > 0;

    int game_idx = 0;
    map<string, vector<long long>> stats;
    QDataset rows;
    vector<vector<pair<int, int>>> tree_rows;  // env_index -> [(row_idx, node)]
    // UsageCounter usage;

    vector<EnvExpandState*> envs;

    while (game_idx < n_games || envs.size() > 0) {

        if (envs.size() < batch_size && game_idx < n_games) {
            EnvExpandState* env = new EnvExpandState(lczero::ChessBoard::kStartposFen, n_max_episode_steps);
            env->node_going_to_dataset = true;
            envs.push_back(env);
            tree_rows.push_back({});
            game_idx++;
        }
        else {
            vector<int> nodes;
            vector<QTrainNode*> qnodes;

            // usage.MarkCheckpoint("1. Init");

            for (int ei = 0; ei < envs.size(); ei++) {
                int node = envs[ei]->PopTopPriorityNode();
                assert(envs[ei]->tree.GetGameResult(node) == lczero::GameResult::UNDECIDED);

                envs[ei]->nodes[node]->AddValidMoves(envs[ei]->tree.node_valid_moves[node]);

                nodes.push_back(node);
                qnodes.push_back(envs[ei]->nodes[node]);
            }

            // usage.MarkCheckpoint("2. Pop nodes");

            GetQModelEstimation_Nodes(qnodes, q_model, device);

            // usage.MarkCheckpoint("3. Q estimate");

            for (QTrainNode* qnode : qnodes) {
                assert(qnode->state == NodeState::FRONTIER_MOVES_COMPUTED);
                qnode->state = NodeState::FRONTIER_Q_COMPUTED;
            }

            for (int ei = 0; ei < envs.size(); ei++)
                envs[ei]->ExpandNodeAfterQComputed(nodes[ei], tree_max_depth);

            // usage.MarkCheckpoint("4. Expand");

            for (int ei = envs.size() - 1; ei >= 0; ei--) {
                envs[ei]->n_qsa_calls++;
                
                if (envs[ei]->beam.size() > 0 && envs[ei]->n_qsa_calls < tree_num_q_s_a_calls)    // need expand more
                    continue;

                // usage.MarkCheckpoint("5. Pre compute V");

                envs[ei]->ComputeTreeValues(v_model, device, batch_size);

                // usage.MarkCheckpoint("6. Compute V");

                while (true) {
                    int prev_top_node = envs[ei]->top_node;
                    assert(envs[ei]->tree.GetGameResult(prev_top_node) == lczero::GameResult::UNDECIDED);

                    if (envs[ei]->node_going_to_dataset) {
                        QTrainNode* qnode = envs[ei]->nodes[prev_top_node];

                        tree_rows[ei].push_back({ rows.size(), prev_top_node });
                        rows.push_back({});
                        assert(qnode->input_planes.size() == lczero::kInputPlanes);
                        rows.back().input_planes = qnode->input_planes;
                        rows.back().transform = qnode->transform;
                        rows.back().target = qnode->moves_estimation;
                    }

                    int n_subtree_nodes_before_move = 0;
                    if (is_test) {
                        n_subtree_nodes_before_move = envs[ei]->BFS(prev_top_node).size();
                        envs[ei]->TestCorrectness();
                        for (auto [row_idx, node] : tree_rows[ei]) {
                            assert(envs[ei]->nodes[node]->kids.size() > 0);
                            assert(envs[ei]->nodes[node]->kids.size() == rows[row_idx].target.size());
                            for (int ki = 0; ki < envs[ei]->nodes[node]->kids.size(); ki++) {
                                assert(envs[ei]->nodes[node]->moves_estimation[ki].move == rows[row_idx].target[ki].move);
                                assert(abs(envs[ei]->nodes[node]->moves_estimation[ki].score - rows[row_idx].target[ki].score) < 1e-5);
                            }
                        }
                    } 

                    envs[ei]->MoveTopNode(exploration_full_random, exploration_num_first_moves);
                    envs[ei]->node_going_to_dataset = rand() % 1000000 > dataset_drop_ratio * 1000000;

                    if (is_test) {
                        int n_subtree_nodes_after_move = envs[ei]->BFS(envs[ei]->top_node).size();
                        stats["subtree_size"].push_back(n_subtree_nodes_before_move);
                        stats["reused_size"].push_back(n_subtree_nodes_after_move);
                    }

                    auto game_result = envs[ei]->tree.GetGameResult(envs[ei]->top_node);

                    if (game_result != lczero::GameResult::UNDECIDED) {
                        stats["tree_size"].push_back(envs[ei]->nodes.size());
                        // cout << "Env finished, dataset size " << rows.size() << endl;

                        if (ei < envs.size() - 1) {
                            swap(envs[ei], envs[envs.size() - 1]);
                            swap(tree_rows[ei], tree_rows[tree_rows.size() - 1]);   // TODO: optimize
                        }

                        for (QTrainNode* qnode : envs.back()->nodes)
                            if (qnode != nullptr)
                                delete qnode;
                        delete envs.back();
                        envs.pop_back();
                        tree_rows.pop_back();
                        break;
                    }

                    if (envs[ei]->beam.size() > 0)
                        break;
                    // if beam is empty - play until game is decided and add all moves to the dataset
                }

                // usage.MarkCheckpoint("7. Finalizing");
            }
        }
    }
    assert(envs.size() == 0);
    if (is_test)
        cout << "Q train ok" << endl;

    // usage.PrintStats();

    for (auto& kvp : stats) {
        long long cnt = kvp.second.size();
        long long sm = 0; for (auto val : kvp.second) sm += val;
        long long mx = 0; for (auto val : kvp.second) mx = max(mx, val);
        cout << "Stat " << kvp.first << ": count=" << cnt << "; mean=" << (double)sm / cnt << "; max=" << mx << endl;
    }
    return rows;
}


void TrainQ(const ConfigParser& config_parser, ofstream& losses_file, ResNet q_model, at::Device& device, torch::optim::AdamW& q_optimizer, QDataset& q_dataset) {
    q_model->train();

    int dataset_size = q_dataset.size();
    cout << "[Train.Q] Train Q model on dataset with " << dataset_size << " rows" << endl;

    int batch_size = config_parser.GetInt("training.batch_size", true, 0);

    vector<int> indices(dataset_size);
    for (int i = 0; i < dataset_size; i++) indices[i] = i;
    for (int i = 0; i < dataset_size; i++) swap(indices[i], indices[i + rand() % (dataset_size - i)]);

    for (int end = batch_size; end <= dataset_size; end += batch_size) {
        // torch::Tensor target = torch::zeros({batch_size, 1});
        torch::Tensor input = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});
        auto input_accessor = input.accessor<float, 4>();

        torch::Tensor target_vals = torch::zeros({batch_size, lczero::kNumOutputPolicyFilters, 8, 8});
        auto target_vals_accessor = target_vals.accessor<float, 4>();

        torch::Tensor target_mask = torch::zeros({batch_size, lczero::kNumOutputPolicyFilters, 8, 8});
        auto target_mask_accessor = target_mask.accessor<float, 4>();

        int actions_mask_norm = 0;

        for (int ri = end - batch_size; ri < end; ri++) {
            int bi = ri - end + batch_size;
            int row_i = indices[ri];

            for (const auto& move_target : q_dataset[row_i].target) {
                int move_idx = move_target.move.as_nn_index(q_dataset[row_i].transform);
                int policy_idx = move_to_policy_idx_map[move_idx];
                int displacement = policy_idx / 64;
                int square = policy_idx % 64;
                int row = square / 8;
                int col = square % 8;
                target_vals_accessor[bi][displacement][row][col] = move_target.score;
                target_mask_accessor[bi][displacement][row][col] = 1;
                actions_mask_norm++;
            }

            FillInputTensor(input_accessor, bi, q_dataset[row_i].input_planes);
        }

        // cout << "Input: " << DebugString(input) << endl;
        // cout << "TargetVals: " << DebugString(target_vals) << endl;
        // cout << "TargetMask: " << DebugString(target_mask) << endl;

        input = input.to(device);
        target_vals = target_vals.to(device);
        target_mask = target_mask.to(device);

        q_optimizer.zero_grad();

        torch::Tensor q_prediction = q_model->forward(input);
        // cout << "Prediction: " << DebugString(q_prediction.to(torch::kCPU)) << endl;

        torch::Tensor loss = torch::mse_loss(q_prediction, target_vals, at::Reduction::None);
        // cout << "Loss: " << DebugString(loss) << endl;

        loss = torch::sum(loss * target_mask) / actions_mask_norm;

        // cout << "Loss: " << DebugString(loss) << endl;

        loss.backward();

        q_optimizer.step();
        losses_file << "QLoss: " << loss.item<float>() << endl;
    }
}



}  // namespace probs
