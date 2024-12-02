#include "training/q_train.h"

using namespace std;


namespace probs {


struct QTrainNode {
    vector<int> kids;
    vector<MoveEstimation> moves;
    int depth;          // depth from top node

    int transform;
    lczero::InputPlanes input_planes;

    bool valid_moves_computed;
    vector<MoveEstimation> moves_estimation;

    QTrainNode(const lczero::PositionHistory& lchistory, int depth) : depth(depth), valid_moves_computed(false) {
        input_planes = Encode(lchistory, &transform);
    }

    void ComputeValidMoves(const lczero::Position& lcposition) {
        assert(valid_moves_computed == false);
        assert(moves_estimation.size() == 0);
        valid_moves_computed = true;
        for (auto move : lcposition.GetBoard().GenerateLegalMoves())
            moves_estimation.push_back({move, 0.0f});
    }

    void AppendKid(int kid, lczero::Move move, float score) {
        kids.push_back(kid);
        moves.push_back({move, score});
    }

    int FindKidByMove(const lczero::Move move) const {
        assert(kids.size() == moves.size());
        for (int ki = 0; ki < kids.size(); ki++)
            if (moves[ki].move == move)
                return kids[ki];
        return -1;
    }

    bool IsLeaf() const {
        return kids.size() == 0;
    }

    void ShowKidsInfo() const {
        assert(kids.size() == moves.size());
        for (int ki = 0; ki < kids.size(); ki++)
            cout << "     Kid=" << kids[ki] << "; move=" << moves[ki].move.as_string() << "; " << moves[ki].score << endl;
    }
};


struct EnvExpandState {
    PositionHistoryTree tree;
    int top_node;
    map<pair<float, int>, int> beam;        // (priority, node) -> node
    vector<QTrainNode> nodes;
    int expand_tree_size;
    int n_qsa_calls;

    EnvExpandState(string start_pos, int n_max_episode_steps)
            : tree(PositionHistoryTree(lczero::ChessBoard::kStartposFen, n_max_episode_steps)) {

        // Root node:
        top_node = 0;
        beam.insert({{1000.0, 0}, 0});
        nodes.push_back(QTrainNode(tree.ToLczeroHistory(-1), 0));
        expand_tree_size = 1;
        n_qsa_calls = 0;
    }

    int PopTopPriorityNode() {
        assert(beam.size() > 0);
        int node = beam.rbegin()->second;
        beam.erase(prev(beam.end()));
        return node;
    }

    vector<int> BFS(const int start_node) const {
        vector<int> queue {start_node};
        for (int qi = 0; qi < queue.size(); qi++)
            for (auto& kid : nodes[queue[qi]].kids)
                queue.push_back(kid);
        return queue;
    }

    void AppendChildren(const int node, const int& max_depth) {
        auto& qnode = nodes[node];
        int node_depth = qnode.depth;

        assert(qnode.valid_moves_computed == true);

        for (int mi = 0; mi < qnode.moves_estimation.size(); mi++) {
            auto move = qnode.moves_estimation[mi].move;
            float score = qnode.moves_estimation[mi].score;

            int kid_node = tree.Move(node, move);

            if (node_depth + 1 < max_depth) {
                float kid_priority = node_depth == 0 ? 1000.0 : score;    // always expand first turn
                // float kid_priority = score;
                beam.insert({{kid_priority, kid_node}, kid_node});
            }

            assert(kid_node == nodes.size());
            nodes.push_back(QTrainNode(tree.ToLczeroHistory(kid_node), node_depth + 1));

            nodes[node].AppendKid(kid_node, move, score);

            expand_tree_size++;
        }
    }

    map<int, float> ComputeLeafValues(ResNet v_model, const at::Device device, const int batch_size) {
        vector<int> bfs = BFS(top_node);

        vector<int> leaf_nodes;
        for (int node : bfs)
            if (nodes[node].IsLeaf())
                leaf_nodes.push_back(node);

        map<int, float> leaf_values;
        for (int start = 0; start < leaf_nodes.size(); start += batch_size) {
            int end = min((int)leaf_nodes.size(), start + batch_size);

            torch::Tensor input = torch::zeros({end - start, lczero::kInputPlanes, 8, 8});
            for (int bi = 0; bi < end - start; bi++)
                for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
                    const auto& plane = nodes[leaf_nodes[start + bi]].input_planes[pi];
                    for (auto bit : lczero::IterateBits(plane.mask))
                        input[bi][pi][bit / 8][bit % 8] = plane.value;
                }
            input = input.to(device);
            torch::Tensor predictions = v_model->forward(input);
            predictions = predictions.contiguous();
            predictions = predictions.to(torch::kCPU);
            vector<float> pred_values(predictions.data_ptr<float>(), predictions.data_ptr<float>() + predictions.numel());   // TODO: remove and test speed
            assert(pred_values.size() == end - start);

            for (int bi = 0; bi < end - start; bi++)
                leaf_values[leaf_nodes[bi + start]] = pred_values[bi];
        }
        return leaf_values;
    }

    void ComputeTreeValues(ResNet v_model, at::Device device, int batch_size) {
        map<int, float> leaf_values = ComputeLeafValues(v_model, device, batch_size);

        auto dfs = [&](auto& self, int node)->float {

            auto& qnode = nodes[node];

            if (qnode.IsLeaf())
                return leaf_values[node];

            float curr_val = -1000.0;

            assert(qnode.kids.size() == qnode.moves.size());
            for (int ki = 0; ki < qnode.kids.size(); ki++) {
                float kid_val = self(self, qnode.kids[ki]);
                qnode.moves[ki].score = kid_val;
                curr_val = max(curr_val, -kid_val);
            }
            return curr_val;
        };
        dfs(dfs, top_node);
    }

    void Move(bool exploration_full_random, int exploration_num_first_moves) {
        auto move = GetMoveWithExploration(nodes[top_node].moves, tree.positions[top_node].GetGamePly(), exploration_full_random, exploration_num_first_moves);

        int new_top_node = nodes[top_node].FindKidByMove(move);
        assert(new_top_node > top_node);

        vector<int> new_tree_nodes_arr = BFS(new_top_node);

        if (nodes[new_top_node].kids.size() == 0) {
            assert(new_tree_nodes_arr.size() == 1);
            beam.clear();
            beam.insert({{1000, new_top_node}, new_top_node});
        }
        else {
            set<int> new_tree_nodes_set(begin(new_tree_nodes_arr), end(new_tree_nodes_arr));

            map<pair<float, int>, int> new_beam;
            for (const auto& kvp : beam) {
                float priority = kvp.first.first;
                int node = kvp.second;
                if (new_tree_nodes_set.find(node) != new_tree_nodes_set.end()) {
                    if (tree.parents[node] == new_top_node)
                        priority = 1000;
                    new_beam.insert({{priority, node}, node});
                }
            }
            beam = new_beam;
        }

        top_node = new_top_node;

        expand_tree_size = new_tree_nodes_arr.size();

        n_qsa_calls = 0;
        // for (int node : new_tree_nodes_arr)
        //     if (!nodes[node].IsLeaf())
        //         n_qsa_calls++;
    }

    void ShowStructure() {
        cout << "EnvExpandState: beam size=" << beam.size()
             << "; expand_tree_size=" << expand_tree_size
             << "; n_qsa_calls=" << n_qsa_calls
             << "; top_node=" << top_node << endl;

        cout << "Beam:" << endl;
        for (auto& kvp : beam)
            cout << "  Node=" << kvp.second << "; priority=" << kvp.first.first << endl;

        cout << "Tree (ordered by BFS):" << endl;

        for (int node : BFS(0)) {
            cout << "  Node=" << node << "; depth=" << nodes[node].depth << "; kids:" << endl;
            nodes[node].ShowKidsInfo();
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
    map<string, vector<long long>> stats;
    QDataset rows;

    vector<EnvExpandState*> envs;
    vector<PositionHistoryTree*> trees;

    while (game_idx < n_games || envs.size() > 0) {

        if (envs.size() < batch_size && game_idx < n_games) {
// cout << 1 << endl;
            EnvExpandState* env = new EnvExpandState(lczero::ChessBoard::kStartposFen, n_max_episode_steps);
            envs.push_back(env);
            trees.push_back(&envs.back()->tree);
            game_idx++;
        }

        else {
// cout << 2 << endl;
            vector<int> nodes;
            vector<QTrainNode*> qnodes;

            for (int ei = 0; ei < envs.size(); ei++) {
                int node = envs[ei]->PopTopPriorityNode();
                nodes.push_back(node);

                envs[ei]->nodes[node].ComputeValidMoves(envs[ei]->tree.positions[node]);
                qnodes.push_back(&(envs[ei]->nodes[node]));
            }
// cout << 3 << endl;

            GetQModelEstimation_Nodes(qnodes, q_model, device);

            for (int ei = 0; ei < envs.size(); ei++)
                envs[ei]->AppendChildren(nodes[ei], tree_max_depth);

            for (int ei = envs.size() - 1; ei >= 0; ei--) {
// cout << 4 << endl;
                envs[ei]->n_qsa_calls++;
                
                if (envs[ei]->beam.size() > 0 && envs[ei]->n_qsa_calls < tree_num_q_s_a_calls)    // need expand more
                    continue;

                envs[ei]->ComputeTreeValues(v_model, device, batch_size);

                if (rand() % 1000000 > dataset_drop_ratio * 1000000) {
                    int top_node = envs[ei]->top_node;
                    auto& qnode = envs[ei]->nodes[top_node];

                    rows.push_back({});
                    assert(qnode.input_planes.size() == lczero::kInputPlanes);
                    assert(qnode.valid_moves_computed == true);
                    rows.back().input_planes = qnode.input_planes;
                    rows.back().transform = qnode.transform;
                    rows.back().target = qnode.moves_estimation;
// cout << 5 << " dataset size " << rows.size() << endl;
                }

                auto game_result = envs[ei]->tree.GetGameResult(envs[ei]->top_node);   // TODO: ComputeGameResult can reuse computed valid moves

                if (game_result == lczero::GameResult::UNDECIDED) {
// cout << 6 << endl;
                    envs[ei]->Move(exploration_full_random, exploration_num_first_moves);
                }
                else {
// cout << 7 << endl;
                    stats["tree_size"].push_back(envs[ei]->nodes.size());
cout << "Env finished, dataset size " << rows.size() << endl;

                    if (ei < envs.size() - 1) {
                        swap(envs[ei], envs[envs.size() - 1]);
                        swap(trees[ei], trees[trees.size() - 1]);
                    }
                    envs.pop_back();
                    trees.pop_back();
                }
            }
        }
    }

    for (auto& kvp : stats) {
        long long cnt = kvp.second.size();
        long long sm = 0; for (auto val : kvp.second) sm += val;
        long long mx = 0; for (auto val : kvp.second) mx = max(mx, val);
        cout << "Stat " << kvp.first << ": count=" << cnt << "; mean=" << (double)sm / cnt << "; max=" << mx << endl;
    }

    for (EnvExpandState* env : envs)
        delete env;

    return rows;
}


void TrainQ(const ConfigParser& config_parser, ResNet q_model, at::Device& device, torch::optim::AdamW& q_optimizer, QDataset& q_dataset) {
    q_model->train();

    int dataset_size = q_dataset.size();
    cout << "[Train.Q] Train Q model on dataset with " << dataset_size << " rows" << endl;

    int batch_size = config_parser.GetInt("training.batch_size");

    vector<int> indices(dataset_size);
    for (int i = 0; i < dataset_size; i++) indices[i] = i;
    for (int i = 0; i < dataset_size; i++) swap(indices[i], indices[i + rand() % (dataset_size - i)]);

    for (int end = batch_size; end <= dataset_size; end += batch_size) {
        // torch::Tensor target = torch::zeros({batch_size, 1});
        torch::Tensor input = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});
        torch::Tensor target_vals = torch::zeros({batch_size, lczero::kNumOutputPolicyFilters, 8, 8});
        torch::Tensor target_mask = torch::zeros({batch_size, lczero::kNumOutputPolicyFilters, 8, 8});
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
                target_vals[bi][displacement][row][col] = move_target.score;
                target_mask[bi][displacement][row][col] = 1;
                actions_mask_norm++;
            }

            for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
                const lczero::InputPlane& plane = q_dataset[row_i].input_planes[pi];
                for (auto bit : lczero::IterateBits(plane.mask)) {
                    input[bi][pi][bit / 8][bit % 8] = plane.value;
                }
            }
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
        cout << "QLoss: " << loss.item<float>() << endl;
    }
}



}  // namespace probs
