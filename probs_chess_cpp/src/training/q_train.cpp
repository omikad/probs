#include "training/q_train.h"

using namespace std;


namespace probs {


QDatasetRow::QDatasetRow(const EncodedPositionBatch& node_encoded, const std::vector<MoveEstimation>& moves_estimation) {
    assert(node_encoded.planes.size() == 1);
    input_planes = node_encoded.planes[0];
    transform = node_encoded.transforms[0];
    target = moves_estimation;
}


struct QTrainNode {
    vector<int> kids;
    vector<MoveEstimation> moves;
    int depth;          // depth from top node

    QTrainNode(int depth) : depth(depth) {}

    void AppendKid(int kid, lczero::Move move, float score) {
        kids.push_back(kid);
        moves.push_back({move, score});
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
    EnvPlayer env;
    int top_node;
    map<pair<float, int>, int> beam;        // (priority, node) -> node
    vector<QTrainNode> nodes;
    int expand_tree_size;
    int n_qsa_calls;

    EnvExpandState(string start_pos, int n_max_episode_steps)
            : env(EnvPlayer(lczero::ChessBoard::kStartposFen, n_max_episode_steps)) {

        // Root node:
        top_node = 0;
        beam.insert({{1000.0, 0}, 0});
        nodes.push_back(QTrainNode(0));
        expand_tree_size = 1;
        n_qsa_calls = 0;
    }

    int GetTopPriorityNode() const {
        return beam.rbegin()->second;
    }

    vector<int> BFS(const int start_node) const {
        vector<int> queue {start_node};
        for (int qi = 0; qi < queue.size(); qi++)
            for (auto& kid : nodes[queue[qi]].kids)
                queue.push_back(kid);
        return queue;
    }

    void AppendChildren(const int node, const vector<MoveEstimation>& moves_estimation, const int& max_depth) {
        int node_depth = nodes[node].depth;

        for (int mi = 0; mi < moves_estimation.size(); mi++) {
            auto move = moves_estimation[mi].move;
            float score = moves_estimation[mi].score;

            int kid_node = env.Tree().Append(node, move);

            if (node_depth + 1 < max_depth) {
                float kid_priority = node_depth == 0 ? 1000.0 : score;    // always expand first turn
                // float kid_priority = score;
                beam.insert({{kid_priority, kid_node}, kid_node});
            }

            nodes.push_back(QTrainNode(node_depth + 1));

            nodes[node].AppendKid(kid_node, move, score);

            expand_tree_size++;
        }
    }


    map<int, float> ComputeLeafValues(const vector<int>& bfs, const PositionHistoryTree& tree, ResNet v_model, const at::Device device, const int batch_size) const {
        vector<int> leaf_nodes;
        vector<lczero::InputPlanes> leaf_input_planes;
        for (int node : bfs)
            if (nodes[node].IsLeaf()) {
                leaf_nodes.push_back(node);
                int transform_out;
                leaf_input_planes.push_back(Encode(tree.ToLczeroHistory(node), &transform_out));
            }

        map<int, float> leaf_values;
        for (int start = 0; start < leaf_nodes.size(); start += batch_size) {
            int end = min((int)leaf_nodes.size(), start + batch_size);

            torch::Tensor input = torch::zeros({end - start, lczero::kInputPlanes, 8, 8});
            for (int bi = 0; bi < end - start; bi++)
                for (int pi = 0; pi < lczero::kInputPlanes; pi++) {
                    const auto& plane = leaf_input_planes[start + bi][pi];
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


    void RecomputeMovesEstimation(const PositionHistoryTree& tree, ResNet v_model, at::Device device, int batch_size) {
        vector<int> bfs = BFS(top_node);

        map<int, float> leaf_values = ComputeLeafValues(bfs, tree, v_model, device, batch_size);

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
        auto move = GetMoveWithExploration(nodes[top_node].moves, env.Tree().positions[top_node].GetGamePly(), exploration_full_random, exploration_num_first_moves);

        // TODO: reuse previous computations

        set<int> removed_nodes;

        while (top_node < env.Tree().positions.size() - 1) {
            removed_nodes.insert(env.Tree().LastIndex());
            nodes.pop_back();
        }
        assert(top_node == env.Tree().LastIndex());
        assert(top_node == nodes.size() - 1);

        for (int node = 0; node < nodes.size(); node++) {
            auto& qnode = nodes[node];
            for (int ki = qnode.kids.size() - 1; ki >= 0; ki--) {
                int kid_node = qnode.kids[ki];
                if (removed_nodes.find(kid_node) != removed_nodes.end()) {
                    int j = qnode.kids.size() - 1;
                    swap(qnode.kids[ki], qnode.kids[j]);
                    swap(qnode.moves[ki], qnode.moves[j]);
                    qnode.kids.pop_back();
                    qnode.moves.pop_back();
                }
            }
        }

        env.Move(move);

        int new_top_node = env.Tree().LastIndex();

        nodes[top_node].AppendKid(new_top_node, move, 0);
        nodes.push_back(QTrainNode(0));

        top_node = new_top_node;
        beam.clear();
        beam.insert({{1000.0, top_node}, top_node});
        expand_tree_size = 1;
        n_qsa_calls = 0;
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
    QDataset rows;

    vector<shared_ptr<EnvExpandState>> envs;

    while (game_idx < n_games || envs.size() > 0) {
cout << 1 << endl;

        if (envs.size() < batch_size && game_idx < n_games) {
cout << 2 << endl;
            envs.push_back(make_shared<EnvExpandState>(EnvExpandState(lczero::ChessBoard::kStartposFen, n_max_episode_steps)));
            game_idx++;
        }

        else {
cout << 3 << endl;

            vector<PositionHistoryTree*> trees;
            vector<int> nodes;
            for (int ei = 0; ei < envs.size(); ei++) {
                trees.push_back(&envs[ei]->env.Tree());
                nodes.push_back(envs[ei]->GetTopPriorityNode());
                envs[ei]->beam.erase(std::prev(envs[ei]->beam.end()));
            }
            auto encoded_batch = GetQModelEstimation(trees, nodes, q_model, device);
cout << 4 << endl;

            for (int ei = 0; ei < envs.size(); ei++)
                envs[ei]->AppendChildren(nodes[ei], encoded_batch->moves_estimation[ei], tree_max_depth);
cout << 5 << endl;

            for (int ei = envs.size() - 1; ei >= 0; ei--) {
cout << 6 << endl;
                envs[ei]->n_qsa_calls++;
                
                if (envs[ei]->n_qsa_calls < tree_num_q_s_a_calls)    // need expand more
                    continue;
cout << 7 << endl;

                envs[ei]->RecomputeMovesEstimation(*trees[ei], v_model, device, batch_size);

                if (rand() % 1000000 > dataset_drop_ratio * 1000000) {
cout << 8 << endl;
                    int top_node = envs[ei]->top_node;
                    auto top_node_encoded = GetQModelEstimation_OneNode(*trees[ei], top_node, q_model, device);       // TODO: it was computed above, need to save it somewhere and reuse here
                    rows.push_back(QDatasetRow(*top_node_encoded, envs[ei]->nodes[top_node].moves));

cout << "dataset rows: " << rows.size() << endl;
                }
cout << 9 << endl;

                auto game_result = envs[ei]->env.GameResult();

                if (game_result == lczero::GameResult::UNDECIDED) {
cout << 10 << endl;
                    envs[ei]->Move(exploration_full_random, exploration_num_first_moves);
                }
                else {
cout << 11 << endl;
                    if (ei < envs.size() - 1) {
                        swap(envs[ei], envs[envs.size() - 1]);
                    }
                    envs.pop_back();
                }
            }
        }
    }

cout << "DONE!" << endl;

    return rows;
}


}  // namespace probs
