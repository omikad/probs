#include "infra/uci_player.h"

using namespace std;

namespace probs {


int UciPlayer::AppendLastTreeNode() {
    int tree_node_idx = (int)tree.positions.size() - 1;
    NodeInfo node;
    for (auto move : tree.node_valid_moves[tree_node_idx]) {
        KidInfo kid_info;
        kid_info.kid_node = -1;
        kid_info.move = move;
        kid_info.q_nn_score = 0;
        kid_info.q_tree_score = 0;
        node.kids.push_back(kid_info);
    }
    if (node.kids.size() == 0) {
        node.is_terminal = true;
        auto score = tree.GetRelativePositionScore(tree_node_idx);
        node.v_tree_score = score.has_value() ? 0 : score.value();
    }
    else {
        node.is_terminal = false;
        node.v_tree_score = -1000;
    }
    nodes.push_back(node);
    assert(nodes.size() == tree.positions.size());
    return tree_node_idx;
}


UciPlayer::UciPlayer(ConfigParser& config)
        : n_max_episode_steps(config.GetInt("env.n_max_episode_steps", true, 0))
        , q_agent_batch_size(config.GetInt("infra.q_agent_batch_size", true, 0))
        , search_mode(config.GetInt("infra.search_mode", false, 2))
        , tree(lczero::ChessBoard::kStartposFen, n_max_episode_steps)
        , is_in_search(false)
        , stop_search_flag(false)
        , debug_on(false)
        , q_model(nullptr)
        , device(torch::kCPU) {

    ModelKeeper model_keeper(config, "model");
    device = GetDeviceFromConfig(config);
    model_keeper.To(device);
    model_keeper.SetEvalMode();
    q_model = model_keeper.q_model;

    top_node = AppendLastTreeNode();
}


void UciPlayer::OnNewGame() {
    assert(is_in_search == false);
}


int UciPlayer::FindKidIndex(const int node, const lczero::Move move) const {
    for (int ki = 0; ki < (int)nodes[node].kids.size(); ki++)
        if (nodes[node].kids[ki].move == move)
            return ki;
    return -1;
}


void UciPlayer::SetPosition(const string& starting_fen, const vector<string>& moves) {
    assert(is_in_search == false);

    bool continuation = false;
    if (last_pos_fen == starting_fen && last_pos_moves.size() < moves.size()) {
        bool eq = true;
        for (int i = 0; i < (int)last_pos_moves.size(); i++)
            if (moves[i] != last_pos_moves[i]) {
                eq = false;
                break;
            }

        if (eq)
            continuation = true;
    }

    if (!continuation) {
        last_pos_fen = starting_fen;
        last_pos_moves.clear();
        nodes.clear();
        tree = PositionHistoryTree(starting_fen, n_max_episode_steps);
        top_node = AppendLastTreeNode();
        assert(top_node == 0);
    }

    for (int i = last_pos_moves.size(); i < (int)moves.size(); i++) {
        assert(nodes[top_node].is_terminal == false);

        auto& move_str = moves[i];
        auto move = lczero::Move(move_str, tree.positions[top_node].IsBlackToMove());
        move = tree.positions[top_node].GetBoard().GetModernMove(move);

        int ki = FindKidIndex(top_node, move);
        if (ki < 0) {
            cerr << move_str << endl;
            cerr << "tree:"; for (auto& kid_move : tree.node_valid_moves[top_node]) cerr << " " << kid_move.as_string(); cerr << endl;
            cerr << "kids:"; for (auto& kid : nodes[top_node].kids) cerr << " " << kid.move.as_string(); cerr << endl;
        }
        assert(ki >= 0);

        int new_top_node;
        if (nodes[top_node].kids[ki].kid_node < 0) {
            tree.Move(top_node, move);
            new_top_node = AppendLastTreeNode();
            nodes[top_node].kids[ki].kid_node = new_top_node;
        }
        else {
            new_top_node = nodes[top_node].kids[ki].kid_node;
        }

        top_node = new_top_node;
        last_pos_moves.push_back(move_str);
    }

    if (debug_on)
        cerr << "[DEBUG] Last position:\n" << tree.positions[top_node].DebugString() << endl;
}


void UciPlayer::Stop() {
    stop_search_flag = true;
}


void UciPlayer::EstimateNodesActions(const vector<int>& nodes_to_estimate) {
    int batch_size = nodes_to_estimate.size();

    vector<int> transform_out(batch_size);
    vector<lczero::InputPlanes> planes(batch_size);

    vector<int> history_nodes;
    for (int bi = 0; bi < batch_size; bi++) {
        int node = nodes_to_estimate[bi];
        if (bi == 0)
            history_nodes = tree.GetHistoryPathNodes(node);
        else {
            // assuming all nodes are from subtree of top_node - recompute only subtree part
            int ply = tree.positions[node].GetGamePly();
            while (history_nodes.size() > ply + 1) history_nodes.pop_back();
            while (history_nodes.size() < ply + 1) history_nodes.push_back(0);

            int j = ply;
            int runnode = node;
            while (runnode != top_node) {
                assert(runnode >= 0);
                assert(j >= 0);
                history_nodes[j] = runnode;
                runnode = tree.parents[runnode];
                j--;
            }
        }

        planes[bi] = Encode(tree, history_nodes, &transform_out[bi]);
    }

    torch::Tensor input = torch::zeros({batch_size, lczero::kInputPlanes, 8, 8});
    auto input_accessor = input.accessor<float, 4>();

    for (int bi = 0; bi < batch_size; bi++)
        FillInputTensor(input_accessor, bi, planes[bi]);

    input = input.to(device);
    torch::Tensor q_values = q_model->forward(input);
    q_values = q_values.to(torch::kCPU);
    auto q_values_accessor = q_values.accessor<float, 4>();

    for (int bi = 0; bi < batch_size; bi++) {
        int node = nodes_to_estimate[bi];
        for (auto& kid_info : nodes[node].kids) {
            auto move = kid_info.move;
            int move_idx = move.as_nn_index(transform_out[bi]);
            int policy_idx = move_to_policy_idx_map[move_idx];
            int displacement = policy_idx / 64;
            int square = policy_idx % 64;
            int row = square / 8;
            int col = square % 8;
            float score = q_values_accessor[bi][displacement][row][col];

            kid_info.q_nn_score = score;
        }
    }
}


lczero::Move UciPlayer::GoSearch__Random() {
    int ki = rand() % nodes[top_node].kids.size();
    auto best_move = nodes[top_node].kids[ki].move;
    return best_move;
}


lczero::Move UciPlayer::GoSearch__SingleQCall() {
    EstimateNodesActions({top_node});
    int best_ki = 0;
    for (int ki = 0; ki < (int)nodes[top_node].kids.size(); ki++)
        if (nodes[top_node].kids[ki].q_nn_score > nodes[top_node].kids[best_ki].q_nn_score)
            best_ki = ki;
    auto best_move = nodes[top_node].kids[best_ki].move;
    return best_move;
}


lczero::Move UciPlayer::GoSearch__FullSearch() {
    map<pair<float, int>, pair<int, int>> beam; // {(priority, counter) -> (node, kid index)}
    int beam_counter = 0;
    beam.insert({{1000.0, beam_counter++}, {top_node, -1}});

// cerr << "[EXPAND] push top_node=" << top_node << endl;

    for (int step_i = 0 ; step_i < 10; step_i++) {

        vector<int> nodes_to_expand;
        while (nodes_to_expand.size() < q_agent_batch_size && beam.size() > 0) {
            auto [node, ki] = beam.rbegin()->second;
            beam.erase(prev(beam.end()));

// cerr << "[EXPAND] top_node=" << top_node << "; pop node=" << node << "; pop ki=" << ki << endl;

            if (ki < 0) {
                nodes_to_expand.push_back(node);
            }
            else {
                auto& kid = nodes[node].kids[ki];
                if (kid.kid_node < 0) {
                    tree.Move(node, kid.move);
                    kid.kid_node = AppendLastTreeNode();
                }
                nodes_to_expand.push_back(kid.kid_node);
            }
        }
        if (nodes_to_expand.size() == 0) break;

        EstimateNodesActions(nodes_to_expand);

        for (int node : nodes_to_expand) {

            if (!nodes[node].is_terminal) {
                for (int ki = 0; ki < (int)nodes[node].kids.size(); ki++) {
                    auto& kid = nodes[node].kids[ki];
                    beam.insert({{kid.q_nn_score, beam_counter++}, {node, ki}});
                }
            }

            int runnode = node;
            while (true) {
                assert(runnode >= 0);

                float best_kid_score = -1000;
                for (auto& kid_info : nodes[runnode].kids)
                    if (kid_info.kid_node >= 0) {
                        kid_info.q_tree_score = -nodes[kid_info.kid_node].v_tree_score;
                        best_kid_score = max(best_kid_score, kid_info.q_tree_score);
                    }
                    else {
                        kid_info.q_tree_score = kid_info.q_nn_score;
                        best_kid_score = max(best_kid_score, kid_info.q_nn_score);
                    }

// cerr << "[EXPAND] bubble up result node=" << node << "; runnode=" << runnode << "; best_kid_score=" << best_kid_score << endl;

                if (abs(nodes[runnode].v_tree_score - best_kid_score) < 1e-6)
                    break;
                nodes[runnode].v_tree_score = best_kid_score;

                if (runnode == top_node)
                    break;
                runnode = tree.parents[runnode];
            }
        }
    }

    float best_score = -10000000;
    lczero::Move best_move;

    for (auto& kid_info : nodes[top_node].kids)
        if (kid_info.q_tree_score >= best_score) {
            best_score = kid_info.q_tree_score;
            best_move = kid_info.move;
        }
    assert(best_score > -10000000);

    return best_move;
}


void UciPlayer::StartSearch(
        optional<chrono::milliseconds> search_wtime,
        optional<chrono::milliseconds> search_btime,
        optional<chrono::milliseconds> search_winc,
        optional<chrono::milliseconds> search_binc,
        optional<int> search_moves_to_go,
        optional<int> search_depth,
        optional<uint64_t> search_nodes,
        optional<int> search_mate,
        optional<chrono::milliseconds> search_fixed_time,
        bool search_infinite,
        vector<string>& search_moves
    ) {
    is_in_search = true;
    stop_search_flag = false;

    if (debug_on) {
        cerr << "Start search with params:" << endl;
        cerr << "search_wtime=" << (search_wtime.has_value() ? to_string(search_wtime.value().count()) : "null") << endl;
        cerr << "search_btime=" << (search_btime.has_value() ? to_string(search_btime.value().count()) : "null") << endl;
        cerr << "search_winc=" << (search_winc.has_value() ? to_string(search_winc.value().count()) : "null") << endl;
        cerr << "search_binc=" << (search_binc.has_value() ? to_string(search_binc.value().count()) : "null") << endl;
        cerr << "search_moves_to_go=" << (search_moves_to_go.has_value() ? to_string(search_moves_to_go.value()) : "null") << endl;
        cerr << "search_depth=" << (search_depth.has_value() ? to_string(search_depth.value()) : "null") << endl;
        cerr << "search_nodes=" << (search_nodes.has_value() ? to_string(search_nodes.value()) : "null") << endl;
        cerr << "search_mate=" << (search_mate.has_value() ? to_string(search_mate.value()) : "null") << endl;
        cerr << "search_fixed_time=" << (search_fixed_time.has_value() ? to_string(search_fixed_time.value().count()) : "null") << endl;
        cerr << "search_infinite=" << (search_infinite ? "true" : "false") << endl;
        cerr << "search_moves=["; for (auto& move : search_moves) cerr << " " << move; cerr << " ]" << endl;
    }

    assert(!nodes[top_node].is_terminal);

    auto best_move =
        search_mode == 2 ? GoSearch__FullSearch()
        : search_mode == 1 ? GoSearch__SingleQCall()
        : GoSearch__Random();

    auto& top_position = tree.positions[top_node];
    best_move = top_position.GetBoard().GetLegacyMove(best_move);
    if (top_position.IsBlackToMove())
        best_move.Mirror();

    cout << "bestmove " << best_move.as_string() << "\n";
    cout << flush;

    is_in_search = false;
}


void UciPlayer::WaitForReadyState() {
    if (search_future.valid()) {
        search_future.get();
    }
}


void UciPlayer::SetDebug(bool turn_on) {
    debug_on = turn_on;
}


}  // namespace probs
