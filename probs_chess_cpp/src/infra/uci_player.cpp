#include "infra/uci_player.h"

using namespace std;

namespace probs {


int UciPlayer::AppendLastTreeNode() {
    int tree_node_idx = (int)tree.positions.size() - 1;
    NodeInfo node;
    node.q_model_called = false;
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
        node.v_tree_score = score.has_value() ? score.value() : 0;
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

    for (int warmup = 0; warmup < 5; warmup++) {
        EstimateNodesActions({top_node});
        nodes[top_node].q_model_called = false;
    }
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

    continuation = false;           // TODO : use continuation

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
        assert(nodes[node].q_model_called == false);
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

            // TODO: remove 2
            vector<int> assert_history_nodes = tree.GetHistoryPathNodes(node);
            assert(assert_history_nodes.size() == history_nodes.size());
            for (int i = 0; i < (int)assert_history_nodes.size(); i++)
                assert(assert_history_nodes[i] == history_nodes[i]);
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

        float best_kid_score = -1000.0;

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
            kid_info.q_tree_score = score;
            best_kid_score = max(best_kid_score, score);
        }

        nodes[node].q_model_called = true;
        nodes[node].v_tree_score = best_kid_score;
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


const long long LIMIT_TIME_GAP = 20;
struct SearchHelper {

    SearchConstraintsInfo search_info;
    chrono::high_resolution_clock::time_point start_time;
    optional<long long> limit_time;
    long long calls;

    SearchHelper(SearchConstraintsInfo& search_info, bool is_black)
            : search_info(search_info)
            , start_time(chrono::high_resolution_clock::now())
            , limit_time(nullopt)
            , calls(0) {
        if (search_info.fixed_time.has_value())
            limit_time = search_info.fixed_time.value().count();
        else if (!is_black && search_info.wtime.has_value()) {
            int denum = search_info.moves_to_go.has_value() ? search_info.moves_to_go.value() : 100;
            limit_time = search_info.wtime.value().count() / denum;
        }
        else if (is_black && search_info.btime.has_value()) {
            int denum = search_info.moves_to_go.has_value() ? search_info.moves_to_go.value() : 100;
            limit_time = search_info.btime.value().count() / denum;
        }
    }

    bool CheckFlagStop() {
        calls++;
        if (limit_time.has_value()) {
            auto elapsed = GetElapsedTime();
            if (elapsed >= limit_time.value() - LIMIT_TIME_GAP)
                return true;
        }
        return false;
    }

    int64_t GetElapsedTime() {
        const auto time_now = std::chrono::high_resolution_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(time_now - start_time).count();
        return elapsed;
    }

    void PrintSearchStuff() {
        long long elapsed = GetElapsedTime();
        cerr << "Elapsed: " << elapsed << "ms; calls=" << calls << "; ms per call=" << elapsed / calls << endl;
    }
};


lczero::Move UciPlayer::GoSearch__FullSearch() {
    SearchHelper search_helper(search_info, tree.positions[top_node].IsBlackToMove());

    map<pair<float, int>, pair<int, int>> beam; // {(priority, counter) -> (node, kid index)}
    int beam_counter = 0;

    // vector<int> queue({top_node});
    // for (int qi = 0; qi < (int)queue.size(); qi++) {
    //     int node = queue[qi];

    //     if (node == top_node && nodes[node].q_model_called == false) {
    //         beam.insert({{1000.0, beam_counter++}, {top_node, -1}});
    //     }

    //     if (nodes[node].q_model_called)
    //         for (int ki = 0; ki < (int)nodes[node].kids.size(); ki++) {
    //             auto& kid = nodes[node].kids[ki];
    //             if (kid.kid_node < 0)
    //                 beam.insert({{kid.q_nn_score, beam_counter++}, {node, ki}});
    //             else
    //                 queue.push_back(kid.kid_node);
    //         }

    //     else                                        // TODO: remove
    //         for (auto& kid : nodes[node].kids)
    //             assert(kid.kid_node < 0);
    // }
    beam.insert({{1000.0, beam_counter++}, {top_node, -1}});

    while (!search_helper.CheckFlagStop() && beam.size() > 0) {

        vector<int> nodes_to_expand;
        while (nodes_to_expand.size() < q_agent_batch_size && beam.size() > 0) {
            auto [node, ki] = beam.rbegin()->second;
            beam.erase(prev(end(beam)));

            if (ki < 0) {
                nodes_to_expand.push_back(node);
            }
            else {
                auto& kid = nodes[node].kids[ki];
                assert(kid.kid_node < 0);
                tree.Move(node, kid.move);
                kid.kid_node = AppendLastTreeNode();
                if (!nodes[kid.kid_node].is_terminal)
                    nodes_to_expand.push_back(kid.kid_node);
            }
        }
        if (nodes_to_expand.size() == 0) continue;

        EstimateNodesActions(nodes_to_expand);

        for (int node : nodes_to_expand) {
            assert(!nodes[node].is_terminal);

            for (int ki = 0; ki < (int)nodes[node].kids.size(); ki++) {
                auto& kid = nodes[node].kids[ki];
                beam.insert({{kid.q_nn_score, beam_counter++}, {node, ki}});
                assert(kid.kid_node < 0);
            }

            // int runnode = node;
            // while (true) {
            //     assert(runnode >= 0);

            //     if (!nodes[runnode].is_terminal) {
            //         float best_kid_score = -1000;
            //         for (auto& kid : nodes[runnode].kids)
            //             if (kid.kid_node >= 0) {
            //                 kid.q_tree_score = -nodes[kid.kid_node].v_tree_score;
            //                 best_kid_score = max(best_kid_score, kid.q_tree_score);
            //             }
            //             else {
            //                 kid.q_tree_score = kid.q_nn_score;
            //                 best_kid_score = max(best_kid_score, kid.q_nn_score);
            //             }

            //         // if (abs(nodes[runnode].v_tree_score - best_kid_score) < 1e-6)
            //         //     break;
            //         nodes[runnode].v_tree_score = best_kid_score;
            //     }

            //     if (runnode == top_node)
            //         break;
            //     runnode = tree.parents[runnode];
            // }
        }
    }

    {
        vector<int> queue({top_node});
        for (int qi = 0; qi < (int)queue.size(); qi++)
            for (auto& kid : nodes[queue[qi]].kids)
                if (kid.kid_node >= 0)
                    queue.push_back(kid.kid_node);

        for (int qi = (int)queue.size() - 1; qi >= 0; qi--) {
            int node = queue[qi];
            if (!nodes[node].is_terminal) {
                float best_kid_score = -1000;
                for (auto& kid : nodes[node].kids) {
                    if (kid.kid_node >= 0) {
                        kid.q_tree_score = -nodes[kid.kid_node].v_tree_score;
                        assert(kid.q_tree_score >= -10 && kid.q_tree_score <= 10);
                    }
                    else {
                        kid.q_tree_score = kid.q_nn_score;
                        assert(kid.q_tree_score >= -10 && kid.q_tree_score <= 10);
                    }
                    best_kid_score = max(best_kid_score, kid.q_tree_score);
                }
                nodes[node].v_tree_score = best_kid_score;
            }
        }
cerr << "queue.size()=" << queue.size() << endl;
    }

    search_helper.PrintSearchStuff();

    float best_score = -10000000;
    lczero::Move best_move;

    assert(nodes[top_node].q_model_called);
    for (auto& kid_info : nodes[top_node].kids)
        if (kid_info.q_tree_score >= best_score) {
            best_score = kid_info.q_tree_score;
            best_move = kid_info.move;
        }
    assert(best_score > -10000000);

    return best_move;
}


void UciPlayer::StartSearch(SearchConstraintsInfo& _search_info) {
    assert(is_in_search == false);
    is_in_search = true;
    stop_search_flag = false;

    search_info = _search_info;

    if (debug_on) {
        cerr << "Start search with params:" << endl;
        cerr << "search_info.wtime=" << (search_info.wtime.has_value() ? to_string(search_info.wtime.value().count()) : "null") << endl;
        cerr << "search_info.btime=" << (search_info.btime.has_value() ? to_string(search_info.btime.value().count()) : "null") << endl;
        cerr << "search_info.winc=" << (search_info.winc.has_value() ? to_string(search_info.winc.value().count()) : "null") << endl;
        cerr << "search_info.binc=" << (search_info.binc.has_value() ? to_string(search_info.binc.value().count()) : "null") << endl;
        cerr << "search_info.moves_to_go=" << (search_info.moves_to_go.has_value() ? to_string(search_info.moves_to_go.value()) : "null") << endl;
        cerr << "search_info.depth=" << (search_info.depth.has_value() ? to_string(search_info.depth.value()) : "null") << endl;
        cerr << "search_info.nodes=" << (search_info.nodes.has_value() ? to_string(search_info.nodes.value()) : "null") << endl;
        cerr << "search_info.mate=" << (search_info.mate.has_value() ? to_string(search_info.mate.value()) : "null") << endl;
        cerr << "search_info.fixed_time=" << (search_info.fixed_time.has_value() ? to_string(search_info.fixed_time.value().count()) : "null") << endl;
        cerr << "search_info.infinite=" << (search_info.infinite ? "true" : "false") << endl;
        cerr << "search_info.moves=["; for (auto& move : search_info.moves) cerr << " " << move; cerr << " ]" << endl;
    }

    assert(!nodes[top_node].is_terminal);

    search_future = std::async(std::launch::async, [&] {
        auto best_move =
            search_mode == 2 ? GoSearch__FullSearch()
            : search_mode == 1 ? GoSearch__SingleQCall()
            : GoSearch__Random();

        auto& top_position = tree.positions[top_node];
        best_move = top_position.GetBoard().GetLegacyMove(best_move);
        if (top_position.IsBlackToMove())
            best_move.Mirror();

        is_in_search = false;

        cout << "bestmove " << best_move.as_string() << "\n";
        cout << flush;
    });
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
